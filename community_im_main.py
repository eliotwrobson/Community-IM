"""
Main file for running community IM experiments
"""

import heapq
import shelve
import time
import typing as t
from collections import defaultdict

import igraph as ig
import leidenalg as la
import networkx as nx
import tqdm
from cynetdiff.models import DiffusionModel
from cynetdiff.utils import networkx_to_ic_model

import dataset_manager as dm

DictPartition = dict[int, list[int]]

CACHE_FILE_NAME = "cache.db"


def initialize_cache() -> None:
    db_keys = [
        "partitions",
        "graph_diffusion_degree_offsets",
        "marg_gain_lists",
    ]

    with shelve.open(CACHE_FILE_NAME) as cache:
        for key in db_keys:
            if key not in cache:
                cache[key] = {}


def convert_partition_to_dict(partition: la.VertexPartition) -> DictPartition:
    return {i: vertices for i, vertices in enumerate(partition)}


def get_partition(graph: nx.DiGraph) -> DictPartition:
    """
    TODO add a way to try different clustering methods
    """
    with shelve.open(CACHE_FILE_NAME, writeback=True) as cache:
        if graph.name not in cache["partitions"]:
            print("Starting partition")

            igraph_graph = ig.Graph.from_networkx(graph)
            result = convert_partition_to_dict(
                la.find_partition(igraph_graph, la.ModularityVertexPartition)
            )
            cache["partitions"][graph.name] = result

            print("Partitioning done")
            return result

        print("Reading partition from cache")
        return cache["partitions"][graph.name]


def reverse_partition(
    partition: DictPartition,
) -> dict[int, int]:
    res_dict = {}

    for i, part in partition.items():
        for vtx in part:
            res_dict[vtx] = i

    return res_dict


def compute_community_aware_diffusion_degrees(
    graph: nx.DiGraph,
    rev_partition_dict: dict[int, int],
) -> dict[int, float]:
    """
    TODO add a test case for very simple double check of this calculation.
    """

    with shelve.open(CACHE_FILE_NAME, writeback=True) as cache:
        if graph.name in cache["graph_diffusion_degree_offsets"]:
            return cache["graph_diffusion_degree_offsets"][graph.name]

        print("Computing community aware diffusion degree")

        res_dict = {}

        for start_node in graph:
            modified_graph = nx.subgraph_view(
                graph,
                filter_node=lambda x: x == start_node
                or rev_partition_dict[x] != rev_partition_dict[start_node],
            )

            route_proba_dict: defaultdict[int, int] = defaultdict(lambda: 1)

            for neighbor in modified_graph.neighbors(start_node):
                # Add to probability
                route_proba_dict[neighbor] *= (
                    1.0 - modified_graph[start_node][neighbor]["activation_prob"]
                )

                for second_neighbor in modified_graph.neighbors(neighbor):
                    if (
                        second_neighbor == start_node
                    ):  # Avoid going back to the start node
                        continue

                    route_proba_dict[second_neighbor] *= 1.0 - (
                        modified_graph[start_node][neighbor]["activation_prob"]
                        * modified_graph[neighbor][second_neighbor]["activation_prob"]
                    )

            res_dict[start_node] = sum(
                1.0 - route_prod for route_prod in route_proba_dict.values()
            )

        cache["graph_diffusion_degree_offsets"][graph.name] = res_dict

        return res_dict


def evaluate_diffusion(
    model: DiffusionModel, seed_set: t.Iterable[int], *, num_samples=10_000
) -> float:
    model.set_seeds(seed_set)

    total = 0.0

    for _ in range(num_samples):
        # Resetting the model doesn't change the initial seed set used.
        model.reset_model()
        model.advance_until_completion()
        total += model.get_num_activated_nodes()

    return total / num_samples


def compute_marginal_gain(
    model: DiffusionModel,
    vertex_weight_dict: dict[int, float] | None,
    new_node: int,
    seeds_list: t.List[int],
    *,
    num_trials: int = 1_000,
) -> float:
    """
    Compute the marginal gain in the spread of influence by adding a new node to the set of seed nodes,
    by summing the differences of spreads for each trial and then taking the average.

    Parameters:
    - model: The model used for simulating the spread of influence.
    - new_node: The new node to consider adding to the set of seed nodes.
    - seeds: The current set of seed nodes.
    - num_trials: The number of trials to average the spread of influence over.

    Returns:
    - The average marginal gain in the spread of influence by adding the new node.
    """
    seeds = set(seeds_list)
    original_spread = 0.0
    # If no seeds at the beginning, original spread is always just zero.
    # Prevents wasted work in cases where the seed set is empty.
    if len(seeds) > 0:
        model.set_seeds(seeds)

        for _ in range(num_trials):
            model.reset_model()
            model.advance_until_completion()
            original_spread += model.get_num_activated_nodes()

            if vertex_weight_dict is not None:
                for activated_node in model.get_activated_nodes():
                    original_spread += vertex_weight_dict[activated_node]

    new_seeds = seeds.union({new_node})
    model.set_seeds(new_seeds)

    new_spread = 0.0
    for _ in range(num_trials):
        model.reset_model()
        model.advance_until_completion()
        new_spread += model.get_num_activated_nodes()

        if vertex_weight_dict is not None:
            for activated_node in model.get_activated_nodes():
                new_spread += vertex_weight_dict[activated_node]

    # Check to make sure the program isn't going crazy.
    if (new_spread - original_spread) / num_trials < -5:
        print(seeds, new_node)
        print(new_spread, original_spread)
        raise Exception

    # Avoid floating point division until the very end.
    return (new_spread - original_spread) / num_trials


def celf(
    model: DiffusionModel,
    max_budget: int,
    nodes: list[int],
    vertex_weight_dict: dict[int, float] | None,
    *,
    num_trials: int = 1_000,
    tqdm_budget: bool = False,
) -> t.Generator[tuple[float, int], None, None]:  # tuple[list[int], list[float]]:
    """
    Input: graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    Code adapted from this blog post:
    https://hautahi.com/im_greedycelf
    """

    # Run the CELF algorithm
    marg_gain = []

    # print("Computing marginal gains.")
    # First, compute all marginal gains
    for node in tqdm.tqdm(nodes, leave=False):
        marg_gain.append(
            (
                -compute_marginal_gain(
                    model,
                    vertex_weight_dict,
                    node,
                    [],
                    num_trials=num_trials,
                ),
                node,
            )
        )

    # Convert to heap
    heapq.heapify(marg_gain)

    max_mg, selected_node = heapq.heappop(marg_gain)
    S = [selected_node]
    spreads = [max_mg]
    max_budget = min(max_budget, len(marg_gain))

    # print("Greedily selecting nodes.")
    # Greedily select remaining nodes
    # TODO Add option to use tqdm display here

    budget_iterator = (
        tqdm.trange(max_budget - 1) if tqdm_budget else range(max_budget - 1)
    )

    for _ in budget_iterator:
        while True:
            _, current_node = heapq.heappop(marg_gain)
            new_mg = -compute_marginal_gain(
                model,
                vertex_weight_dict,
                current_node,
                S,
                num_trials=num_trials,
            )

            if new_mg <= -marg_gain[0][0]:
                break
            else:
                heapq.heappush(marg_gain, (new_mg, current_node))

        S.append(current_node)
        spreads.append(new_mg)

        yield new_mg, current_node

    # Return the maximizing set S and the increasing spread values.
    # return S, spreads


def get_nested_solutions(
    graph_only_community_edges: nx.DiGraph,
    partition: DictPartition,
    budget: int,
    vertex_weight_dict: dict[int, float],
) -> list[t.Iterator[tuple[float, int]]]:
    # dict[int, list[int]]:
    """
    Given the graph with only community edges (no edges across communities),
    a partition of the vertices in each community, a budget, and a dictionary
    of additional vertex weights.

    Returns a list of iterators of the marginal gains in each community, in increasing
    order for each community.
    """

    model, _ = networkx_to_ic_model(graph_only_community_edges)

    # TODO maybe get rid of the stuff from the small communities?
    marg_gain_lists: list[t.Iterator[tuple[float, int]]] = [
        celf(model, budget, community, vertex_weight_dict)
        for community in partition.values()
    ]

    return marg_gain_lists


def assemble_best_seed_set(
    marg_gain_lists: list[t.Iterator[tuple[float, int]]], budget: int
) -> list[tuple[float, int]]:
    """
    Given a list of iterators to the marginal gains from each community,
    assemble the best set of nodes according to the given budget.
    """

    # Next, load these into a heap.
    min_heap: list[tuple[tuple[float, int], t.Iterator[tuple[float, int]]]] = []

    for it in tqdm.tqdm(marg_gain_lists):
        first_element = next(it, None)
        if first_element is not None:
            # (value, iterator)
            heapq.heappush(min_heap, (first_element, it))

    result: list[tuple[float, int]] = []

    for _ in tqdm.trange(budget):
        if not min_heap:
            break  # Exit early if no more nodes.

        value, it = heapq.heappop(min_heap)
        result.append(value)

        # Get the next element from the iterator
        next_value = next(it, None)
        if next_value is not None:
            heapq.heappush(min_heap, (next_value, it))

    assert len(result) == budget

    return result


def run_community_im(
    graph: nx.DiGraph,
    budget: int,
):
    parts = get_partition(graph)

    # Next, remove inter-community edges from graph
    rev_partition_dict = reverse_partition(parts)

    graph_only_community_edges = nx.subgraph_view(
        graph, filter_edge=lambda u, v: rev_partition_dict[u] == rev_partition_dict[v]
    )

    # TODO add the partitioning method to the graph name
    graph_no_community_edges = nx.subgraph_view(
        graph, filter_edge=lambda u, v: rev_partition_dict[u] != rev_partition_dict[v]
    )

    vertex_weight_dict = compute_community_aware_diffusion_degrees(
        graph, rev_partition_dict
    )

    # Now that we have the dict with weights, do influence max using these weights on each
    # community separately.

    nested_solution_list = get_nested_solutions(
        graph_only_community_edges, parts, budget, vertex_weight_dict
    )
    best_marg_gain_set = assemble_best_seed_set(nested_solution_list, budget)

    return {seed for _, seed in best_marg_gain_set}


def main() -> None:
    """
    Method used in experiments consists of three steps.

    1. Get graph and split into communities.
    2. Run greedy selection algorithm on each community
    3. Use progressive budgeting to obtain the final solution.
    """

    initialize_cache()

    # First, generate graph and partition
    # TODO this is the stuff to change when running later experiments
    graph = dm.get_graph("amazon")
    budget = 10_000

    start = time.perf_counter()
    best_seed_set = run_community_im(graph, budget)
    end = time.perf_counter()

    print(f"Community IM runtime {end-start}")

    # Now, evaluate
    model, _ = networkx_to_ic_model(graph)
    influence = evaluate_diffusion(model, best_seed_set)
    print(f"Community IM influence: {influence}")

    # Compare with CELF
    start = time.perf_counter()
    celf_marg_seeds = list(
        celf(
            model, budget, list(graph.nodes()), None, num_trials=1_000, tqdm_budget=True
        )
    )
    end = time.perf_counter()
    print(f"CELF runtime {end-start}")

    celf_seeds = {seed for _, seed in celf_marg_seeds}

    celf_value = evaluate_diffusion(model, celf_seeds)
    print(f"CELF influence {celf_value}")


if __name__ == "__main__":
    main()
