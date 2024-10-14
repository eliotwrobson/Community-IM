"""
Main file for running community IM experiments
"""

import heapq
import shelve
import time
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from typing import assert_never

import igraph as ig
import leidenalg as la
import networkx as nx
import tqdm
from cynetdiff.models import DiffusionModel
from cynetdiff.utils import networkx_to_ic_model

import dataset_manager as dm

DictPartition = dict[int, list[int]]
PartitionMethod = t.Literal[
    "ModularityVertexPartition", "RBConfigurationVertexPartition"
]

CACHE_FILE_NAME = "cache.db"


@dataclass
class ExperimentResult:
    algorithm: str
    budget: int
    time_taken: float
    partition_time_taken: float
    diffusion_degree_time_taken: float
    graph: str
    weighting_scheme: str
    marginal_gain_error: float
    partition_method: PartitionMethod | None
    use_diffusion_degree: bool
    seed_set: set[int]


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


def get_partition(
    graph: nx.DiGraph, partition_method: PartitionMethod
) -> tuple[float, DictPartition, dict[int, int]]:
    """
    TODO add a way to try different clustering methods
    TODO add the partitioning method to the graph name
    """
    partition_name = f"{graph.name}_{graph.weighting_scheme}_{partition_method}"

    with shelve.open(CACHE_FILE_NAME, writeback=True) as cache:
        if partition_name not in cache["partitions"]:
            print("Starting partition")

            start_time = time.perf_counter()
            igraph_graph = ig.Graph.from_networkx(graph)

            # https://leidenalg.readthedocs.io/en/latest/reference.html
            if partition_method == "ModularityVertexPartition":
                partition_method_class = la.ModularityVertexPartition
            elif partition_method == "RBConfigurationVertexPartition":
                partition_method_class = la.RBConfigurationVertexPartition
            else:
                assert_never(partition_method)

            partition = la.find_partition(
                igraph_graph,
                partition_method_class,
                weights="activation_prob",
            )

            result = {i: vertices for i, vertices in enumerate(partition)}

            end_time = time.perf_counter()

            res_tup = (
                end_time - start_time,
                result,
                reverse_partition(result),
            )

            cache["partitions"][partition_name] = res_tup

        print("Reading partition from cache")
        return cache["partitions"][partition_name]


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
    partition_method: PartitionMethod,
    rev_partition_dict: dict[int, int],
) -> tuple[float, dict[int, float]]:
    """
    TODO add a test case for very simple double check of this calculation.
    """

    cache_entry_name = f"{graph.name}_{graph.weighting_scheme}_{partition_method}"

    with shelve.open(CACHE_FILE_NAME, writeback=True) as cache:
        if cache_entry_name in cache["graph_diffusion_degree_offsets"]:
            return cache["graph_diffusion_degree_offsets"][cache_entry_name]

        print("Computing community aware diffusion degree")
        start_time = time.perf_counter()
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
                    # Avoid going back to the start node
                    if second_neighbor == start_node:
                        continue

                    route_proba_dict[second_neighbor] *= 1.0 - (
                        modified_graph[start_node][neighbor]["activation_prob"]
                        * modified_graph[neighbor][second_neighbor]["activation_prob"]
                    )

            res_dict[start_node] = sum(
                1.0 - route_prod for route_prod in route_proba_dict.values()
            )
        end_time = time.perf_counter()
        res_tup = (end_time - start_time, res_dict)

        cache["graph_diffusion_degree_offsets"][cache_entry_name] = res_tup

        return res_tup


def evaluate_diffusion(
    model: DiffusionModel, seed_set: t.Iterable[int], *, num_samples=10_000
) -> float:
    model.set_seeds(seed_set)

    total = 0

    print("Evaluating quality of diffusion")
    for _ in tqdm.trange(num_samples):
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
    # if (new_spread - original_spread) / num_trials < -5:
    #     print(seeds, new_node)
    #     print(new_spread, original_spread)
    #     raise Exception

    # Avoid floating point division until the very end.
    return (new_spread - original_spread) / num_trials


def celf(
    model: DiffusionModel,
    max_budget: int,
    nodes: list[int],
    vertex_weight_dict: dict[int, float] | None,
    num_trials: int,
    marginal_gain_error: float,
    *,
    tqdm_budget: bool = False,
) -> t.Generator[tuple[float, int], None, None]:  # tuple[list[int], list[float]]:
    """
    marginal_gain_error: The amount of slack allowed in the computation of marginal gain.
    Potentially introduces some small error, but likely worth the runtaime gains.

    Input: graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    Code adapted from this blog post:
    https://hautahi.com/im_greedycelf
    """

    if marginal_gain_error < 0.0:
        raise ValueError(
            f"Invalid marg_gain_error {marginal_gain_error}, must be at least 0."
        )

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

    budget_iterator = (
        tqdm.trange(max_budget - 1) if tqdm_budget else range(max_budget - 1)
    )

    for _ in budget_iterator:
        while True:
            celf_pp_cache = {}

            _, current_node = heapq.heappop(marg_gain)

            # CELF++ optimization: Cache nodes. If previously computed, use this result
            if current_node in celf_pp_cache:
                new_mg = celf_pp_cache[current_node]
                break

            new_mg = -compute_marginal_gain(
                model,
                vertex_weight_dict,
                current_node,
                S,
                num_trials=num_trials,
            )

            celf_pp_cache[current_node] = new_mg

            # My own optimization: Add granularity argument to ignore
            # TODO double check this works as expected
            if new_mg - marginal_gain_error <= -marg_gain[0][0]:
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
    num_trials: int,
    marginal_gain_error: float,
) -> list[t.Iterator[tuple[float, int]]]:
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
        celf(
            model,
            budget,
            community,
            vertex_weight_dict,
            num_trials,
            marginal_gain_error,
        )
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


def community_im_runner(
    graph: nx.DiGraph,
    budget: int,
    marginal_gain_error: float,
    partition_method: str,
    use_diffusion_degree: bool,
    num_trials: int,
) -> ExperimentResult:
    partition_time_taken, parts, rev_partition_dict = get_partition(
        graph, partition_method
    )

    # Next, remove inter-community edges from graph
    graph_only_community_edges = nx.subgraph_view(
        graph, filter_edge=lambda u, v: rev_partition_dict[u] == rev_partition_dict[v]
    )

    diffusion_degree_time_taken, vertex_weight_dict = (
        compute_community_aware_diffusion_degrees(
            graph, partition_method, rev_partition_dict
        )
    )

    # Now that we have the dict with weights, do influence max using these weights on each
    # community separately.

    start = time.perf_counter()
    nested_solution_list = get_nested_solutions(
        graph_only_community_edges,
        parts,
        budget,
        vertex_weight_dict,
        num_trials,
        marginal_gain_error,
    )
    best_marg_gain_set = assemble_best_seed_set(nested_solution_list, budget)

    end = time.perf_counter()

    return ExperimentResult(
        algorithm="community-im",
        budget=budget,
        time_taken=end - start,
        partition_time_taken=partition_time_taken,
        diffusion_degree_time_taken=diffusion_degree_time_taken,
        graph=graph.name,
        weighting_scheme=graph.weighting_scheme,
        marginal_gain_error=marginal_gain_error,
        partition_method=partition_method,
        use_diffusion_degree=use_diffusion_degree,
        seed_set={seed for _, seed in best_marg_gain_set},
    )


def celf_pp_runner(
    graph: nx.DiGraph,
    budget: int,
    marginal_gain_error: float,
    num_trials: int,
) -> tuple[DiffusionModel, ExperimentResult]:
    # Now that we have the dict with weights, do influence max using these weights on each
    # community separately.

    start = time.perf_counter()
    model, _ = networkx_to_ic_model(graph)
    celf_marg_seeds = list(
        celf(
            model,
            budget,
            list(graph.nodes()),
            None,
            num_trials=num_trials,
            marginal_gain_error=marginal_gain_error,
            tqdm_budget=True,
        )
    )
    end = time.perf_counter()

    return (
        model,
        ExperimentResult(
            algorithm="celf-pp",
            budget=budget,
            time_taken=end - start,
            partition_time_taken=0.0,
            diffusion_degree_time_taken=0.0,
            graph=graph.name,
            weighting_scheme=graph.weighting_scheme,
            marginal_gain_error=marginal_gain_error,
            partition_method=None,
            use_diffusion_degree=False,
            seed_set={seed for _, seed in celf_marg_seeds},
        ),
    )


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
    graph = dm.get_graph("deezer")
    budget = 100

    model, celf_result = celf_pp_runner(graph, budget, 0.0, 1_000)
    result = community_im_runner(
        graph, budget, 0.0, "ModularityVertexPartition", True, 1_000
    )

    print(result)
    influence = evaluate_diffusion(model, result.seed_set)
    print(influence)
    influence = evaluate_diffusion(model, celf_result.seed_set)
    print(influence)

    # start = time.perf_counter()
    # best_seed_set = run_community_im(graph, budget)
    # end = time.perf_counter()

    # print(f"Community IM runtime {end-start}")

    # # Now, evaluate
    # model, _ = networkx_to_ic_model(graph)
    # influence = evaluate_diffusion(model, best_seed_set)
    # print(f"Community IM influence: {influence}")

    # # Compare with CELF
    # start = time.perf_counter()
    # celf_marg_seeds = list(
    #     celf(
    #         model, budget, list(graph.nodes()), None, num_trials=1_000, tqdm_budget=True
    #     )
    # )
    # end = time.perf_counter()
    # print(f"CELF runtime {end-start}")

    # celf_seeds = {seed for _, seed in celf_marg_seeds}

    # celf_value = evaluate_diffusion(model, celf_seeds)
    # print(f"CELF influence {celf_value}")


if __name__ == "__main__":
    main()
