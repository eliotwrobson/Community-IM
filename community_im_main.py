"""
Main file for running community IM experiments
"""

import heapq
import itertools as it
import json
import os
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
from heapdict import heapdict

import dataset_manager as dm

DictPartition = dict[int, list[int]]
PartitionMethod = t.Literal["RBConfigurationVertexPartition", "CPMVertexPartition"]

CACHE_FILE_NAME = "cache.db"
RESULT_FILE_NAME = "benchmark_result.json"
RANDOM_SEED_DEFAULT = 12345


@dataclass
class ExperimentResult:
    algorithm: str
    times_taken: list[float]
    graph: str
    num_nodes: int
    num_edges: int
    weighting_scheme: str
    seeds: list[int]
    marginal_gain_error: float | None = None
    use_diffusion_degree: bool = False
    partition_method: PartitionMethod | None = None
    partition_time_taken: float = 0.0
    diffusion_degree_time_taken: float = 0.0
    quality: float | None = None
    num_communities: int | None = None
    num_edges_removed: int = 0
    resolution_parameter: float | None = None


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


def write_benchmark_result(
    result: ExperimentResult, budget: int, influence: float
) -> None:
    time_taken = result.times_taken[budget - 1] + result.partition_time_taken

    if result.use_diffusion_degree:
        time_taken += result.diffusion_degree_time_taken

    # Create file if it doesn't already exist
    if not os.path.exists(RESULT_FILE_NAME):
        with open(RESULT_FILE_NAME, "w") as f:
            json.dump({"results": []}, f)

    # Read file and append results
    with open(RESULT_FILE_NAME, "r") as f:
        data = json.load(f)

    result_dict = {
        "graph": result.graph,
        "num nodes": result.num_nodes,
        "num edges": result.num_edges,
        "algorithm": result.algorithm,
        "time taken": time_taken,
        "influence": influence,
        "budget": budget,
        "weighting scheme": result.weighting_scheme,
        "use diffusion degree": result.use_diffusion_degree,
        "marginal gain error": result.marginal_gain_error,
        "quality": result.quality,
        "partition method": result.partition_method,
        "resolution parameter": result.resolution_parameter,
        "num communities": result.num_communities,
        "num edges removed": result.num_edges_removed,
    }

    data["results"].append(result_dict)

    with open(RESULT_FILE_NAME, "w") as f:
        json.dump(data, f, indent=4)


def get_partition(
    graph: nx.DiGraph,
    partition_method: PartitionMethod,
    resolution_parameter: float | None,
    *,
    random_seed: int = 12345,
) -> tuple[float, DictPartition, dict[int, int], float]:
    """ """
    partition_name = f"{graph.name}_{graph.weighting_scheme}_{partition_method}_{resolution_parameter}"

    with shelve.open(CACHE_FILE_NAME, writeback=True) as cache:
        if partition_name not in cache["partitions"]:
            print(f"Computing partition {partition_method}")

            start_time = time.perf_counter()

            igraph_graph = ig.Graph.from_networkx(graph)

            # https://leidenalg.readthedocs.io/en/latest/reference.html
            if partition_method == "RBConfigurationVertexPartition":
                partition_method_class = la.RBConfigurationVertexPartition
            elif partition_method == "CPMVertexPartition":
                partition_method_class = la.CPMVertexPartition
            else:
                assert_never(partition_method)

            partition = la.find_partition(
                igraph_graph,
                partition_method_class,
                weights="activation_prob",
                resolution_parameter=resolution_parameter,
                seed=random_seed,
            )

            quality = partition.quality()

            end_time = time.perf_counter()

            # Compute partition and reversal
            result = {}
            rev_dict = {}

            for i, part in enumerate(partition):
                result[i] = part

                for vtx in part:
                    rev_dict[vtx] = i

            # Assemble into tuple and cache
            res_tup = (
                end_time - start_time,
                result,
                rev_dict,
                quality,
            )

            cache["partitions"][partition_name] = res_tup

        return cache["partitions"][partition_name]


def compute_community_aware_diffusion_degrees(
    graph: nx.DiGraph,
    partition_method: PartitionMethod,
    resolution_parameter: float | None,
    rev_partition_dict: dict[int, int],
) -> tuple[float, dict[int, float]]:
    """
    TODO add a test case for very simple double check of this calculation.
    """

    cache_entry_name = f"{graph.name}_{graph.weighting_scheme}_{partition_method}_{resolution_parameter}"

    with shelve.open(CACHE_FILE_NAME, writeback=True) as cache:
        if cache_entry_name in cache["graph_diffusion_degree_offsets"]:
            return cache["graph_diffusion_degree_offsets"][cache_entry_name]

        print("Computing community aware diffusion degree")
        start_time = time.perf_counter()
        res_dict = {}

        for start_node in tqdm.tqdm(graph, total=graph.number_of_nodes()):
            # Ignore all edges within the same community as start_node
            modified_graph = nx.subgraph_view(
                graph,
                filter_edge=lambda u, v: not (
                    rev_partition_dict[u]
                    == rev_partition_dict[v]
                    == rev_partition_dict[start_node]
                ),
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

            node_score = sum(
                1.0 - route_prod for route_prod in route_proba_dict.values()
            )

            assert node_score >= 0.0

            res_dict[start_node] = node_score

        end_time = time.perf_counter()
        res_tup = (end_time - start_time, res_dict)

        cache["graph_diffusion_degree_offsets"][cache_entry_name] = res_tup

        return res_tup


def evaluate_diffusion(
    model: DiffusionModel, seed_set: t.Iterable[int], *, num_samples=10_000
) -> float:
    model.set_seeds(seed_set)

    total = 0

    for _ in tqdm.trange(num_samples, leave=False):
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
) -> t.Generator[tuple[float, int], None, None]:
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
    # spreads = [max_mg]
    max_budget = min(max_budget, len(marg_gain))

    budget_iterator = tqdm.trange(max_budget) if tqdm_budget else range(max_budget)

    for _ in budget_iterator:
        # num_iters = 0
        while True:
            celf_pp_cache: dict[int, float] = {}

            _, current_node = heapq.heappop(marg_gain)

            # CELF++ optimization: Cache nodes. If previously computed, use this result
            if current_node in celf_pp_cache:
                new_mg = celf_pp_cache[current_node]
                break

            new_mg = compute_marginal_gain(
                model,
                vertex_weight_dict,
                current_node,
                S,
                num_trials=num_trials,
            )

            # NOTE uncomment for timing code
            # num_iters += 1
            # if num_iters % 100 == 1:
            #     print(new_mg, -marg_gain[0][0])

            celf_pp_cache[current_node] = new_mg

            # My own optimization: Add granularity argument to ignore
            # TODO change the marginal gain error to be a scaling factor larger than 1.0
            if not marg_gain or new_mg + marginal_gain_error >= -marg_gain[0][0]:
                break
            else:
                heapq.heappush(marg_gain, (-new_mg, current_node))

        # print(num_iters)
        S.append(current_node)
        # spreads.append(new_mg)

        yield -new_mg, current_node

    # Return the maximizing set S and the increasing spread values.
    # return S, spreads


def get_nested_solutions(
    graph_only_community_edges: nx.DiGraph,
    partition: DictPartition,
    budget: int,
    vertex_weight_dict: dict[int, float] | None,
    num_trials: int,
    marginal_gain_error: float,
) -> tuple[list[float], list[int]]:
    """
    Given the graph with only community edges (no edges across communities),
    a partition of the vertices in each community, a budget, and a dictionary
    of additional vertex weights.

    Returns a list of iterators of the marginal gains in each community, in increasing
    order for each community.
    """

    start_time = time.perf_counter()
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

    # Next, load these into a heap.
    min_heap: list[tuple[tuple[float, int], t.Iterator[tuple[float, int]]]] = []

    for iterable in tqdm.tqdm(marg_gain_lists):
        first_element = next(iterable, None)
        if first_element is not None:
            # (value, iterator)
            heapq.heappush(min_heap, (first_element, iterable))

    seeds: list[int] = []
    times: list[float] = []

    for _ in tqdm.trange(budget):
        if not min_heap:
            break  # Exit early if no more nodes.

        (_, node), iterable = heapq.heappop(min_heap)
        end_time = time.perf_counter()

        seeds.append(node)
        times.append(end_time - start_time)

        # Get the next element from the iterator
        next_value = next(iterable, None)
        if next_value is not None:
            heapq.heappush(min_heap, (next_value, iterable))

    assert len(seeds) == budget

    return times, seeds


def community_im_runner(
    graph: nx.DiGraph,
    budget: int,
    marginal_gain_error: float,
    partition_method: PartitionMethod,
    resolution_parameter: float | None,
    use_diffusion_degree: bool,
    num_trials: int,
    *,
    random_seed: int = 12345,
) -> ExperimentResult:
    partition_time_taken, parts, rev_partition_dict, quality = get_partition(
        graph, partition_method, resolution_parameter, random_seed=random_seed
    )

    # Next, remove inter-community edges from graph
    graph_only_community_edges = nx.subgraph_view(
        graph, filter_edge=lambda u, v: rev_partition_dict[u] == rev_partition_dict[v]
    )

    # Compute diffusion degree heuristic if necessary

    if use_diffusion_degree:
        diffusion_degree_time_taken, vertex_weight_dict = (
            compute_community_aware_diffusion_degrees(
                graph, partition_method, resolution_parameter, rev_partition_dict
            )
        )
    else:
        diffusion_degree_time_taken = 0.0
        vertex_weight_dict = None

    # Now that we have the dict with weights, do influence max using these weights on each
    # community separately.

    times_taken, seeds = get_nested_solutions(
        graph_only_community_edges,
        parts,
        budget,
        vertex_weight_dict,
        num_trials,
        marginal_gain_error,
    )

    return ExperimentResult(
        algorithm="community-im",
        times_taken=times_taken,
        partition_time_taken=partition_time_taken,
        diffusion_degree_time_taken=diffusion_degree_time_taken,
        graph=graph.name,
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        weighting_scheme=graph.weighting_scheme,
        marginal_gain_error=marginal_gain_error,
        partition_method=partition_method,
        use_diffusion_degree=use_diffusion_degree,
        seeds=seeds,
        quality=quality,
        num_communities=len(parts),
        num_edges_removed=graph.number_of_edges()
        - graph_only_community_edges.number_of_edges(),
        resolution_parameter=resolution_parameter,
    )


def celf_pp_runner(
    graph: nx.DiGraph,
    budget: int,
    marginal_gain_error: float,
    num_trials: int,
) -> tuple[DiffusionModel, ExperimentResult]:
    # Now that we have the dict with weights, do influence max using these weights on each
    # community separately.

    start_time = time.perf_counter()
    model, _ = networkx_to_ic_model(graph)

    times_taken = []
    seeds = []

    celf_iter = celf(
        model,
        budget,
        list(graph.nodes()),
        None,
        num_trials=num_trials,
        marginal_gain_error=marginal_gain_error,
        tqdm_budget=True,
    )

    for _, seed in celf_iter:
        end_time = time.perf_counter()

        seeds.append(seed)
        times_taken.append(end_time - start_time)

    assert len(seeds) == budget

    return (
        model,
        ExperimentResult(
            algorithm="celf-pp",
            times_taken=times_taken,
            graph=graph.name,
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            weighting_scheme=graph.weighting_scheme,
            marginal_gain_error=marginal_gain_error,
            seeds=seeds,
        ),
    )


def degree_runner(
    graph: nx.DiGraph,
    budget: int,
) -> ExperimentResult:
    start_time = time.perf_counter()

    # Create a min-heap to store the top k nodes
    min_heap: list[tuple[int, int]] = []

    # Iterate through the nodes and their out-degrees
    for node in graph.nodes():
        node_tup = (-graph.out_degree(node), node)

        # If the heap exceeds size k, pop the smallest element
        if len(min_heap) < budget:
            heapq.heappush(min_heap, node_tup)
        else:
            heapq.heappushpop(min_heap, node_tup)

    seeds = []
    times_taken = []

    while min_heap:
        _, node = heapq.heappop(min_heap)
        end_time = time.perf_counter()

        seeds.append(node)
        times_taken.append(end_time - start_time)

    assert len(seeds) == budget

    return ExperimentResult(
        algorithm="degree",
        times_taken=times_taken,
        graph=graph.name,
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        weighting_scheme=graph.weighting_scheme,
        seeds=seeds,
    )


def degree_discount_runner(
    graph: nx.DiGraph,
    budget: int,
) -> ExperimentResult:
    start_time = time.perf_counter()

    # Initialize data structures
    t_dict: defaultdict[int, int] = defaultdict(int)
    best_node_dict = heapdict()

    for node, out_degree in graph.out_degree():
        best_node_dict[node] = -out_degree

    seeds = []
    times_taken = []

    for i in range(budget):
        node, _ = best_node_dict.popitem()
        end_time = time.perf_counter()

        seeds.append(node)
        times_taken.append(end_time - start_time)

        if i == budget - 1:
            break

        for v in graph.neighbors(node):
            if v not in best_node_dict:
                continue

            t_dict[v] += 1

            d_v = graph.out_degree(v)
            t_v = t_dict[v]

            best_node_dict[v] = (
                d_v - 2 * t_v - (d_v - t_v) * t_v * graph[node][v]["activation_prob"]
            )

    assert len(seeds) == budget

    return ExperimentResult(
        algorithm="degree-discount",
        times_taken=times_taken,
        graph=graph.name,
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        weighting_scheme=graph.weighting_scheme,
        seeds=seeds,
    )


def main() -> None:
    """
    Method used in experiments consists of three steps.

    1. Get graph and split into communities.
    2. Run greedy selection algorithm on each community
    3. Use progressive budgeting to obtain the final solution.
    """

    initialize_cache()

    with open("benchmark_configs/community_im_settings.json") as f:
        settings_dict = json.load(f)

    # TODO add progress indicator for total number of experiments that are being run.
    # maybe use logging?
    graphs = it.product(settings_dict["graphs"], settings_dict["weighting_schemes"])
    budgets = sorted(settings_dict["budgets"])
    max_budget = budgets[-1]
    random_seed = settings_dict.get("random_seed", RANDOM_SEED_DEFAULT)

    skip_celf = settings_dict.get("skip_celf", False)

    for graph_name, weighting_scheme in graphs:
        graph = dm.get_graph(graph_name, weighting_scheme, random_seed=random_seed)

        print(
            f"Running degree and degree discount on {graph_name} with budget {max_budget}."
        )

        # Start with degree discount baseline because it's easy to compute.
        graph_benchmark_results: list[ExperimentResult] = [
            degree_runner(graph, max_budget),
            degree_discount_runner(graph, max_budget),
        ]

        for marginal_gain_error, num_samples in it.product(
            settings_dict["marginal_gain_errors"], settings_dict["num_samples"]
        ):
            # Run CELF algorithm
            if not skip_celf:
                print(f"Running celfpp on {graph_name} with budget {max_budget}.")
                model, celf_result = celf_pp_runner(
                    graph, max_budget, marginal_gain_error, num_samples
                )

                graph_benchmark_results.append(celf_result)

            for use_diffusion_degree, partitioning_method in it.product(
                settings_dict["use_diffusion_degree"],
                settings_dict["partitioning_methods"],
            ):
                partitioning_algorithm = partitioning_method["partitioning_algorithm"]
                resolution_parameter = partitioning_method.get("resolution_parameter")

                print(f"Running community-im on {graph_name} with budget {max_budget}.")
                community_im_result = community_im_runner(
                    graph,
                    max_budget,
                    marginal_gain_error,
                    partitioning_algorithm,
                    resolution_parameter,
                    use_diffusion_degree,
                    num_samples,
                    random_seed=random_seed,
                )

                graph_benchmark_results.append(community_im_result)

        if skip_celf:
            model, _ = networkx_to_ic_model(graph)

        print("Evaluating quality of seed sets.")
        length = len(graph_benchmark_results) * len(budgets)
        for result, budget in tqdm.tqdm(
            it.product(graph_benchmark_results, budgets), total=length
        ):
            seeds = result.seeds[:budget]
            influence = evaluate_diffusion(model, seeds)

            write_benchmark_result(result, budget, influence)


if __name__ == "__main__":
    main()
