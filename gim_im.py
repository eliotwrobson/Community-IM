"""
File running the GIM influence max algo. This algorithm is defined using
an arbitrary convex function, but here we restrict ourselves to a linear
function. Uses the CELF optimization.
"""

import heapq
import time
from collections import defaultdict

import networkx as nx
import tqdm
from cynetdiff.utils import networkx_to_ic_model
from depqdict import DepqDict

from frac_influence import compute_fractional_influence_linear


def gim_degree_discount(
    graph: nx.DiGraph,
    budget: float,
) -> list[tuple[int, float, float, float]]:
    start_time = time.perf_counter()

    # Initialize data structures
    t_dict: defaultdict[int, int] = defaultdict(int)
    best_node_dict: DepqDict[int, float] = DepqDict()
    activation_cost_dict: dict[int, tuple[float, float]] = {}

    for node, out_degree in graph.out_degree():
        y_v = (1.0 - graph.nodes[node]["b"]) / graph.nodes[node]["a"]
        activation_cost = y_v * graph.nodes[node]["w"]

        assert activation_cost > 0

        best_node_dict[node] = out_degree / activation_cost
        activation_cost_dict[node] = (y_v, activation_cost)

    seeds = []
    weighted_total = 0.0

    while weighted_total < budget and len(best_node_dict) > 0:
        node, _ = best_node_dict.pop_max_item()
        end_time = time.perf_counter()

        y_v, activation_cost = activation_cost_dict[node]

        weighted_total += activation_cost

        seeds.append((node, y_v, weighted_total, end_time - start_time))

        for v in graph.predecessors(node):
            if v not in best_node_dict:
                continue

            t_dict[v] += 1

            d_v = graph.out_degree(v)
            t_v = t_dict[v]

            # NOTE this assumes that the activation probability is the same for all out
            # edges of v.
            best_node_dict[v] = (
                d_v - 2 * t_v - (d_v - t_v) * t_v * graph[v][node]["activation_prob"]
            ) / activation_cost_dict[v][1]

    return seeds


def gim_im(
    network: nx.Graph,
    budget: float,  # TODO maybe can this be fractional?
    *,
    random_seed: int = 12345,
    num_trials: int = 1_000,  # TODO this should be passed in from the outside
) -> tuple[list[tuple[int, float, float, float]], float]:
    num_nodes = network.number_of_nodes()
    node_scaling: dict[int, float] = {}

    start_time = time.perf_counter()

    for node, data in network.nodes(data=True):
        # f_inv_dict[node] = (1.0 - b_val) / a_val

        node_scaling[node] = data["a"] / data["w"]

    # TODO double check that we set the weighting scheme outside of this.
    model, _ = networkx_to_ic_model(network, rng=random_seed)
    # We run our algorithm natively using cynetdiff

    # Create heap for CELF-type algorithm
    marg_gain_heap = [
        (
            -model.compute_marginal_gains([node], [], num_trials=num_trials)[0]
            * node_scaling[node],
            node,
        )
        for node in tqdm.trange(num_nodes)
    ]

    heapq.heapify(marg_gain_heap)

    # Storing the result vector as a dict because it's sparse.
    discount_dict: list[tuple[int, float, float, float]] = []
    weighted_total = 0.0
    seed_set: set[int] = set()

    # TODO make sure to account for floating point error
    while weighted_total < budget and len(seed_set) < num_nodes:
        print(len(discount_dict), weighted_total, budget)

        # Code here based off of this blog post:
        # https://hautahi.com/im_greedycelf
        matches = False
        # Put this here to avoid popping from the empty heap
        while not matches:
            # Get element with max marginal gain
            _, current_node = marg_gain_heap[0]

            # Compute updated marginal gain for this element
            new_mg_neg = (
                -model.compute_marginal_gains(
                    seed_set, [current_node], num_trials=num_trials
                )[1]
                * node_scaling[current_node]
            )

            # Insert node with updated marginal gain
            heapq.heappushpop(marg_gain_heap, (new_mg_neg, current_node))

            # Check if top element has not changed after update
            matches = marg_gain_heap[0][1] == current_node

        curr_node_weight = network.nodes[current_node]["w"]

        # TODO double check this is the correct way to assign this.
        curr_val = min(
            (1.0 - network.nodes[current_node]["b"]) / network.nodes[current_node]["a"],
            (budget - weighted_total) / curr_node_weight,
        )

        weighted_total += curr_val * curr_node_weight
        curr_time = time.perf_counter()

        discount_dict.append(
            (current_node, curr_val, weighted_total, curr_time - start_time)
        )

        seed_set.add(current_node)
        heapq.heappop(marg_gain_heap)

    avg_influence = compute_fractional_influence_linear(model, discount_dict, network)

    print(
        "Avg influence of final seed set:",
        avg_influence,
    )
    print(f"Sum total: {weighted_total}, Budget: {budget}")
    return discount_dict, avg_influence
