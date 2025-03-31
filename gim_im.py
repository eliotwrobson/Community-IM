"""
File running the GIM influence max algo. This algorithm is defined using
an arbitrary convex function, but here we restrict ourselves to a linear
function. Uses the CELF optimization.
"""

import heapq
import random

import networkx as nx
import tqdm
from cynetdiff.utils import networkx_to_ic_model

from frac_influence import compute_fractional_influence_linear

# def gim_im(
#     network: nx.Graph,
#     budget: float,  # TODO maybe can this be fractional?
#     a_vals: tuple[int],
#     b_vals: tuple[int],
# ):
#     num_nodes = network.number_of_nodes()

#     a_list = random.choices(a_vals, k=num_nodes)
#     b_list = random.choices(b_vals, k=num_nodes)

#     # TODO double check that we set the weighting scheme outside of this.
#     model, node_mapping_dict = networkx_to_ic_model(network)
#     # We run our algorithm natively using cynetdiff

#     # Storing the result vector as a dict because it's sparse.
#     discount_dict = {}
#     sum_total = 0.0
#     seed_set = set()

#     # Create heap for CELF-type algorithm
#     # TODO rescale using a vector
#     marg_gain_heap = [
#         (
#             -model.compute_marginal_gains([node], [], num_trials=num_trials)[0]
#             * a_list[node],
#             node,
#         )
#         for node in tqdm.trange(num_nodes)
#     ]

#     heapq.heapify(marg_gain_heap)

#     # TODO make sure to account for floating point error
#     while sum_total < budget and len(seed_set) < num_nodes:
#         print(discount_dict, sum_total, budget)

#         # Code here based off of this blog post:
#         # https://hautahi.com/im_greedycelf
#         matches = False
#         # Put this here to avoid popping from the empty heap
#         while not matches:
#             # Get element with max marginal gain
#             _, current_node = marg_gain_heap[0]

#             # Compute updated marginal gain for this element
#             new_mg_neg = -(
#                 model.compute_marginal_gains(
#                     current_node, seed_set, num_trials=num_trials
#                 )[1]
#                 * a_list[current_node]
#             )

#             # Insert node with updated marginal gain
#             heapq.heappushpop(marg_gain_heap, (new_mg_neg, current_node))

#             # Check if top element has not changed after update
#             matches = marg_gain_heap[0][1] == current_node

#         # TODO double check this is the correct way to assign this.
#         discount_dict[current_node] = min(
#             (1 - b_list[current_node]) / a_list[current_node], budget - sum_total
#         )
#         sum_total += discount_dict[current_node]
#         seed_set.add(current_node)
#         heapq.heappop(marg_gain_heap)

#     print("Avg influence of final seed set:", avg_influence(model, seed_set))
#     print(f"Sum total: {sum_total}, Budget: {budget}")
#     return discount_dict, compute_fractional_influence(
#         model, discount_dict, a_list, b_list
#     )


def gim_im(
    network: nx.Graph,
    budget: float,  # TODO maybe can this be fractional?
    a_vals: tuple[float, ...],
    b_vals: tuple[float, ...],
    *,
    random_seed: int = 12345,
    num_trials: int = 1_000,  # TODO this should be passed in from the outside
):
    num_nodes = network.number_of_nodes()

    random.seed(random_seed)  # Set the random seed for reproducibility
    a_dict: dict[int, float] = {}
    b_dict: dict[int, float] = {}

    for node, data in network.nodes(data=True):
        a_val = random.choice(a_vals)
        b_val = random.choice(b_vals)

        a_dict[node] = a_val
        b_dict[node] = b_val
        # f_inv_dict[node] = (1.0 - b_val) / a_val

        data["payoff"] = a_val

    # TODO double check that we set the weighting scheme outside of this.
    model, _ = networkx_to_ic_model(network, rng=random_seed)
    # We run our algorithm natively using cynetdiff

    # Storing the result vector as a dict because it's sparse.
    discount_dict: dict[int, float] = {}
    sum_total = 0.0
    seed_set: set[int] = set()

    # Create heap for CELF-type algorithm
    marg_gain_heap = [
        (
            -model.compute_marginal_gains([node], [], num_trials=num_trials)[0],
            node,
        )
        for node in tqdm.trange(num_nodes)
    ]

    heapq.heapify(marg_gain_heap)

    # TODO make sure to account for floating point error
    while sum_total < budget and len(seed_set) < num_nodes:
        print(discount_dict, sum_total, budget)
        # orig_influence = 0

        # if seed_set:
        #     orig_influence = avg_influence(model, seed_set)

        # Code here based off of this blog post:
        # https://hautahi.com/im_greedycelf
        matches = False
        # Put this here to avoid popping from the empty heap
        while not matches:
            # Get element with max marginal gain
            _, current_node = marg_gain_heap[0]

            # Compute updated marginal gain for this element
            new_mg_neg = -model.compute_marginal_gains(
                seed_set, [current_node], num_trials=num_trials
            )[1]

            # Insert node with updated marginal gain
            heapq.heappushpop(marg_gain_heap, (new_mg_neg, current_node))

            # Check if top element has not changed after update
            matches = marg_gain_heap[0][1] == current_node

        # TODO double check this is the correct way to assign this.
        discount_dict[current_node] = min(
            (1.0 - b_dict[current_node]) / a_dict[current_node], budget - sum_total
        )
        sum_total += discount_dict[current_node]
        seed_set.add(current_node)
        heapq.heappop(marg_gain_heap)

    print(
        "Avg influence of final seed set:",
        model.compute_marginal_gains(seed_set, [], num_trials=num_trials)[0],
    )
    print(f"Sum total: {sum_total}, Budget: {budget}")
    return discount_dict, compute_fractional_influence_linear(
        model, discount_dict, a_dict, b_dict
    )
