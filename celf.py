import heapq
import typing as t

import networkx as nx
import tqdm
from cynetdiff.models import DiffusionModel
from cynetdiff.utils import networkx_to_ic_model

DiffusionGraphT = t.Union[nx.Graph, nx.DiGraph]


def compute_marginal_gain(
    cynetdiff_model: DiffusionModel,
    new_node: int,
    seeds: list[int],  # TODO replace with None in case of no new seeds
    num_trials: int,
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

    original_spread = 0
    new_spread = 0
    # If no seeds at the beginning, original spread is always just zero.
    if not seeds:
        cynetdiff_model.set_seeds(seeds)

        for _ in range(num_trials):
            cynetdiff_model.reset_model()
            cynetdiff_model.advance_until_completion()
            original_spread += cynetdiff_model.get_num_activated_nodes()

    new_seeds = seeds + [new_node]
    cynetdiff_model.set_seeds(new_seeds)

    for _ in range(num_trials):
        cynetdiff_model.reset_model()
        cynetdiff_model.advance_until_completion()
        new_spread += cynetdiff_model.get_num_activated_nodes()

    return (new_spread - original_spread) / num_trials


# TODO have this take in the model itself.
def celf_im(
    graph: DiffusionGraphT, k: int, num_trials: int = 1_000
) -> tuple[list[int], list[float]]:
    """
    Input: graph object, number of seed nodes
    Output: optimal seed set (as a list, in order of marg gains),
    resulting spread, time for each iteration
    Code adapted from this blog post:
    https://hautahi.com/im_greedycelf
    """
    print("here")
    print("Starting CELF algorithm.")
    # Make cynetdiff model
    cynetdiff_model, _ = networkx_to_ic_model(graph)

    # Prepare graph
    dir_graph = graph
    if not dir_graph.is_directed():
        dir_graph = dir_graph.to_directed()

    # Run the CELF algorithm
    marg_gain = []

    # First, compute all marginal gains
    print("Computing initial marginal gains.")
    for node in tqdm(list(dir_graph.nodes())):
        marg_gain.append(
            (
                -compute_marginal_gain(
                    cynetdiff_model,
                    node,
                    [],
                    num_trials,
                ),
                node,
            )
        )

    heapq.heapify(marg_gain)

    max_mg, selected_node = heapq.heappop(marg_gain)
    S = [selected_node]
    spread = -max_mg
    spreads = [spread]

    print("Performing greedy selection.")
    for _ in range(k - 1):
        while True:
            current_mg, current_node = heapq.heappop(marg_gain)
            new_mg_neg = -compute_marginal_gain(
                cynetdiff_model,
                current_node,
                S,
                num_trials,
            )

            if new_mg_neg > current_mg:
                break
            else:
                heapq.heappush(marg_gain, (current_mg, current_node))

        spread += -new_mg_neg
        S.append(current_node)
        spreads.append(spread)

    return S, spreads
