import random

import networkx as nx
import tqdm
from cynetdiff.models import IndependentCascadeModel


def compute_fractional_influence(
    model: IndependentCascadeModel,
    frac_alloc_dict: dict[int, float],
    *,
    random_seed: int = 12345,
    num_trials: int = 10_000,
) -> float:
    random.seed(random_seed)

    total_activated = 0.0

    seeds, probs = list(zip(*frac_alloc_dict.items()))  # Unzip dict
    model.set_seeds(seeds, probs)

    for _ in tqdm.trange(num_trials):
        model.reset_model()
        model.advance_until_completion()
        total_activated += model.get_num_activated_nodes()

    return total_activated / num_trials


def compute_fractional_influence_linear(
    model: IndependentCascadeModel,
    frac_alloc_dict: list[tuple[int, float, float, float]],
    network: nx.DiGraph,
    *,
    budget: float | None = None,
    num_trials: int = 10_000,
    random_seed: int = 12345,
) -> float:
    """
    Computes the expected number of activated nodes given a fractional allocation.
    This is a linear approximation for the influence spread.
    """

    seeds = []
    probs = []

    remainging_budget = budget if budget is not None else float("inf")

    for node, discount, _, _ in frac_alloc_dict:
        # Calculate the expected influence of each node based on a and b values
        # This is a linear approximation
        a_val = network.nodes[node]["a"]
        b_val = network.nodes[node]["b"]
        w_val = network.nodes[node]["w"]

        seeds.append(node)
        cost = w_val * discount

        # Make sure we don't go over the budget
        if remainging_budget < cost:
            discount = remainging_budget / w_val
            remainging_budget = 0.0
        else:
            remainging_budget -= cost

        prob = a_val * discount + b_val

        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Invalid probability computed for node {node}: {prob}")

        probs.append(prob)
        if remainging_budget <= 0.0:
            break

    model.set_rng(random_seed)
    model.set_seeds(seeds, probs)
    total_activated = 0.0

    for _ in range(num_trials):
        model.reset_model()
        model.advance_until_completion()
        total_activated += model.get_num_activated_nodes()

    return total_activated / num_trials
