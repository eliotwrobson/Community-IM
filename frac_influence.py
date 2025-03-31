import random

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
    frac_alloc_dict: dict[int, float],
    a_dict: dict[int, float],
    b_dict: dict[int, float],
    *,
    num_trials: int = 10_000,
) -> float:
    """
    Computes the expected number of activated nodes given a fractional allocation.
    This is a linear approximation for the influence spread.
    """
    total_activated = 0.0

    seeds = []
    probs = []

    for node, prob in frac_alloc_dict.items():
        # Calculate the expected influence of each node based on a and b values
        # This is a linear approximation
        a_val = a_dict[node]
        b_val = b_dict[node]

        seeds.append(node)

        prob = a_val * prob + b_val

        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Invalid pobability computed for node: {prob}")

        probs.append(prob)

    model.set_seeds(seeds, probs)

    for _ in range(num_trials):
        model.reset_model()
        model.advance_until_completion()
        total_activated += model.get_num_activated_nodes()

    return total_activated / num_trials
