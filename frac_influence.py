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
