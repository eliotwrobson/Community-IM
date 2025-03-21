import random

from cynetdiff.models import IndependentCascadeModel


def compute_fractional_influence(
    model: IndependentCascadeModel,
    frac_alloc_dict: dict[int, float],
    # a_list: list[float],
    # b_list: list[float],
    *,
    random_seed: int = 12345,
    num_trials: int = 10_000,
) -> float:
    random.seed(random_seed)
    # TODO this stuff is weird
    # node_discount_probas = []
    # for node, y_val in frac_alloc_dict.items():
    #    node_discount_probas.append((node, a_list[node] * y_val + b_list[node]))

    total_activated = 0.0

    seeds, probs = list(zip(*frac_alloc_dict.items()))  # Unzip dict
    print(seeds)
    print(probs)
    exit()

    for _ in range(num_trials):
        model.reset_model()

        seeds = []
        for node, discount in frac_alloc_dict.items():
            if not (0.0 <= discount <= 1.0):
                raise Exception(f"Invalid discount {discount} for {node}.")

            if random.random() < discount:
                seeds.append(node)

        model.set_seeds(seeds)
        model.advance_until_completion()
        total_activated += model.get_num_activated_nodes()

    return total_activated / num_trials
