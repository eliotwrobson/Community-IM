from cynetdiff.models import IndependentCascadeModel

import frac_influence as fi


def assemble_dict(nested_solution: list[int], budget: float) -> dict[int, float]:
    res_dict = {}
    nested_solution_iter = iter(nested_solution)

    while budget > 1.0:
        res_dict[next(nested_solution_iter)] = 1.0
        budget -= 1.0

    # TODO maybe floating point error here?
    res_dict[next(nested_solution_iter)] = budget

    return res_dict


def mle_selection(
    nested_solution: list[int],
    graph_model: IndependentCascadeModel,
    profit_per_node: float,
    desired_profit: float,
    *,
    num_trials=1_000,
    eps=0.5,
) -> float:
    lo = 0.0
    hi = float(len(nested_solution))

    while lo + eps < hi:
        mid_budget = (hi - lo) // 2 + lo

        budget_dict = assemble_dict(nested_solution, mid_budget)

        influence = fi.compute_fractional_influence(
            graph_model, budget_dict, num_trials=num_trials
        )
        total_profit = influence * profit_per_node

        if total_profit >= desired_profit:
            hi = mid_budget
        else:
            lo = mid_budget

    return lo
