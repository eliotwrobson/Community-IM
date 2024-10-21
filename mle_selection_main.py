from cynetdiff.models import IndependentCascadeModel
from cynetdiff.utils import networkx_to_ic_model

import dataset_manager as dm
import frac_influence as fi
import ris_selection as rs


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
    eps=0.2,
) -> tuple[float, dict[int, float], float]:
    lo = 0.0
    hi = float(len(nested_solution))

    while lo + eps < hi:
        print(lo, hi)
        mid_budget = (hi - lo) / 2 + lo

        budget_dict = assemble_dict(nested_solution, mid_budget)

        influence = fi.compute_fractional_influence(
            graph_model, budget_dict, num_trials=num_trials
        )
        total_profit = influence * profit_per_node

        if total_profit >= desired_profit:
            hi = mid_budget
        else:
            lo = mid_budget

    # NOTE this code makes sure that the final output is the right thing (we want lo, not hi _I think_)
    # final_profit = fi.compute_fractional_influence(
    #     graph_model, budget_dict, num_trials=num_trials
    # )
    # total_profit = final_profit * profit_per_node
    # print(total_profit)

    budget_dict = assemble_dict(nested_solution, lo)

    influence = fi.compute_fractional_influence(
        graph_model, budget_dict, num_trials=num_trials
    )

    return lo, budget_dict, total_profit


def selection_im_experiments() -> None:
    graphs = [
        dm.get_graph("wikipedia"),
        dm.get_graph("facebook"),
        dm.get_graph("deezer"),
        dm.get_graph("amazon"),
        dm.get_graph("dblp"),
    ]

    payoffs = [100, 200, 500, 1000]

    for graph in graphs:
        print(f"starting selection on graph {graph.name}")
        model, _ = networkx_to_ic_model(graph)
        vertices, _ = rs.ris_im(graph, 100)
        for payoff in payoffs:
            print("Running algo")
            budget, selection, total_profit = mle_selection(vertices, model, 1.0, 400)
            print(budget, total_profit)


if __name__ == "__main__":
    selection_im_experiments()
