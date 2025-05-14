import time

import networkx as nx
import pandas as pd
from cynetdiff.models import IndependentCascadeModel
from cynetdiff.utils import networkx_to_ic_model, set_activation_weighted_cascade

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
    num_trials=10_000,
    eps=0.1,
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

    budget_dict = assemble_dict(nested_solution, hi)

    influence = fi.compute_fractional_influence(
        graph_model, budget_dict, num_trials=num_trials
    )

    return hi, budget_dict, influence * profit_per_node


def selection_im_experiments() -> None:
    graphs = [
        dm.get_graph("wikipedia"),
        dm.get_graph("facebook"),
        dm.get_graph("deezer"),
        dm.get_graph("amazon"),
        dm.get_graph("dblp"),
    ]

    payoffs = [100, 200, 500, 1000]

    result_dicts = []
    eps = 0.1

    for graph in graphs:
        print(f"starting selection on graph {graph.name}")
        model, _ = networkx_to_ic_model(graph)
        vertices, ris_time = rs.ris_im(graph, 20)
        for payoff in payoffs:
            print("Running algo")
            start = time.perf_counter()
            budget, selection, total_profit = mle_selection(
                vertices, model, 1.0, payoff, eps=eps
            )
            end = time.perf_counter()

            result_dicts.append(
                {
                    "graph": graph.name,
                    "desired profit": payoff,
                    "actual profit": total_profit,
                    "used budget": budget,
                    "time taken": end - start,
                    "eps": eps,
                    "ris_time": ris_time,
                }
            )

    # From result dicts, turn into a CSV
    df = pd.DataFrame(result_dicts)
    df.to_csv("mle_selection_benchmark_results.csv")


### Start of cost-benefit search code ###


def cost_benefit_search(
    nested_solution: list[int],
    graph_model: IndependentCascadeModel,
    profit_per_node: float,
    cost_per_unit: float,
    *,
    num_trials=10_000,
    eps=0.1,
) -> tuple[float, dict[int, float], float]:
    """
    Perform a cost-benefit search using ternary search to find the budget
    that achieves the highest profit.
    """

    def compute_profit(budget: float) -> float:
        budget_dict = assemble_dict(nested_solution, budget)
        influence = fi.compute_fractional_influence(
            graph_model, budget_dict, num_trials=num_trials
        )
        return influence * profit_per_node - budget * cost_per_unit

    lo = 0.0
    hi = float(len(nested_solution))

    while lo + eps < hi:
        print(hi - lo, eps)

        m_1 = lo + (hi - lo) / 3
        m_2 = hi - (hi - lo) / 3
        profit_1 = compute_profit(m_1)
        profit_2 = compute_profit(m_2)

        print("Profits:", profit_1, profit_2)

        if profit_1 < profit_2:
            lo = m_1
        else:
            hi = m_2

    budget_dict = assemble_dict(nested_solution, hi)

    return hi, budget_dict, compute_profit(hi)


def tradeoff_im_experiments() -> None:
    graphs = [
        dm.get_graph("amazon"),
        dm.get_graph("dblp"),
        dm.get_graph("deezer"),
        dm.get_graph("facebook"),
        dm.get_graph("wikipedia"),
    ]

    result_dicts = []
    eps = 0.1

    for graph in graphs:
        print(f"starting selection on graph {graph.name}")
        model, _ = networkx_to_ic_model(graph)
        vertices, ris_time = rs.ris_im(graph, 20)

        print("Running algo")

        profit_per_node = 1.0
        cost_per_unit = 100.0

        start = time.perf_counter()
        budget, selection, final_profit = cost_benefit_search(
            vertices, model, profit_per_node, cost_per_unit, eps=eps
        )
        end = time.perf_counter()

        result_dicts.append(
            {
                "graph": graph.name,
                "final profit": final_profit,
                "profit per node": profit_per_node,
                "cost per unit": cost_per_unit,
                "used budget": budget,
                "time taken": end - start,
                "eps": eps,
                "ris_time": ris_time,
            }
        )

    # From result dicts, turn into a CSV
    df = pd.DataFrame(result_dicts)
    df.to_csv("tradeoff_selection_benchmark_results.csv")


def florentine_families_experiment() -> None:
    florentine_families = nx.florentine_families_graph().to_directed()

    set_activation_weighted_cascade(florentine_families)
    florentine_families.name = "florentine_families"
    florentine_families.weighting_scheme = "weighted_cascade"

    model, node_mapping = networkx_to_ic_model(florentine_families)
    vertices = [1, 12, 9]
    # print(node_mapping)
    # exit()

    eps = 0.1

    back_map = {
        v: k for k, v in node_mapping.items()
    }  # Map from model node IDs to original graph node IDs

    def convert_selection(selection: dict[int, float]) -> dict[int, float]:
        """Convert the selection to the original graph node IDs."""
        return {back_map[k]: v for k, v in selection.items()}

    # selection = dict(zip(vertices, [1.0, 1.0, 1.0]))
    # influence = fi.compute_fractional_influence(model, selection, num_trials=10_000)
    # print(convert_selection(selection), influence)
    # exit()

    # First, do cost-benefit search with the florentine families graph
    # using a profit per node of 1.0 and a cost per unit of 100.0
    profit_per_node = 1.0
    cost_per_unit = 2.0

    print("Running cost-benefit search")

    start = time.perf_counter()
    budget, selection, final_profit = cost_benefit_search(
        vertices, model, profit_per_node, cost_per_unit, eps=eps
    )
    end = time.perf_counter()

    selection = convert_selection(selection)  # Convert to original node IDs

    print(
        f"Florentine Families Experiment:\n"
        f"Final profit: {final_profit}\n"
        f"Profit per node: {profit_per_node}\n"
        f"Cost per unit: {cost_per_unit}\n"
        f"Used budget: {budget}\n"
        f"Time taken: {end - start}\n"
        f"Epsilon: {eps}\n"
        f"Selection: {selection}\n"
    )

    # Now, do the MLE selection with the florentine families graph
    # using a desired profit of 9
    desired_profit = 9.0
    start = time.perf_counter()
    budget, selection, total_profit = mle_selection(
        vertices, model, 1.0, desired_profit, eps=eps
    )
    end = time.perf_counter()

    selection = convert_selection(selection)  # Convert to original node IDs

    print(
        f"Florentine Families Experiment (MLE Selection):\n"
        f"Desired profit: {desired_profit}\n"
        f"Actual profit: {total_profit}\n"
        f"Used budget: {budget}\n"
        f"Time taken: {end - start}\n"
        f"Epsilon: {eps}\n"
        f"Selection: {selection}\n"
    )


if __name__ == "__main__":
    tradeoff_im_experiments()
