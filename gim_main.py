import itertools as it
import random
import time
from itertools import count

import networkx as nx
import pandas as pd
from cynetdiff.utils import networkx_to_ic_model

from dataset_manager import get_graph
from frac_influence import compute_fractional_influence_linear
from gim_im import gim_im

RANDOM_SEED = 12345


def set_weights_and_labels(
    w_vals: tuple[float, ...],
    a_vals: tuple[float, ...],
    b_vals: tuple[float, ...],
    graph: nx.DiGraph,
    seed: int,
) -> None:
    """
    Randomly assigns weights and labels to the nodes of the graph.
    The weights are assigned from the given weight values, and the labels
    are assigned from the given a and b values.
    """

    # Set the random seed for reproducibility
    random.seed(seed)

    # Assign weights and labels to nodes
    for _, data in graph.nodes(data=True):
        data["w"] = random.choice(w_vals)
        data["a"] = random.choice(a_vals)
        data["b"] = random.choice(b_vals)


def optimum_budget_selection(
    graph: nx.DiGraph,
    price_per_unit: float,
    cost_per_unit: float,
    *,
    max_budget: float = 20.0,
    epsilon: float = 0.1,
    random_seed: int,
    num_trials: int,
) -> tuple[float, float, float]:
    """
    For a given graph, selects the budget that gives the optimal payoff. The payoff is
    defined as:

        price per unit * influence - cost per unit * budget

    Uses the fact that solution is nested to find the optimal budget.
    """

    discount_dict, frac_influence = gim_im(graph, max_budget, random_seed=random_seed)
    model, _ = networkx_to_ic_model(graph, rng=random_seed)

    search_time = discount_dict[-1][2]

    def compute_payoff(curr_budget: float) -> float:
        # Compute the expected influence for the current budget
        frac_influence = compute_fractional_influence_linear(
            model,
            discount_dict,
            graph,
            budget=curr_budget,
            random_seed=random_seed,
            num_trials=num_trials,
        )
        return price_per_unit * frac_influence - cost_per_unit * curr_budget

    start_time = time.perf_counter()

    low = 0.0
    high = max_budget

    while low + epsilon < high:
        print(low, high)
        # Midpoint computations for ternary search
        mid1 = low + (high - low) / 3.0
        mid2 = high - (high - low) / 3.0

        # Calculate the payoff
        payoff1 = compute_payoff(mid1)
        payoff2 = compute_payoff(mid2)

        print("Mid1 with payoff: ", mid1, payoff1)
        print("Mid2 with payoff: ", mid2, payoff2)

        if payoff1 < payoff2:
            low = mid1
        else:
            high = mid2

    end_time = time.perf_counter()

    # Return the optimal budget and the corresponding payoff
    return high, compute_payoff(high), end_time - start_time, search_time


def main(num_trials: int, random_seed: int) -> None:
    graphs = [get_graph("wikipedia"), get_graph("facebook"), get_graph("deezer")]
    price_per_unit = 1.0
    cost_per_unit = 100.0
    max_budget = 8.0
    eps = 0.1

    results = []

    w_val_tup = (1.0, 0.5)
    a_val_tup = (1.0, 0.5)
    b_val_tup = (0.2, 0.0)

    for graph in graphs:
        print(f"Starting on graph {graph.name}")
        set_weights_and_labels(
            w_vals=w_val_tup,
            a_vals=a_val_tup,
            b_vals=b_val_tup,
            graph=graph,
            seed=random_seed,
        )

        budget, payoff, search_time, initial_time = optimum_budget_selection(
            graph,
            price_per_unit,
            cost_per_unit,
            max_budget=max_budget,
            random_seed=random_seed,
            epsilon=eps,
            num_trials=num_trials,
        )

        workload_dict = {
            "price per unit": price_per_unit,
            "cost per unit": cost_per_unit,
            "graph name": graph.name,
            "eps": eps,
            "num nodes": graph.number_of_nodes(),
            "num edges": graph.number_of_edges(),
            "num trials": num_trials,
            "budget": budget,
            "payoff": payoff,
            "max budget": max_budget,
            "a_vals": a_val_tup,
            "b_vals": b_val_tup,
            "w_vals": w_val_tup,
            "time taken": search_time,
            "initial time": initial_time,
            "weighting scheme": graph.weighting_scheme,
        }

        results.append(workload_dict)

        print(budget, payoff)

    df = pd.DataFrame(results)
    df.to_csv("optimal_budget_results.csv")


def main2(num_trials: int, random_seed: int) -> None:
    graphs = [
        get_graph("wikipedia"),
        get_graph("facebook"),
        get_graph("deezer"),
        get_graph("dblp"),
        get_graph("amazon"),
    ]
    k_vals = [7.0]  # , 5, 10, 15, 20]
    a_vals = [(1.0, 0.5)]  # [(1.0,), (1.0, 0.5)]
    b_vals = [(0.2, 0.0)]  # [(0.0,), (0.2, 0.0)]
    w_vals = [(1.0, 0.5)]  # [(1.0,), (1.0, 0.5)]

    results = []

    workloads = list(it.product(graphs, k_vals, a_vals, b_vals, w_vals))
    i = count(1)

    for graph, k, a_val_tup, b_val_tup, w_val_tup in workloads:
        print(f"Workload {next(i)} / {len(workloads)}")

        set_weights_and_labels(
            w_vals=w_val_tup,
            a_vals=a_val_tup,
            b_vals=b_val_tup,
            graph=graph,
            seed=random_seed,
        )

        discount_dict, _ = gim_im(
            graph, k, num_trials=num_trials // 10, random_seed=random_seed
        )
        model, _ = networkx_to_ic_model(graph, rng=random_seed)

        print("Starting influence / budget computations")

        curr_list = []
        for curr_budget_item in discount_dict:
            curr_time = curr_budget_item[2]
            curr_list.append(curr_budget_item)
            influence = compute_fractional_influence_linear(
                model,
                curr_list,
                graph,
                budget=None,
                num_trials=num_trials,
                random_seed=random_seed,
            )

            curr_budget = sum(item for _, item, _ in curr_list)

            workload_dict = {
                "graph name": graph.name,
                "num nodes": graph.number_of_nodes(),
                "num edges": graph.number_of_edges(),
                "num nodes in seed set": len(curr_list),
                "num trials": num_trials,
                "budget": curr_budget,
                "a_vals": a_val_tup,
                "b_vals": b_val_tup,
                "w_vals": w_val_tup,
                "fractional influence": influence,
                "time taken": curr_time,
                "weighting scheme": graph.weighting_scheme,
            }

            results.append(workload_dict)

    # print(discount_dict, frac_influence)
    df = pd.DataFrame(results)
    df.to_csv("simple_greedy_results.csv")


if __name__ == "__main__":
    main(num_trials=10_000, random_seed=RANDOM_SEED)
