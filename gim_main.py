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
    seed: int = 12345,
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
    random_seed: int = 12345,
) -> tuple[float, float]:
    """
    For a given graph, selects the budget that gives the optimal payoff. The payoff is
    defined as:

        price per unit * influence - cost per unit * budget

    Uses the fact that solution is nested to find the optimal budget.
    """

    discount_dict, frac_influence = gim_im(graph, max_budget, random_seed=random_seed)
    model, _ = networkx_to_ic_model(graph, rng=random_seed)

    low = 0.0
    high = max_budget

    while low + epsilon < high:
        print(low, high)
        # Midpoint computations for ternary search
        mid1 = low + (high - low) / 3.0
        mid2 = high - (high - low) / 3.0

        frac_influence1 = compute_fractional_influence_linear(
            model, discount_dict, graph, budget=mid1
        )
        frac_influence2 = compute_fractional_influence_linear(
            model, discount_dict, graph, budget=mid2
        )

        # Calculate the payoff
        payoff1 = price_per_unit * frac_influence1 - cost_per_unit * mid1
        payoff2 = price_per_unit * frac_influence2 - cost_per_unit * mid2

        if payoff1 < payoff2:
            low = mid1
        else:
            high = mid2

    # Return the optimal budget and the corresponding payoff
    return high, compute_fractional_influence_linear(
        model, discount_dict, graph, budget=high
    )


def main() -> None:
    graph = get_graph("wikipedia")
    price_per_unit = 1.0
    cost_per_unit = 0.5
    max_budget = 5.0
    eps = 0.1

    set_weights_and_labels(
        w_vals=(1.0, 0.5),
        a_vals=(1.0, 0.5),
        b_vals=(0.2, 0.0),
        graph=graph,
        seed=RANDOM_SEED,
    )

    res = optimum_budget_selection(
        graph,
        price_per_unit,
        cost_per_unit,
        max_budget=max_budget,
        random_seed=RANDOM_SEED,
        epsilon=eps,
    )

    print(res)


def main2() -> None:
    graphs = [get_graph("wikipedia")]  # , get_graph("facebook"), get_graph("deezer")]
    k_vals = [
        0,
        0.5,
        1,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
    ]  # , 5, 10, 15, 20]
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
            seed=RANDOM_SEED,
        )

        start_time = time.perf_counter()

        discount_dict, frac_influence = gim_im(graph, k)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        workload_dict = {
            "graph name": graph.name,
            "num nodes": graph.number_of_nodes(),
            "num edges": graph.number_of_edges(),
            "k": k,
            "a_vals": a_val_tup,
            "b_vals": b_val_tup,
            "fractional influence": frac_influence,
            "time taken": time_taken,
            "weighting scheme": graph.weighting_scheme,
        }

        results.append(workload_dict)

    # print(discount_dict, frac_influence)
    df = pd.DataFrame(results)
    df.to_csv("simple_greedy_results.csv")


if __name__ == "__main__":
    main()
