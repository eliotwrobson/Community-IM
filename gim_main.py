import itertools as it
import random
import time
from itertools import count

import networkx as nx
import pandas as pd

from dataset_manager import get_graph
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


def main() -> None:
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
