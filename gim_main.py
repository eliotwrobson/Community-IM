import itertools as it
import time

import pandas as pd

from dataset_manager import get_graph
from gim_im import gim_im


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

    results = []

    workloads = list(it.product(graphs, k_vals, a_vals, b_vals))
    i = 1

    for graph, k, a_val_tup, b_val_tup in workloads:
        print(f"Workload {i} / {len(workloads)}")
        i += 1
        start_time = time.perf_counter()
        discount_dict, frac_influence = gim_im(
            graph, k, a_vals=a_val_tup, b_vals=b_val_tup
        )
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
