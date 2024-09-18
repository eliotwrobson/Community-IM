import os

import networkx as nx
import pandas as pd
import tqdm
from cynetdiff.utils import networkx_to_ic_model

import dataset_manager as dm
import frac_influence as fi
import lim_im as li
import ris_selection as rs


def make_temp_graph() -> nx.DiGraph:
    n = 1_000
    p = 0.01
    random_seed = 12345

    graph = nx.gnp_random_graph(n, p, seed=random_seed)
    graph.name = "Temp_graph"

    return graph


def fractional_im_experiments() -> None:
    """
    The goal of these experiments is to run the fractional IM algorithm against various other algorithms.
    """

    # Get graph to run experiments
    # TODO run every graph through this.
    graph = dm.get_graph("facebook")
    # graph = make_temp_graph()

    # First, run LIM code and get data
    lim_seeds, influence, lim_times = li.lim_im(graph)

    # Next, do RIS simulation for rounded budget
    # NOTE Hard-coded to 20 because this is hard-coded in the LIM code
    vertices, ris_runtime = rs.ris_im(graph, 20)

    model, _ = networkx_to_ic_model(graph)

    cd_budgets, cd_influences, cd_times = li.cd_im(graph)

    ud_budgets, ud_influences, ud_times = li.ud_im(graph)

    graph_runtime_info = []

    for seed_dict, lim_time in tqdm.tqdm(
        zip(lim_seeds, lim_times), total=len(lim_seeds)
    ):
        seed_vals = sorted(seed_dict.values(), reverse=True)
        mle_seed_dict = dict(zip(vertices, seed_vals))

        lim_influence = fi.compute_fractional_influence(model, seed_dict)
        mle_influence = fi.compute_fractional_influence(model, mle_seed_dict)
        budget = sum(seed_vals)

        graph_runtime_info.append(
            {
                "lim_influence": lim_influence,
                "lim_runtime": lim_time,
                "mle_influence": mle_influence,
                "mle_total_time": ris_runtime,
                "budget": budget,
            }
        )
        # print(seed_dict, mle_seed_dict)
        # print(lim_influence, mle_influence)
    df = pd.DataFrame(graph_runtime_info)
    df.to_csv("benchmark_results" + os.sep + f"{graph.name}_benchmark_results.csv")

    # Save ud heuristic stuff
    graph_ud_info = [
        {"ud_budget": ud_budget, "ud_influence": ud_influnece, "ud_time": ud_time}
        for ud_budget, ud_influnece, ud_time in zip(ud_budgets, ud_influences, ud_times)
    ]

    df = pd.DataFrame(graph_ud_info)
    df.to_csv("benchmark_results" + os.sep + f"{graph.name}_ud_results.csv")

    # Save cd heuristic stuff
    graph_cd_info = [
        {"cd_budget": cd_budget, "cd_influence": cd_influnece, "cd_time": cd_time}
        for cd_budget, cd_influnece, cd_time in zip(cd_budgets, cd_influences, cd_times)
    ]

    df = pd.DataFrame(graph_cd_info)
    df.to_csv("benchmark_results" + os.sep + f"{graph.name}_cd_results.csv")


def main() -> None:
    # import ris_selection as rs

    # n = 1_000
    # p = 0.01

    # graph = nx.gnp_random_graph(n, p)
    # graph.name = "Temp_graph_2"
    # res = rs.ris_im(graph, 10)
    # print(res)
    # exit()
    # TODO add different functions for each experiment. Then, from the command line,
    # an experiment can be selected.
    fractional_im_experiments()


if __name__ == "__main__":
    main()
