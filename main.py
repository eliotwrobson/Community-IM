import math
import os

import networkx as nx
import pandas as pd
import tqdm
from cynetdiff.utils import networkx_to_ic_model

import dataset_manager as dm
import frac_influence as fi
import lim_im as li
import ris_selection as rs

NUM_TRIALS = 10_000


def make_temp_graph() -> nx.DiGraph:
    n = 1_000
    p = 0.01
    random_seed = 12345

    graph = nx.gnp_random_graph(n, p, seed=random_seed)
    graph.name = "Temp_graph"

    return graph


def get_mle_runtime(graph_name: str, budget: float) -> float:
    """
    Get MLE runtime for the given budget
    """

    int_budget = math.ceil(budget)

    if int_budget == 0:
        return 0.0

    ris_folder = "ris_code_release"
    out_filename = os.path.join(
        ris_folder, "result", f"{graph_name}_subsim_k{int_budget}_wc"
    )

    # Only run code if outfile does not exist for graph already.
    if not os.path.exists(out_filename):
        # Running lim software which is written in C and saving the outputs
        os.chdir(ris_folder)
        os.system(f"./subsim -func=format -gname={graph_name} -pdist=wc")
        # The vanilla here means use standard RR method
        os.system(
            f"./subsim -func=im -gname={graph_name} -seedsize={int_budget} -eps=0.01 -vanilla=1"
        )
        os.chdir("..")

    with open(out_filename, "r") as f:
        f_iter = iter(f.readlines())
        next(f_iter)  # Skip first line
        # Second line has time spent
        return float(next(f_iter).split()[2])


def fractional_im_experiments() -> None:
    """
    The goal of these experiments is to run the fractional IM algorithm against various other algorithms.
    """

    # Get graph to run experiments
    # TODO run every graph through this.
    graph = dm.get_graph("amazon")
    # graph = make_temp_graph()

    # First, run LIM code and get data
    lim_seeds, influence, lim_times = li.lim_im(graph)

    # Next, do RIS simulation for rounded budget
    # NOTE Hard-coded to 20 because this is hard-coded in the LIM code
    vertices, _ = rs.ris_im(graph, 20)

    model, _ = networkx_to_ic_model(graph)

    cd_budgets, cd_influences, cd_times = li.cd_im(graph)

    ud_budgets, ud_influences, ud_times = li.ud_im(graph)

    graph_runtime_info = []

    for seed_dict, lim_time in tqdm.tqdm(
        zip(lim_seeds, lim_times), total=len(lim_seeds)
    ):
        seed_vals = sorted(seed_dict.values(), reverse=True)
        mle_seed_dict = dict(zip(vertices, seed_vals))

        lim_influence = fi.compute_fractional_influence(
            model, seed_dict, num_trials=NUM_TRIALS
        )
        mle_influence = fi.compute_fractional_influence(
            model, mle_seed_dict, num_trials=NUM_TRIALS
        )
        budget = sum(seed_vals)

        graph_runtime_info.append(
            {
                "lim_influence": lim_influence,
                "lim_runtime": lim_time,
                "mle_influence": mle_influence,
                "mle_total_time": get_mle_runtime(graph.name, budget),
                "budget": budget,
                "num_samples": NUM_TRIALS,
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
