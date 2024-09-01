import json
import logging
import os
import pickle
import timeit
from pathlib import Path

import networkx as nx
import numpy as np


def lim_im(
    network: nx.Graph,
    budget: int,
    *,
    lim_folder: str = "lim_code_release",  # TODO change to pathlib path
):
    # Set budget as len(network.nodes) if the budget > len(network.nodes)
    if budget > network.number_of_nodes():
        raise ValueError(
            f'Budget "{budget}" is larger than number of nodes in network "{network.name}".'
        )

    # TODO maybe turn the below into a helper function?

    # creating pickle files folder within the results folder
    results_folder_pickle_files = os.path.join(
        "results", f"results_{network.name}", "pickle_files"
    )

    if not os.path.exists(results_folder_pickle_files):
        os.makedirs(results_folder_pickle_files)

    # creating log files folder within the results folder
    results_folder_log_files = os.path.join(
        "results", f"results_{network.name}", "log_files"
    )
    if not os.path.exists(results_folder_log_files):
        os.makedirs(results_folder_log_files)

    # creating runtime files folder within the results folder
    results_folder_runtime_files = os.path.join(
        "results", f"results_{network.name}", "runtime_files"
    )
    if not os.path.exists(results_folder_runtime_files):
        os.makedirs(results_folder_runtime_files)

    # Generating input for lim, i.e., network.name[1:]: the actual diretced graph

    # Number of nodes and number of edges
    num_nodes = str(network.number_of_nodes())
    num_edges = str(network.number_of_edges())

    # Weighted edges
    weighted_edges = []
    for edge in network.edges.data("activation_prob"):
        weighted_edges.append(list(edge))

    # Output filename for best seed set
    out_filename_allocation = os.path.join(
        lim_folder, "allocation", network.name + ".txt_cimm_eps=0.500000_group_0_new"
    )

    # Only run code if outfile does not exist for graph already.
    if not os.path.exists(out_filename_allocation):
        # Writing graph to data file
        f_folder = os.path.join(lim_folder, "data")
        if not os.path.exists(f_folder):
            os.makedirs(f_folder)

        # TODO only run the external program code below if it hasn't been run before.
        file_name_string = f"{network.name}.txt"
        fstr = Path(f_folder) / file_name_string

        with open(fstr, "w") as f:
            f.write(num_nodes)
            f.write("\n")
            f.write(num_edges)
            f.write("\n")
            for weighted_edge in weighted_edges:
                f.write(" ".join(str(x) for x in weighted_edge))
                f.write("\n")

        # Running lim software which is written in C and saving the outputs
        os.chdir(lim_folder + "/src")
        os.system("make")
        start = timeit.default_timer()
        # TODO algorithm is hardcoded. Add as a parameter with a key.
        os.system(f"./run 1 {file_name_string}")
        end = timeit.default_timer()
        runtime = end - start
        os.chdir("..")
        os.chdir("..")

        # Saving runtime info to a text file
        runtime_info = {"lim": runtime}
        fstr = results_folder_runtime_files + os.sep + "runtime_info_lim.txt"
        with open(fstr, "w") as f:
            f.write(json.dumps(runtime_info))

    # Output filename for exp influences
    out_filename_exp = os.path.join(
        lim_folder, "result", network.name + ".txt_cimm_eps=0.500000_group_0_new"
    )

    # Output filename for time
    out_filename_time = os.path.join(
        lim_folder, "time", network.name + ".txt_cimm_eps=0.500000_group_0_new"
    )

    # Getting the best seed sets (allocations) and exp influence
    best_seed_sets = [[float(0) for x in range(network.number_of_nodes())]]

    for best_seed_set in open(out_filename_allocation).readlines():
        best_seed_sets.append([float(x) for x in best_seed_set.split(" ")[:-1]])

    # Getting the exp influences
    exp_influence = [x.split(" ")[7] for x in open(out_filename_exp).readlines()]
    exp_influence = [float(x) for x in exp_influence]

    # Getting the runtimes (cumulative in seconds)
    run_times = [x.split(" ")[7] for x in open(out_filename_time).readlines()]
    # TODO I don't think this cumulative part is necessary.
    run_times = np.cumsum([0] + [float(x) for x in run_times])

    print(best_seed_sets)
    print(exp_influence)
    print(run_times)
    #    exit()

    # Saving all runtimes to a text file
    fstr = results_folder_runtime_files + os.sep + "runtime_info_lim_all.txt"
    with open(fstr, "w") as f:
        for val in run_times:
            f.write(str(val))
            f.write("\n")

    results = {
        "budget": budget,
        # "diffusion_model": diffusion_model,
        "algorithm": "lim",
        # "n_sim": n_sim,
        "best_seed_set": best_seed_sets[-1],
        "network_name": network.name,
        "exp_influence": exp_influence[-1],
    }

    fstr = results_folder_pickle_files + os.sep + "output_lim__%i__.pkl" % (budget)
    with open(fstr, "wb") as f:
        pickle.dump(results, f)

    logging.info("The estimated exp influence is as follows.")
    logging.info(str(exp_influence[-1]))

    return best_seed_set, exp_influence[-1], runtimes
