import json
import os
import timeit
from pathlib import Path

import networkx as nx


def cd_im(
    network: nx.Graph,
    *,
    lim_folder: str = "lim_code_release",  # TODO change to pathlib path
) -> tuple[list[int], list[float], list[float]]:
    # TODO Add this to the LIM code and take in the budget as a parameter.
    budget = 20
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

    # Output filename for exp influences
    # NOTE this is a different directory than the other result folder

    out_filename_exp = os.path.join(
        lim_folder, "result", network.name + ".txt_cd_eps=5e-1"
    )

    # Only run code if outfile does not exist for graph already.
    if not os.path.exists(out_filename_exp):
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
        os.system(f"./run 3 {file_name_string}")
        end = timeit.default_timer()
        runtime = end - start
        os.chdir("..")
        os.chdir("..")

        # # Saving runtime info to a text file
        # runtime_info = {"lim": runtime}
        # fstr = results_folder_runtime_files + os.sep + "runtime_info_lim.txt"
        # with open(fstr, "w") as f:
        #     f.write(json.dumps(runtime_info))

    # Output filename for time
    out_filename_time = os.path.join(
        lim_folder, "time", network.name + ".txt_cd_eps=5e-1"
    )

    # Getting the exp influences
    exp_influence = []
    budgets = []

    for x in open(out_filename_exp).readlines():
        x_list = x.split()

        exp_influence.append(float(x_list[-1]))
        budgets.append(int(x_list[1]))

    # Getting the runtimes (cumulative in seconds)
    run_times = [float(x.split(" ")[-3]) for x in open(out_filename_time).readlines()]

    return budgets, exp_influence, run_times


def ud_im(
    network: nx.Graph,
    *,
    lim_folder: str = "lim_code_release",  # TODO change to pathlib path
) -> tuple[list[int], list[float], list[float]]:
    # TODO Add this to the LIM code and take in the budget as a parameter.
    budget = 20
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

    # Output filename for exp influences
    # NOTE this is a different directory than the other result folder

    out_filename_exp = os.path.join(
        lim_folder, "result", network.name + ".txt_hd_M=200"
    )

    # Only run code if outfile does not exist for graph already.
    if not os.path.exists(out_filename_exp):
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
        os.system(f"./run 4 {file_name_string}")
        end = timeit.default_timer()
        runtime = end - start
        os.chdir("..")
        os.chdir("..")

        # # Saving runtime info to a text file
        # runtime_info = {"lim": runtime}
        # fstr = results_folder_runtime_files + os.sep + "runtime_info_lim.txt"
        # with open(fstr, "w") as f:
        #     f.write(json.dumps(runtime_info))

    # Output filename for time
    out_filename_time = os.path.join(lim_folder, "time", network.name + ".txt_hd_M=200")

    # Getting the exp influences
    exp_influence = []
    budgets = []

    for x in open(out_filename_exp).readlines():
        x_list = x.split()

        exp_influence.append(float(x_list[-1]))
        budgets.append(int(x_list[1]))

    # Getting the runtimes (cumulative in seconds)
    run_times = [float(x.split(" ")[-1]) for x in open(out_filename_time).readlines()]

    return budgets, exp_influence, run_times


def lim_im(
    network: nx.Graph,
    *,
    lim_folder: str = "lim_code_release",  # TODO change to pathlib path
):
    # TODO Add this to the LIM code and take in the budget as a parameter.
    budget = 20
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
    best_seed_sets = [{}]

    for best_seed_set in open(out_filename_allocation).readlines():
        res_dict = {}

        for i, seed_val in enumerate(map(float, best_seed_set.split(" ")[:-1])):
            if seed_val > 0.0:
                res_dict[i] = seed_val

        best_seed_sets.append(res_dict)

    # Getting the exp influences
    exp_influence = [float(x.split(" ")[7]) for x in open(out_filename_exp).readlines()]

    # Getting the runtimes (cumulative in seconds)
    run_times = [float(x.split(" ")[7]) for x in open(out_filename_time).readlines()]
    # TODO I don't think this cumulative part is necessary.
    # run_times = np.cumsum([0] + [float(x) for x in run_times])

    # Saving all runtimes to a text file
    # fstr = results_folder_runtime_files + os.sep + "runtime_info_lim_all.txt"
    # with open(fstr, "w") as f:
    #     for val in run_times:
    #         f.write(str(val))
    #         f.write("\n")

    return best_seed_sets, exp_influence[-1], run_times
