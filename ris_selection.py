import os

import networkx as nx


def ris_im(
    network: nx.Graph, budget: int, *, ris_folder: str = "ris_code_release"
) -> tuple[set[int], float]:
    # Set budget as len(network.nodes) if the budget > len(network.nodes)
    if budget > network.number_of_nodes():
        raise ValueError(
            f'Budget "{budget}" is larger than number of nodes in network "{network.name}".'
        )
    # Output filename for best seed set
    graph_folder = os.path.join(ris_folder, "graphInfo")
    graph_filename = os.path.join(graph_folder, network.name)
    # Only run code if graph file does not exist already.
    if not os.path.exists(graph_filename):
        # Writing graph to data file
        # f_folder = os.path.join(ris_folder, "graphInfo")
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder)

        # TODO only run the external program code below if it hasn't been run before.
        # file_name_string = network.name
        # fstr = Path(f_folder) / file_name_string

        with open(graph_filename, "w") as f:
            f.write(str(network.number_of_nodes()))
            f.write(" ")
            f.write(str(network.number_of_edges()))
            f.write("\n")
            for edge in network.edges():
                f.write(" ".join(str(x) for x in edge))
                f.write("\n")

    # Output filename for best seed set
    out_filename = os.path.join(
        ris_folder, "result", f"{network.name}_subsim_k{budget}_wc"
    )

    # Only run code if outfile does not exist for graph already.
    if not os.path.exists(out_filename):
        # Running lim software which is written in C and saving the outputs
        os.chdir(ris_folder)
        os.system("make")
        print("HERE")
        # https://github.com/abhishekumrawal/Fractional-IM/tree/main/greedy-approximation/ris_code_release
        os.system(f"./subsim -func=format -gname={network.name} -pdist=wc")
        # The vanilla here means use standard RR method
        os.system(
            f"./subsim -func=im -gname={network.name} -seedsize={budget} -eps=0.01 -vanilla=1"
        )
        os.chdir("..")

    # Read run time from output file
    with open(out_filename, "r") as f:
        f_iter = iter(f.readlines())
        next(f_iter)  # Skip first line
        # Second line has time spent
        time_taken = float(next(f_iter).split()[2])

    # Read seeds from seed file
    seed_filename = os.path.join(
        ris_folder, "result", "seed", f"seed_{network.name}_subsim_k{budget}_wc"
    )

    with open(seed_filename, "r") as f:
        seeds = {int(x) for x in f.readlines()}

    return seeds, time_taken

    # # Output filename for exp influences
    # out_filename_exp = os.path.join(
    #     lim_folder, "result", network.name + ".txt_cimm_eps=0.500000_group_0_new"
    # )

    # # Output filename for time
    # out_filename_time = os.path.join(
    #     lim_folder, "time", network.name + ".txt_cimm_eps=0.500000_group_0_new"
    # )

    # Getting the best seed sets (allocations) and exp influence
    # best_seed_sets = [{}]

    # for best_seed_set in open(out_filename).readlines():
    #     res_dict = {}

    #     for i, seed_val in enumerate(map(float, best_seed_set.split(" ")[:-1])):
    #         if seed_val > 0.0:
    #             res_dict[i] = seed_val

    #     best_seed_sets.append(res_dict)

    # # Getting the exp influences
    # exp_influence = [x.split(" ")[7] for x in open(out_filename_exp).readlines()]
    # exp_influence = [float(x) for x in exp_influence]

    # # Getting the runtimes (cumulative in seconds)
    # run_times = [float(x.split(" ")[7]) for x in open(out_filename_time).readlines()]
    # # TODO I don't think this cumulative part is necessary.
    # # run_times = np.cumsum([0] + [float(x) for x in run_times])

    # # Saving all runtimes to a text file
    # fstr = results_folder_runtime_files + os.sep + "runtime_info_lim_all.txt"
    # with open(fstr, "w") as f:
    #     for val in run_times:
    #         f.write(str(val))
    #         f.write("\n")

    return best_seed_sets, exp_influence[-1], run_times
