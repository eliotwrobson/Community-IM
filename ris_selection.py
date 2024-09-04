import os

import networkx as nx


def ris_im(
    network: nx.Graph, budget: int, *, ris_folder: str = "ris_code_release"
) -> tuple[list[int], float]:
    # TODO add ability to have different weighting schemes.
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

    # NOTE
    # Greedy selection from first to least most seen
    with open(seed_filename, "r") as f:
        seeds = [int(x) for x in f.readlines()]

    return seeds, time_taken
