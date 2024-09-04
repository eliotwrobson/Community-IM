import networkx as nx
from cynetdiff.utils import networkx_to_ic_model

import dataset_manager as dm
import frac_influence as fi
import lim_im as li
import mle_greedy as mg
import ris_selection as rs

li
mg
dm


def fractional_im_experiments() -> None:
    """
    The goal of these experiments is to run the fractional IM algorithm against various other algorithms.
    """

    # Get graph to run experiments
    # TODO run every graph through this.
    # graph = dm.get_graph("deezer")

    n = 1_000
    p = 0.01
    random_seed = 12345

    graph = nx.gnp_random_graph(n, p, seed=random_seed)
    graph.name = "Temp_graph"

    # First, run LIM code and get data
    seeds, influence, times = li.lim_im(graph)

    # Next, do RIS simulation for rounded budget
    # NOTE Hard-coded to 20 because this is hard-coded in the LIM code
    vertices, runtime = rs.ris_im(graph, 20)

    model, _ = networkx_to_ic_model(graph)

    for seed_dict in seeds:
        seed_vals = sorted(seed_dict.values(), reverse=True)
        mle_seed_dict = dict(zip(vertices, seed_vals))

        lim_influence = fi.compute_fractional_influence(model, seed_dict)
        mle_influence = fi.compute_fractional_influence(model, mle_seed_dict)

        print(lim_influence, mle_influence)


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
