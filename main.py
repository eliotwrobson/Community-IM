# import dataset_manager as dm
# import mle_greedy as mle
import networkx as nx

import lim_im as li


def fractional_im_experiments() -> None:
    """
    The goal of these experiments is to run the fractional IM algorithm against various other algorithms.
    """
    # graph = dm.get_graph("youtube")

    # mle.mle_greedy(graph, 10.4)

    n = 1_000
    p = 0.01

    graph = nx.gnp_random_graph(n, p)
    graph.name = "Temp_graph"

    li.lim_im(graph, 10)
    print("Here")


def main() -> None:
    # TODO add different functions for each experiment. Then, from the command line,
    # an experiment can be selected.
    fractional_im_experiments()


if __name__ == "__main__":
    main()
