# import dataset_manager as dm
# import mle_greedy as mle

import networkx as nx
from cynetdiff.utils import networkx_to_ic_model

import lim_im as li
import mle_greedy as mg


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

    li
    mg
    # celf_set, _ = mg.mle_greedy(graph, 20)
    seeds, influence, res = li.lim_im(graph, 10)

    print(seeds)

    cynetdiff_model, _ = networkx_to_ic_model(graph)
    cynetdiff_model.set_seeds(seeds[-1].keys())
    num_trials = 10000
    spread = 0.0
    for _ in range(num_trials):
        cynetdiff_model.reset_model()
        cynetdiff_model.advance_until_completion()
        spread += cynetdiff_model.get_num_activated_nodes()

    print(spread / num_trials)


def main() -> None:
    import ris_selection as rs

    n = 1_000
    p = 0.01

    graph = nx.gnp_random_graph(n, p)
    graph.name = "Temp_graph_2"
    res = rs.ris_im(graph, 10)
    print(res)
    exit()
    # TODO add different functions for each experiment. Then, from the command line,
    # an experiment can be selected.
    fractional_im_experiments()


if __name__ == "__main__":
    main()
