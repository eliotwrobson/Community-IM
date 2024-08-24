import dataset_manager as dm
import mle_greedy as mle


def fractional_im_experiments() -> None:
    """
    The goal of these experiments is to run the fractional IM algorithm against various other algorithms.
    """
    graph = dm.get_graph("youtube")

    mle.mle_greedy(graph, 10.4)


def main() -> None:
    # TODO add different functions for each experiment. Then, from the command line,
    # an experiment can be selected.
    fractional_im_experiments()


if __name__ == "__main__":
    main()
