import math
import typing as t

import networkx as nx

from celf import celf_im

DiffusionGraphT = t.Union[nx.Graph, nx.DiGraph]


def mle_greedy(
    graph: DiffusionGraphT, k: float, num_trials: int = 1_000
) -> tuple[list[int], float]:
    """
    Run MLE greedy algorithm and return the result.
    """

    k_floor = math.floor(k)

    S, influences = celf_im(graph, k_floor + 1, num_trials=num_trials)
    fractional_part = k - k_floor

    return S, fractional_part
