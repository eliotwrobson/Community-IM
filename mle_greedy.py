import math
import typing as t

import networkx as nx

from celf import celf_im

DiffusionGraphT = t.Union[nx.Graph, nx.DiGraph]


def mle_greedy(graph: DiffusionGraphT, k: float, num_trials: int = 1_000) -> None:
    k_floor = math.floor(k)

    celf_im(graph, k_floor, num_trials=num_trials)
