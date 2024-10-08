"""
Main file for running community IM experiments
"""

import igraph as ig
import leidenalg as la
import networkx as nx

import dataset_manager as dm


def get_partition(graph: nx.DiGraph) -> la.VertexPartition:
    """
    TODO add saving to file if this is slow for large graphs.
    """
    igraph_graph = ig.Graph.from_networkx(graph)
    return la.find_partition(igraph_graph, la.ModularityVertexPartition)


def reverse_partition(
    partition: la.VertexPartition,
) -> dict[int, int]:
    res_dict = {}

    for i, part in enumerate(partition):
        for vtx in part:
            res_dict[vtx] = i

    return res_dict


def main() -> None:
    """
    Method used in experiments consists of three steps.

    1. Get graph and split into communities.
    2. Run greedy selection algorithm on each community
    3. Use progressive budgeting to obtain the final solution.
    """

    # First, generate graph and partition
    graph = dm.get_graph("deezer")
    parts = get_partition(graph)

    # Next, remove inter-community edges from graph
    rev_partition_dict = reverse_partition(parts)

    graph_edges_removed = nx.subgraph_view(
        graph, filter_edge=lambda u, v: rev_partition_dict[u] == rev_partition_dict[v]
    )

    print(f"Graph number of edges: {graph.number_of_edges()}")
    print(f"Graph new number of edges: {graph_edges_removed.number_of_edges()}")

    # for part in parts:
    #    print(part)


if __name__ == "__main__":
    main()
