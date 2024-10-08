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


def compute_community_aware_diffusion_degrees(
    graph: nx.DiGraph, rev_partition_dict: dict[int, int]
) -> dict[int, int]:
    graph_no_community_edges = nx.subgraph_view(
        graph, filter_edge=lambda u, v: rev_partition_dict[u] != rev_partition_dict[v]
    )

    # TODO add temp dict here

    for start_node in graph_no_community_edges:
        for neighbor in graph_no_community_edges.neighbors(start_node):
            # Add to probability
            graph_no_community_edges[start_node][neighbor]["activation_prob"]

            for second_neighbor in graph_no_community_edges.neighbors(neighbor):
                if second_neighbor == start_node:  # Avoid going back to the start node
                    continue

                graph_no_community_edges[start_node][neighbor]["activation_prob"]
                graph_no_community_edges[neighbor][second_neighbor]["activation_prob"]

                path = (start_node, neighbor, second_neighbor)
                print(path)

    return {}


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

    graph_only_community_edges = nx.subgraph_view(
        graph, filter_edge=lambda u, v: rev_partition_dict[u] == rev_partition_dict[v]
    )
    compute_community_aware_diffusion_degrees(graph, rev_partition_dict)
    print(f"Graph number of edges: {graph.number_of_edges()}")
    print(f"Graph new number of edges: {graph_only_community_edges.number_of_edges()}")

    # for part in parts:
    #    print(part)


if __name__ == "__main__":
    main()
