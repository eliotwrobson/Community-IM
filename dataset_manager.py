"""
Retrieves and processes raw graphs from SNAP.
"""

import gzip
import zipfile

import networkx as nx
import pooch
from cynetdiff.utils import set_activation_weighted_cascade


def process_deezer(file_path: str) -> nx.DiGraph:
    with zipfile.ZipFile(file_path, "r") as zfile:
        with zfile.open("deezer_europe/deezer_europe_edges.csv") as edgefile:
            line_iter = iter(edgefile.readlines())

            # Skip first line because it's just schema
            next(line_iter)

            deezer_graph = nx.from_edgelist(
                tuple(map(int, line.strip().split(b","))) for line in line_iter
            )

    deezer_graph = deezer_graph.to_directed()
    set_activation_weighted_cascade(deezer_graph)

    return deezer_graph


def process_facebook(file_path: str) -> nx.DiGraph:
    with gzip.open(file_path, "r") as f:
        facebook_graph = nx.from_edgelist(
            tuple(map(int, line.split())) for line in f.readlines()
        )

    facebook_graph = facebook_graph.to_directed()
    set_activation_weighted_cascade(facebook_graph)
    return facebook_graph


def process_wikipedia(file_path: str) -> nx.DiGraph:
    """
    TODO switch to using iterators for this
    """
    edge_list = []

    with gzip.open(file_path, "r") as f:
        for line in f.readlines():
            line_list = line.split()

            if line_list[0] == b"#":
                continue

            edge_list.append(tuple(map(int, line_list)))

    wikipedia_graph = nx.convert_node_labels_to_integers(
        nx.from_edgelist(edge_list).to_directed()
    )
    set_activation_weighted_cascade(wikipedia_graph)

    return wikipedia_graph


def process_epinions1(file_path: str) -> nx.DiGraph:
    with gzip.open(file_path, "r") as f:
        line_iter = iter(f.readlines())

        # Skip first 4 lines because they're comments
        for _ in range(4):
            next(line_iter)

        epinions1_graph = nx.from_edgelist(
            tuple(map(int, line.strip().split())) for line in line_iter
        ).to_directed()

    set_activation_weighted_cascade(epinions1_graph)
    return epinions1_graph


def process_sgn_epinions(file_path: str) -> nx.DiGraph:
    with gzip.open(file_path, "r") as f:
        line_iter = iter(f.readlines())

        # Skip first 4 lines because they're comments
        for _ in range(4):
            next(line_iter)

        sgn_epinions_graph = nx.from_edgelist(
            tuple(map(int, line.strip().split()[:2])) for line in line_iter
        ).to_directed()

    # Remove some self-loop edges.
    sgn_epinions_graph.remove_edges_from(nx.selfloop_edges(sgn_epinions_graph))

    set_activation_weighted_cascade(sgn_epinions_graph)
    return sgn_epinions_graph


def process_youtube(file_path: str) -> nx.DiGraph:
    """
    TODO this is really slow. Instead of making a digraph, directly turn this into CSR arrays.
    """

    with gzip.open(file_path, "r") as f:
        line_iter = iter(f.readlines())

        # Skip first 4 lines because they're comments
        for _ in range(4):
            next(line_iter)

        youtube_graph = nx.convert_node_labels_to_integers(
            nx.from_edgelist(
                tuple(map(int, line.strip().split())) for line in line_iter
            ).to_directed()
        )

    set_activation_weighted_cascade(youtube_graph)
    return youtube_graph


def get_graph(dataset_name: str) -> nx.DiGraph:
    DATASETS = {
        "facebook": {
            "link": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
            "processor": process_facebook,
            "hash": "125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7",
        },
        "wikipedia": {
            "link": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
            "processor": process_wikipedia,
            "hash": "7d3e53626e14b8b09fb3b396bece9d481ad606bd64ceab066349ff57d4ada7fc",
        },
        "deezer": {
            "link": "https://snap.stanford.edu/data/deezer_europe.zip",
            "processor": process_deezer,
            "hash": "dd66a73f8d8690b5bc300ba378883fb2c2f6316aec8917b6a2428e352fc9e498",
        },
        "epinions1": {
            "link": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
            "processor": process_epinions1,
            "hash": "69a2dab71fa5e3a0715487599fc16ca17ddc847379325a6c765bbad6e3e36938",
        },
        "sgn_epinions": {
            "link": "https://snap.stanford.edu/data/soc-sign-epinions.txt.gz",
            "processor": process_sgn_epinions,
            "hash": "214513a32f1375695ceab7d7581e463ebe44daa573492de699b0c5d1cf3dde60",
        },
        "youtube": {
            "link": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
            "processor": process_youtube,
            "hash": "dff1b97ba7d2fa9c59884b67dcd2275e717ff9501f86ed82ce6582ed4971f3e0",
        },
    }

    dataset_info = DATASETS[dataset_name]
    data_path = pooch.retrieve(
        url=dataset_info["link"],
        known_hash=dataset_info["hash"],
        progressbar=True,
    )

    graph = dataset_info["processor"](data_path)
    graph.name = dataset_name
    return graph