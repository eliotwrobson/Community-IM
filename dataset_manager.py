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

            deezer_graph = nx.Graph(
                tuple(map(int, line.strip().split(b","))) for line in line_iter
            )

    # print(deezer_graph.number_of_edges(), deezer_graph.number_of_nodes())
    deezer_graph = deezer_graph.to_directed()
    set_activation_weighted_cascade(deezer_graph)

    return deezer_graph


# TODO I think the exception is happening here? Somewhere in this file at least
def process_facebook(file_path: str) -> nx.DiGraph:
    with gzip.open(file_path, "r") as f:
        facebook_graph = nx.Graph(
            tuple(map(int, line.split())) for line in f.readlines()
        )

    # facebook_graph.name = "facebook"
    facebook_graph = facebook_graph.to_directed()
    print("thing")
    set_activation_weighted_cascade(facebook_graph)
    print("other thing")
    return facebook_graph


def process_wikipedia(file_path: str) -> nx.DiGraph:
    edge_list = []

    with gzip.open(file_path, "r") as f:
        for line in f.readlines():
            line_list = line.split()

            if line_list[0] == b"#":
                continue

            edge_list.append(tuple(map(int, line_list)))

    wikipedia_graph = nx.DiGraph(edge_list)
    # wikipedia_graph.name = "wikipedia"
    set_activation_weighted_cascade(wikipedia_graph)

    return wikipedia_graph


def get_graph(dataset_name: str):
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
    }

    dataset_info = DATASETS[dataset_name]
    data_path = pooch.retrieve(
        url=dataset_info["link"],
        known_hash=dataset_info["hash"],
        progressbar=True,
    )
    print("here?")
    graph = dataset_info["processor"](data_path)
    print("doen")
    graph.name = dataset_name
    return graph
