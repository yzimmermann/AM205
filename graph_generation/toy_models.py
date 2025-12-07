# graph_generation/toy_models.py
# Author: Yoel Zimmermann

import networkx as nx
import numpy as np
from .spec import GraphSpec

def grid_graph(spec: GraphSpec) -> nx.DiGraph:
    side = int(np.sqrt(spec.n))
    G = nx.grid_2d_graph(side, side)
    G = nx.convert_node_labels_to_integers(G)
    return G.to_directed()

def toy_reducible_graph(spec: GraphSpec) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2)])
    G.add_edges_from([(3, 4), (4, 5)])
    # maybe no edges from comp2 -> comp1 to make it reducible
    return G

def toy_dangling_graph(spec: GraphSpec) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_node(3)
    G.add_edge(3, 0)
    return G

def generate_toy_graph(spec: GraphSpec) -> nx.DiGraph:
    if spec.model == "grid":
        return grid_graph(spec)
    elif spec.model == "toy_reducible":
        return toy_reducible_graph(spec)
    elif spec.model == "toy_dangling":
        return toy_dangling_graph(spec)
    else:
        raise ValueError(f"Unknown toy model: {spec.model}")
