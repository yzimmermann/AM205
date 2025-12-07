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

def barbell_graph(spec: GraphSpec) -> nx.DiGraph:
    # If GraphSpec only has n, split it
    if not hasattr(spec, "m1") or not hasattr(spec, "m2"):
        # Default: two cliques of size n//3 and a path of size n - 2*(n//3)
        m1 = spec.n // 3
        m2 = spec.n - 2 * m1
    else:
        m1 = spec.m1
        m2 = spec.m2
    G = nx.barbell_graph(m1, m2)
    return nx.DiGraph(G)

def generate_toy_graph(spec: GraphSpec) -> nx.DiGraph:
    if spec.model == "grid":
        return grid_graph(spec)
    elif spec.model == "toy_reducible":
        return toy_reducible_graph(spec)
    elif spec.model == "toy_dangling":
        return toy_dangling_graph(spec)
    elif spec.model == "barbell":
        return barbell_graph(spec)
    else:
        raise ValueError(f"Unknown toy model: {spec.model}")
