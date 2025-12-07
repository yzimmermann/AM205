# graph_generation/factory.py
# Author: Yoel Zimmermann

from .spec import GraphSpec, GraphData
from . import random_models, toy_models, transforms

def generate_graph(spec: GraphSpec) -> GraphData:
    if spec.model == "erdos_renyi":
        G = random_models.erdos_renyi_graph(spec)
    elif spec.model == "barabasi_albert":
        G = random_models.barabasi_albert_graph(spec)
    elif spec.model == "directed_scale_free":
        G = random_models.directed_scale_free_graph(spec)
    elif spec.model in {"grid", "toy_reducible", "toy_dangling"}:
        G = toy_models.generate_toy_graph(spec)
    else:
        raise ValueError(f"Unknown graph model: {spec.model}")

    G = transforms.ensure_directed(G, spec.directed)
    G = transforms.remove_isolated_nodes(G)

    A = transforms.graph_to_adjacency(G)
    P = transforms.adjacency_to_pagerank_matrix(A)

    metadata = {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "model": spec.model,
        "params": dict(spec.params),
    }

    return GraphData(spec=spec, G=G, A=A, P=P, metadata=metadata)
