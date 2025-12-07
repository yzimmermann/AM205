# graph_generation/random_models.py
# Author: Yoel Zimmermann

import networkx as nx
from .spec import GraphSpec

def erdos_renyi_graph(spec: GraphSpec) -> nx.Graph | nx.DiGraph:
    p = spec.params.get("p", 0.01)
    return nx.erdos_renyi_graph(
        n=spec.n,
        p=p,
        seed=spec.seed,
        directed=spec.directed,
    )

def barabasi_albert_graph(spec: GraphSpec) -> nx.Graph:
    m = spec.params.get("m", 3)
    G = nx.barabasi_albert_graph(spec.n, m, seed=spec.seed)
    return G.to_directed() if spec.directed else G

def directed_scale_free_graph(spec: GraphSpec) -> nx.DiGraph:
    G = nx.scale_free_graph(
        spec.n,
        alpha=spec.params.get("alpha", 0.41),
        beta=spec.params.get("beta", 0.54),
        gamma=spec.params.get("gamma", 0.05),
        delta_in=spec.params.get("delta_in", 0.2),
        delta_out=spec.params.get("delta_out", 0.0),
        seed=spec.seed,
    )
    # strip multi-edges / self-loops:
    return nx.DiGraph(G)
