# graph_generation/spec.py
# Author: Yoel Zimmermann

from dataclasses import dataclass
from typing import Literal, Mapping, Any
import networkx as nx
import scipy.sparse as sp

GraphModel = Literal[
    "erdos_renyi",
    "barabasi_albert",
    "directed_scale_free",
    "stochastic_block",
    "grid",
    "toy_reducible",
    "toy_dangling",
    "barbell",
]

@dataclass
class GraphSpec:
    model: GraphModel
    n: int
    params: Mapping[str, Any]
    directed: bool = True
    seed: int | None = None
    name: str | None = None

@dataclass
class GraphData:
    spec: GraphSpec
    G: nx.Graph | nx.DiGraph
    A: sp.csr_matrix
    P: sp.csr_matrix
    metadata: dict
