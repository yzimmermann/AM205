# graph_generation/transforms.py
# Author: Yoel Zimmermann

import numpy as np
import scipy.sparse as sp
import networkx as nx

def ensure_directed(G: nx.Graph | nx.DiGraph, directed: bool):
    if directed:
        return G if isinstance(G, nx.DiGraph) else G.to_directed()
    else:
        return G.to_undirected()

def remove_isolated_nodes(G: nx.Graph | nx.DiGraph):
    iso = list(nx.isolates(G))
    if not iso:
        return G
    G = G.copy()
    G.remove_nodes_from(iso)
    return G

def graph_to_adjacency(G: nx.Graph | nx.DiGraph) -> sp.csr_matrix:
    return nx.to_scipy_sparse_array(G, format="csr", dtype=float)

# Old version before profiling
"""
def adjacency_to_pagerank_matrix(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.tocsr()
    n = A.shape[0]

    out_deg = np.array(A.sum(axis=1)).flatten()
    is_dangling = out_deg == 0

    inv_deg = np.zeros_like(out_deg, dtype=float)
    inv_deg[~is_dangling] = 1.0 / out_deg[~is_dangling]

    D_inv = sp.diags(inv_deg)
    P = D_inv @ A

    if is_dangling.any():
        dangling_idx = np.where(is_dangling)[0]
        U = np.zeros((len(dangling_idx), n), dtype=float)
        U[:, :] = 1.0 / n
        U = sp.csr_matrix(U)
        P[dangling_idx, :] = U

    return P
"""

def adjacency_to_pagerank_matrix(A: sp.csr_matrix) -> sp.csr_matrix:
    P = A.tocsr(copy=True)
    n = P.shape[0]

    # Row sums = out-degrees
    out_deg = np.asarray(P.sum(axis=1)).ravel()
    is_dangling = out_deg == 0

    indptr = P.indptr
    data = P.data
    for i in range(n):
        if not is_dangling[i]:
            start, end = indptr[i], indptr[i + 1]
            if start < end:
                data[start:end] /= out_deg[i]

    # Handle dangling rows: set them to uniform 1/n
    if is_dangling.any():
        P = P.tolil()
        dangling_idx = np.where(is_dangling)[0]
        for i in dangling_idx:
            P.rows[i] = list(range(n))
            P.data[i] = [1.0 / n] * n
        P = P.tocsr()

    return P

def add_teleportation(P: sp.csr_matrix, alpha: float = 0.85) -> sp.csr_matrix:
    n = P.shape[0]
    J = np.ones((n, n), dtype=float) / n
    J = sp.csr_matrix(J)
    return alpha * P + (1 - alpha) * J
