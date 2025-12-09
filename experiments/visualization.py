# experiments/visualization.py
# Author: Yoel Zimmermann

from graph_generation import GraphSpec, generate_graph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("thesis.mplstyle")

def get_base_specs():
    specs = [
        GraphSpec(
            model="erdos_renyi",
            n=0,  # will be overridden
            params={"p": 0.01},
            directed=True,
            seed=1,
            name="Erdős–Rényi"
        ),
        GraphSpec(
            model="barabasi_albert",
            n=0,  # overridden
            params={"m": 3},
            directed=True,
            seed=1,
            name="Barabási–Albert"
        ),
        GraphSpec(
            model="directed_scale_free",
            n=0,  # overridden
            params={},
            directed=True,
            seed=1,
            name="Directed Scale-Free"
        ),
        GraphSpec(
            model="barbell",
            n=0,  # overridden
            params={},
            directed=True,
            seed=1,
            name="Barbell"
        ),
    ]
    return specs


def with_size(base_spec: GraphSpec, n: int, seed: int | None = None) -> GraphSpec:
    return GraphSpec(
        model=base_spec.model,
        n=n,
        params=base_spec.params,
        directed=True,
        seed=base_spec.seed if seed is None else seed,
        name=base_spec.name,
    )



def visualize_small_layouts(n_small: int = 200):
    """
    For each graph model, generate a small graph (n_small) and plot
    a spring layout with node size/color ~ out-degree.
    """
    specs = get_base_specs()

    for base in specs:
        spec_small = with_size(base, n_small, seed=1)
        data = generate_graph(spec_small)
        G = data.G
        outdeg = dict(G.out_degree())
        indeg = dict(G.in_degree())
        node_color = [outdeg[v] for v in G.nodes()]
        node_size = [50 + 4 * indeg[v] for v in G.nodes()]

        pos = nx.spring_layout(G, seed=1)

        plt.figure(figsize=(5, 5))
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            node_color=node_color,
            cmap="viridis",
            alpha=0.85,
        )
        nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.3, width=0.5)
        plt.colorbar(nodes, label="Out-degree")
        plt.title(f"{spec_small.name} (n={n_small})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{spec_small.name}_layout_n{n_small}.pdf")

def visualize_adjacency_spy(n_medium: int = 2000):
    """
    For each graph model, generate a medium-size graph (n_medium)
    and plot the adjacency pattern using a spy-plot.
    """
    specs = get_base_specs()

    for base in specs:
        spec_med = with_size(base, n_medium, seed=0)
        data = generate_graph(spec_med)
        A = data.A.tocsr()

        plt.figure(figsize=(5, 5))
        plt.spy(A, markersize=1)
        plt.title(f"Adjacency pattern: {spec_med.name} (n={n_medium})")
        plt.tight_layout()
        plt.show()


def visualize_degree_distributions(n_large: int = 20000, bins: int = 50):
    """
    For each graph model, generate a large graph (n_large)
    and plot the out-degree distribution on log–log axes.
    """
    specs = get_base_specs()

    for base in specs:
        spec_large = with_size(base, n_large, seed=0)
        data = generate_graph(spec_large)
        G = data.G

        out_deg = np.array([d for _, d in G.out_degree()])

        plt.figure(figsize=(5, 4))
        # use density=False to count occurrences
        plt.hist(out_deg, bins=bins, log=False)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Out-degree")
        plt.ylabel("Count")
        plt.title(f"Out-degree distribution: {spec_large.name} (n={n_large})")
        plt.tight_layout()
        plt.savefig(f"{spec_large.name}_outdegree_dist_n{n_large}.pdf")



def main():
    visualize_small_layouts(n_small=200)

    #visualize_adjacency_spy(n_medium=2000)

    #visualize_degree_distributions(n_large=20000)


if __name__ == "__main__":
    main()