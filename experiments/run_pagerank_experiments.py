# experiments/run_pagerank_experiments.py
# Author: Yoel Zimmermann

from graph_generation import GraphSpec, generate_graph
from pagerank import PowerMethod
import matplotlib.pyplot as plt

def main():
    specs = [
        GraphSpec(
            model="erdos_renyi",
            n=2000,
            params={"p": 0.003},
            directed=True,
            seed=0,
            name="ER"
        ),
        GraphSpec(
            model="barabasi_albert",
            n=2000,
            params={"m": 3},
            directed=True,
            seed=0,
            name="BA"
        ),
        GraphSpec(
            model="directed_scale_free",
            n=2000,
            params={},  # use defaults
            directed=True,
            seed=0,
            name="ScaleFree"
        ),
    ]

    for spec in specs:
        data = generate_graph(spec)
        P_row = data.P

        # Now redundant:

        # add teleportation
        # from graph_generation.transforms import add_teleportation
        # M = add_teleportation(P, alpha=0.85)

        pm = PowerMethod(
            matrix=P_row.T,
            alpha=0.85,
            tol=10e-10,
            max_iter=10000,
            MODE='auto',
        )
        pm.run()

        history = pm.residuals

        print(
            f"{spec.name}: n={data.metadata['n']}, m={data.metadata['m']}, "
            f"iters={len(history)}, final_res={history[-1]:.2e}"
        )

        plt.semilogy(history, label=spec.name)

    plt.xlabel("Iteration")
    plt.ylabel("L1 residual")
    plt.legend()
    plt.title("PageRank power iteration convergence")
    plt.show()

if __name__ == "__main__":
    main()
