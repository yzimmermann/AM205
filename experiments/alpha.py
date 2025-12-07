# experiments/run_pagerank_experiments.py
# Author: Yoel Zimmermann

from graph_generation import GraphSpec, generate_graph
from pagerank import PowerMethod

import numpy as np
import matplotlib.pyplot as plt

def experiment_alpha_convergence():
    specs = [
        GraphSpec(
            model="erdos_renyi",
            n=2000,
            params={"p": 0.03},
            directed=True,
            seed=0,
            name="ER",
        ),
        GraphSpec(
            model="barabasi_albert",
            n=2000,
            params={"m": 3},
            directed=True,
            seed=0,
            name="BA",
        ),
        GraphSpec(
            model="directed_scale_free",
            n=2000,
            params={},  # use default values from OG paper
            directed=True,
            seed=0,
            name="ScaleFree",
        ),
        GraphSpec(
            model="barbell",
            n=2000,
            params={},
            name="Barbell",
        )
    ]

    specs = [
        GraphSpec(
            model="directed_scale_free",
            n=20000,
            params={},  # use default values from OG paper
            directed=True,
            seed=0,
            name="ScaleFree",
        ),
    ]

    alphas = [0.00, 0.05, 0.2, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.99]

    tol = 1e-10
    max_iter = 10_000

    example_residuals = None
    example_label = None

    results = {}

    for spec in specs:
        print(f"\n=== Graph: {spec.name} (model={spec.model}) ===")
        data = generate_graph(spec)
        P = data.P

        alpha_vals = []
        iter_counts = []
        lambda2_vals = []
        gaps = []

        for alpha in alphas:
            print(f"  Alpha = {alpha:.2f}")

            pm = PowerMethod(
                matrix = P.T,
                alpha = alpha,
                tol = tol,
                max_iter = max_iter,
                MODE = 'auto'
            )

            pm.iterate()
            history = pm.residuals
            iters = len(history)
            final_res = history[-1]

            #lam2 = pm.estimate_second_eigenvalue()
            lam2 = 0
            gap = 1.0 - lam2 if not np.isnan(lam2) else np.nan

            print(
                f"    iters={iters}, final_res={final_res:.2e}, "
                f"|lambda_2|={lam2:.4f}, gap={gap:.4f}"
            )

            alpha_vals.append(alpha)
            iter_counts.append(iters)
            lambda2_vals.append(lam2)
            gaps.append(gap)

            # Optional example
            if example_residuals is None and np.isclose(alpha, 0.90):
                example_residuals = history
                example_label = f"{spec.name}, alpha={alpha:.2f}"

        results[spec.name] = {
            "alphas": np.array(alpha_vals),
            "iters": np.array(iter_counts),
            "lambda2": np.array(lambda2_vals),
            "gap": np.array(gaps),
        }

    plt.figure(figsize=(6, 4))
    for name, res in results.items():
        plt.plot(res["alphas"], res["iters"], marker="o", label=name)
    plt.xlabel(r"$\alpha$ (teleportation parameter)")
    plt.ylabel("Iterations to convergence")
    plt.title("Impact of $\\alpha$ on PageRank convergence")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    for name, res in results.items():
        plt.plot(res["alphas"], res["lambda2"], marker="o", label=name)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$|\lambda_2(M_\alpha)|$")
    plt.title(r"Second eigenvalue magnitude vs $\alpha$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    for name, res in results.items():
        plt.plot(res["alphas"], res["gap"], marker="o", label=name)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"Spectral gap $1 - |\lambda_2|$")
    plt.title(r"Spectral gap vs $\alpha$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if example_residuals is not None:
        plt.figure(figsize=(6, 4))
        plt.semilogy(example_residuals)
        plt.xlabel("Iteration")
        plt.ylabel(r"$\|x_{k+1} - x_k\|_1$")
        plt.title(f"Power iteration residual ({example_label})")
        plt.tight_layout()
        plt.show()


def main():
    experiment_alpha_convergence()


if __name__ == "__main__":
    main()
