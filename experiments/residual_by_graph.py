# experiments/residual_by_graph_model.py
# Author: Yoel Zimmermann

from graph_generation import GraphSpec, generate_graph
from pagerank import PowerMethod
import numpy as np

import matplotlib.pyplot as plt

plt.style.use("thesis.mplstyle")

color_dict = {
    "Erdős-Rényi": "orange",
    "Barabási–Albert": "green",
    "Scale-Free": "red",
    "Barbell": "blue",
}

eigvals = {}

def experiment_residuals_by_graph_model():
    specs = [
        GraphSpec(
            model="erdos_renyi",
            n=2000,
            params={"p": 0.03},
            directed=True,
            seed=0,
            name="Erdős-Rényi",
        ),
        GraphSpec(
            model="barabasi_albert",
            n=2000,
            params={"m": 3},
            seed=0,
            name="Barabási–Albert",
        ),
        GraphSpec(
            model="directed_scale_free",
            n=2000,
            params={},  # use default values from OG paper
            directed=True,
            seed=0,
            name="Scale-Free",
        ),
        GraphSpec(
            model="barbell",
            n=2000,
            params={},
            directed=True,
            seed=0,
            name="Barbell",
        ),
    ]

    alpha = 0.85
    tol = 1e-10
    max_iter = 10000

    plt.figure(figsize=(6, 4))

    for spec in specs:
        print(f"\n=== Graph: {spec.name} (model={spec.model}) ===")
        data = generate_graph(spec)
        P = data.P

        pm = PowerMethod(
            matrix=P.T,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            MODE="auto",
        )

        pm.iterate()
        history = pm.residuals

        print(
            f"  n={data.metadata['n']}, m={data.metadata['m']}, "
            f"iters={len(history)}, final_res={history[-1]:.2e}"
        )

        lam2 = pm.estimate_second_eigenvalue()

        rho = lam2

        plt.semilogy(history, label=spec.name + f" ($\\lambda={rho:.4f}$)", color=color_dict[spec.name])


        k = np.arange(len(history))
        r0 = history[0]
        theory = r0 * (rho ** k)

        plt.semilogy(
            theory,
            "--",
            linewidth=1.2,
            #label=f"{spec.name} (theory: $|\\lambda_2|=${rho:.4f})",
            color=color_dict[spec.name],
        )

        # compute whole spectrum
        eigenvalues = pm.full_eigenvalues()
        eigvals[spec.name] = eigenvalues


    plt.xlabel("Iteration")
    plt.ylabel(r"$\|x_{k+1} - x_k\|_1$")
    plt.title("Power iteration residuals for different graph models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("residuals_by_graph_model.pdf")
    plt.show()

    # Scatter eigenvalues
    plt.figure(figsize=(6, 6))
    for name, eigenvalues in eigvals.items():
        plt.scatter(
            eigenvalues.real,
            eigenvalues.imag,
            s=10,
            alpha=0.6,
            label=name,
            color=color_dict[name],
        )
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.title("Eigenvalue spectra of different graph models")
        plt.axhline(0, color="black", linewidth=0.4, linestyle
    ="--")
        plt.axvline(0, color="black", linewidth=0.4, linestyle="--")
        plt.axis("equal")
        #plt.legend()
        plt.tight_layout()
        plt.savefig(f"eigenvalue_spectra_{name}.pdf")
        plt.close()

        # Plot sorted absolute eigenvalue magnitudes
    plt.figure(figsize=(6, 4))
    for name, eigenvalues in eigvals.items():
        abs_eigenvalues = np.abs(eigenvalues)
        sorted_abs_eigenvalues = np.sort(abs_eigenvalues)[::-1]
        plt.plot(
            sorted_abs_eigenvalues,
            marker="o",
            linestyle="-",
            markersize=3,
            label=name,
            color=color_dict[name],
        )
    plt.xlabel("Index")
    plt.ylabel(f"$|\\lambda|$")
    plt.title("Sorted absolute eigenvalue magnitudes")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sorted_absolute_eigenvalues.pdf")
    plt.show()


def main():
    experiment_residuals_by_graph_model()


if __name__ == "__main__":
    main()