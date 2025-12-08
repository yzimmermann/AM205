# experiments/scaling_pagerank.py
# Author: Yoel Zimmermann

from graph_generation import GraphSpec, generate_graph
from pagerank import PowerMethod

import numpy as np
import time


def experiment_scaling_walltime():
    ns = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]

    alpha = 0.85
    tol = 1e-10
    max_iter = 10000

    repeats = 1

    graph_times = np.zeros((len(ns), repeats))
    solve_times = np.zeros((len(ns), repeats))
    total_times = np.zeros((len(ns), repeats))
    iters_mat = np.zeros((len(ns), repeats), dtype=int)

    for i, n in enumerate(ns):
        print(f"\n=== n = {n} ===")

        for r in range(repeats):
            print(f"  Run {r + 1}/{repeats}")

            t0 = time.perf_counter()
            spec = GraphSpec(
                model="barbell",
                n=n,
                params={},
                directed=True,
                seed=r+420,      # vary seed
                name=f"Barbell_n{n}",
            )
            data = generate_graph(spec)
            P = data.P
            t1 = time.perf_counter()

            pm = PowerMethod(
                matrix=P.T,
                alpha=alpha,
                tol=tol,
                max_iter=max_iter,
                MODE="auto",
            )

            t2 = time.perf_counter()
            pm.iterate()
            t3 = time.perf_counter()

            graph_dt = t1 - t0
            solve_dt = t3 - t2
            total_dt = t3 - t0

            graph_times[i, r] = graph_dt
            solve_times[i, r] = solve_dt
            total_times[i, r] = total_dt
            iters_mat[i, r] = len(pm.residuals)

            print(
                f"graph: {graph_dt:.3f}s, "
                f"solve: {solve_dt:.3f}s, "
                f"total: {total_dt:.3f}s, "
                f"iters: {len(pm.residuals)}"
            )

    ns_arr = np.array(ns, dtype=float)

    def mean_and_ci(x):
        """Return mean and 95% confidence interval assuming standard normal."""
        mean = x.mean(axis=1)
        se = x.std(axis=1, ddof=1) / np.sqrt(x.shape[1])
        ci = 1.96 * se
        return mean, ci

    graph_mean, graph_ci = mean_and_ci(graph_times)
    solve_mean, solve_ci = mean_and_ci(solve_times)
    total_mean, total_ci = mean_and_ci(total_times)
    iters_mean, iters_ci = mean_and_ci(iters_mat.astype(float))

    # save all data for future plots
    np.savez(
        "scaling_pagerank_walltimez_ER.npz",
        ns=ns_arr,
        graph_mean=graph_mean,
        graph_ci=graph_ci,
        solve_mean=solve_mean,
        solve_ci=solve_ci,
        total_mean=total_mean,
        total_ci=total_ci,
        iters_mean=iters_mean,
        iters_ci=iters_ci,
    )

def main():
    experiment_scaling_walltime()


if __name__ == "__main__":
    main()