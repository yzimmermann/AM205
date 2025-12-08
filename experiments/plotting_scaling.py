# experiments/plotting_scaling.py
# Author: Yoel Zimmermann

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("thesis.mplstyle")

data = np.load("scaling_pagerank_walltime.npz")
ns_arr = data["ns"]
graph_mean = data["graph_mean"]
graph_ci = data["graph_ci"]
solve_mean = data["solve_mean"]
solve_ci = data["solve_ci"]
total_mean = data["total_mean"]
total_ci = data["total_ci"]
iters_mean = data["iters_mean"]
iters_ci = data["iters_ci"]

# wall time vs n
plt.figure(figsize=(6, 4))
plt.errorbar(ns_arr, total_mean, yerr=total_ci, fmt="o-", capsize=3, label="Total time")
plt.errorbar(ns_arr, solve_mean, yerr=solve_ci, fmt="s--", capsize=3, label="Power iteration")
plt.errorbar(ns_arr, graph_mean, yerr=graph_ci, fmt="d--", capsize=3, label="Graph + P construction")
plt.xlabel("Number of nodes $n$")
plt.ylabel("Wall time [s]")
plt.title("PageRank wall time vs. graph size")
plt.legend()
plt.tight_layout()
plt.show()

# log–log scale
plt.figure(figsize=(6, 4))
plt.errorbar(ns_arr, total_mean, yerr=total_ci, fmt="o-", capsize=3, label="Total time", )
plt.errorbar(ns_arr, solve_mean, yerr=solve_ci, fmt="s--", capsize=3, label="Power iteration")
plt.errorbar(ns_arr, graph_mean, yerr=graph_ci, fmt="d--", capsize=3, label="Graph + P construction")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of nodes $n$")
plt.ylabel("Wall time [s]")
plt.title("Scaling of PageRank with graph size")
plt.xlim(7e2, 1e5)
plt.legend()
plt.tight_layout()
plt.savefig("scaling_pagerank_walltime_loglog.pdf")

# iterations vs n
plt.figure(figsize=(6, 4))
plt.errorbar(ns_arr, iters_mean, yerr=iters_ci, fmt="o-", capsize=3)
plt.xlabel("Number of nodes $n$")
plt.ylabel("Iterations to convergence")
plt.title("Iterations vs. graph size")
plt.tight_layout()
plt.show()
