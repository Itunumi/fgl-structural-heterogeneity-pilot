#!/usr/bin/env python3
"""Summary figure for the outline, built only from the CSVs the pilot scripts write."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(f):
    return list(csv.DictReader(open(f)))

def stat(rows, key, spreads):
    mu, sd = [], []
    for s in spreads:
        v = [float(r[key]) for r in rows if float(r["spread"]) == s]
        mu.append(np.mean(v)); sd.append(np.std(v))
    return np.array(mu) * 100, np.array(sd) * 100

S = [0.0, 0.4, 0.8]
gcn, sage = load("pilot_results_E20.csv"), load("pilot_results_sage.csv")
cg, cs = load("control_gcn.csv"), load("control_sage.csv")
pl = load("partial_local_results.csv")

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2b6cb0", "#c05621", "#718096"
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25))

for rows, lab, col, mk in [(gcn, "GCN", BLUE, "o"), (sage, "GraphSAGE-style", ORANGE, "s")]:
    mu, sd = stat(rows, "gap", S)
    ax[0].errorbar(S, mu, yerr=sd, marker=mk, color=col, capsize=2, lw=1.4, ms=4, label=lab)
ax[0].axhline(0, color=GREY, lw=0.8, ls=":")
ax[0].set_title("(a) Centralized − FedAvg", fontsize=9)
ax[0].set_ylabel("Accuracy points")
ax[0].set_ylim(-8, 8)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")

for rows, lab, col, mk in [(cg, "GCN", BLUE, "o"), (cs, "GraphSAGE-style", ORANGE, "s")]:
    mu, sd = stat(rows, "cost_of_heterogeneity", S)
    ax[1].errorbar(S, mu, yerr=sd, marker=mk, color=col, capsize=2, lw=1.4, ms=4, label=lab)
ax[1].axhline(0, color=GREY, lw=0.8, ls=":")
ax[1].set_title("(b) Cost of one shared model", fontsize=9)
ax[1].set_ylim(-2, 9)

mu, sd = stat(pl, "gain", S)
ax[2].errorbar(S, mu, yerr=sd, marker="s", color=ORANGE, capsize=2, lw=1.4, ms=4)
ax[2].axhline(0, color=GREY, lw=0.8, ls=":")
ax[2].set_title("(c) Gain: local neighbor weights", fontsize=9)
ax[2].set_ylim(-3, 6)

for a in ax:
    a.set_xticks(S)
    a.set_xlabel("Homophily spread across clients")
fig.tight_layout(w_pad=1.2)
fig.savefig("pilot_summary.png", dpi=300)
print("saved pilot_summary.png")
