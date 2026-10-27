#!/usr/bin/env python3
"""
Pilot: does structural heterogeneity ALONE slow / degrade FedAvg for GNNs?

Setup (fully synthetic, numpy only, no GPU, no downloads):
  * K clients, each holding its own graph drawn from a stochastic block model.
  * Label distribution is IDENTICAL across clients (balanced classes) -> no label skew.
  * Node-feature distribution is IDENTICAL across clients              -> no feature skew.
  * Only the graph STRUCTURE differs: client k has edge homophily h_k.
    Mean homophily is held fixed at H_MEAN; we vary the SPREAD of h_k across clients.
  * Same average degree for every client.
Model: 2-layer GraphSAGE-style GNN (separate self / mean-neighbour weights), full-batch, plain SGD, manual backprop.
Compared:
  * Centralised GD on the disjoint union of all client graphs (no federation).
  * FedAvg with E local steps per round (E=1 is mathematically identical to
    centralised GD for equal-sized clients, used as a sanity check).
Reported per spread:
  * spectral divergence between client graphs (mean pairwise 1-Wasserstein distance
    between eigenvalue distributions of the normalised adjacency)  -> a measurable
    structural-divergence quantity
  * gradient dissimilarity zeta^2 = mean_k ||grad F_k(w) - grad F(w)||^2 at the
    global model, averaged over rounds (the term that enters FedAvg bounds)
  * test accuracy of centralised vs FedAvg, and rounds to reach a target accuracy

Run:  python3 pilot_v2_E20.py   (GCN; the GraphSAGE-style variant is pilot_v3_sage.py)
Outputs: a results CSV and figure named after the variant in the current directory.
Requires: numpy, scipy, matplotlib.
"""
import csv
import numpy as np
from scipy.stats import wasserstein_distance

# ---------------- configuration ----------------
K = 8              # clients
N = 300            # nodes per client
C = 4              # classes
D = 16             # feature dimension
AVG_DEG = 10.0     # expected degree, same for every client
H_MEAN = 0.5       # mean edge homophily across clients (held fixed)
SPREADS = [0.0, 0.4, 0.8]   # max(h_k) - min(h_k)
FEAT_NOISE = 3.0   # features weakly informative alone, so structure matters
HIDDEN = 32
LR = 0.1
E_LOCAL = 10       # local steps per round for FedAvg
ROUNDS = 100
SEEDS = [0, 1, 2, 3, 4]
TRAIN_FRAC, TEST_FRAC = 0.2, 0.5
TARGET_FRAC = 0.95  # "rounds to target" = first round reaching 95% of centralised final acc


def make_client(rng, h, class_means):
    y = np.repeat(np.arange(C), N // C)
    rng.shuffle(y)
    x = class_means[y] + FEAT_NOISE * rng.standard_normal((N, D))
    # SBM probabilities giving expected degree AVG_DEG and edge homophily h
    n_c = N / C
    p_in = h * AVG_DEG / (n_c - 1)
    p_out = (1 - h) * AVG_DEG / (N - n_c)
    same = (y[:, None] == y[None, :])
    P = np.where(same, p_in, p_out)
    U = rng.random((N, N))
    A = np.triu((U < P).astype(float), 1)
    A = A + A.T
    A_hat = A + np.eye(N)
    dinv = 1.0 / np.sqrt(A_hat.sum(1))
    A_norm = dinv[:, None] * A_hat * dinv[None, :]   # used only for the spectral measure
    deg = np.maximum(A.sum(1), 1)
    A_mean = A / deg[:, None]                         # mean aggregator (GraphSAGE-style)
    idx = rng.permutation(N)
    ntr, nte = int(TRAIN_FRAC * N), int(TEST_FRAC * N)
    tr, te = np.zeros(N, bool), np.zeros(N, bool)
    tr[idx[:ntr]] = True
    te[idx[ntr:ntr + nte]] = True
    emp_h = (A * same).sum() / max(A.sum(), 1)
    Yo = np.eye(C)[y]
    M = Yo.T @ A @ Yo
    compat = M / M.sum(1, keepdims=True)
    return dict(A=A_mean, X=x, AX=A_mean @ x, y=y, Y=np.eye(C)[y], tr=tr, te=te,
                eig=np.linalg.eigvalsh(A_norm), emp_h=emp_h, compat=compat)


NP = 4
def init_params(rng):
    return [rng.standard_normal((D, HIDDEN)) * np.sqrt(1.0 / D),      # W1_self
            rng.standard_normal((D, HIDDEN)) * np.sqrt(1.0 / D),      # W1_neigh
            rng.standard_normal((HIDDEN, C)) * np.sqrt(0.5 / HIDDEN), # W2_self
            rng.standard_normal((HIDDEN, C)) * np.sqrt(0.5 / HIDDEN)] # W2_neigh


def forward(p, c):
    pre1 = c["X"] @ p[0] + c["AX"] @ p[1]
    h1 = np.maximum(pre1, 0)
    ah1 = c["A"] @ h1
    z = h1 @ p[2] + ah1 @ p[3]
    z = z - z.max(1, keepdims=True)
    s = np.exp(z)
    s /= s.sum(1, keepdims=True)
    return pre1, (h1, ah1), s


def grad(p, c):
    pre1, (h1, ah1), s = forward(p, c)
    m = c["tr"]
    dz = np.zeros_like(s)
    dz[m] = (s[m] - c["Y"][m]) / m.sum()
    dW2s = h1.T @ dz
    dW2n = ah1.T @ dz
    dh1 = dz @ p[2].T + c["A"].T @ (dz @ p[3].T)
    dpre1 = dh1 * (pre1 > 0)
    return [c["X"].T @ dpre1, c["AX"].T @ dpre1, dW2s, dW2n]


def acc(p, clients):
    correct = total = 0
    for c in clients:
        _, _, s = forward(p, c)
        pred = s.argmax(1)
        correct += (pred[c["te"]] == c["y"][c["te"]]).sum()
        total += c["te"].sum()
    return correct / total


def avg(list_of_params):
    return [np.mean([q[i] for q in list_of_params], axis=0) for i in range(NP)]


def run_central(p0, clients):
    p = [w.copy() for w in p0]
    curve = []
    for _ in range(ROUNDS):
        for _ in range(E_LOCAL):          # same number of gradient steps as FedAvg
            g = avg([grad(p, c) for c in clients])
            p = [w - LR * gw for w, gw in zip(p, g)]
        curve.append(acc(p, clients))
    return np.array(curve)


def run_fedavg(p0, clients, E):
    p = [w.copy() for w in p0]
    curve, zetas = [], []
    for _ in range(ROUNDS):
        gs = [grad(p, c) for c in clients]
        gbar = avg(gs)
        zetas.append(np.mean([sum(((gk[i] - gbar[i]) ** 2).sum() for i in range(NP)) for gk in gs]))
        locals_ = []
        for c in clients:
            q = [w.copy() for w in p]
            for _ in range(E):
                g = grad(q, c)
                q = [w - LR * gw for w, gw in zip(q, g)]
            locals_.append(q)
        p = avg(locals_)
        curve.append(acc(p, clients))
    return np.array(curve), float(np.mean(zetas))


def rounds_to(curve, target):
    hit = np.where(curve >= target)[0]
    return int(hit[0]) + 1 if len(hit) else np.nan


def main():
    rows = []
    for spread in SPREADS:
        hs = np.linspace(H_MEAN - spread / 2, H_MEAN + spread / 2, K)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            class_means = rng.standard_normal((C, D))   # shared across clients
            clients = [make_client(rng, h, class_means) for h in hs]
            spec_div = np.mean([wasserstein_distance(clients[i]["eig"], clients[j]["eig"])
                                for i in range(K) for j in range(i + 1, K)])
            compat_div = np.mean([0.5 * np.abs(clients[i]['compat'] - clients[j]['compat']).sum(1).mean()
                                  for i in range(K) for j in range(i + 1, K)])
            p0 = init_params(rng)
            cen = run_central(p0, clients)
            fed, zeta = run_fedavg(p0, clients, E_LOCAL)
            target = TARGET_FRAC * cen[-1]
            row = dict(spread=spread, seed=seed,
                       emp_h_min=min(c["emp_h"] for c in clients),
                       emp_h_max=max(c["emp_h"] for c in clients),
                       spectral_div=spec_div, compat_div=compat_div, zeta2=zeta,
                       acc_central=cen[-1], acc_fedavg=fed[-1],
                       gap=cen[-1] - fed[-1],
                       rounds_central=rounds_to(cen, target),
                       rounds_fedavg=rounds_to(fed, target))
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    keys = ["spread", "seed", "emp_h_min", "emp_h_max", "spectral_div", "compat_div", "zeta2",
            "acc_central", "acc_fedavg", "gap", "rounds_central", "rounds_fedavg"]
    with open("pilot_results_sage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary
    print("\nspread | " + " | ".join(keys[4:]))
    summ = []
    for s in SPREADS:
        R = [r for r in rows if r["spread"] == s]
        f = lambda k: (np.nanmean([r[k] for r in R]), np.nanstd([r[k] for r in R]))
        line = {k: f(k) for k in keys[4:]}
        summ.append((s, line))
        print(f"{s:.1f} | " + " | ".join(f"{line[k][0]:.4f}±{line[k][1]:.4f}" for k in keys[4:]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    xs = [s for s, _ in summ]
    for k, lab in [("acc_central", "Centralised"), ("acc_fedavg", f"FedAvg (E={E_LOCAL})")]:
        ax[0].errorbar(xs, [l[k][0] for _, l in summ], yerr=[l[k][1] for _, l in summ], marker="o", capsize=3, label=lab)
    ax[0].set_xlabel("Homophily spread across clients"); ax[0].set_ylabel("Test accuracy"); ax[0].legend()
    ax[1].errorbar(xs, [l["zeta2"][0] for _, l in summ], yerr=[l["zeta2"][1] for _, l in summ], marker="o", capsize=3)
    ax[1].set_xlabel("Homophily spread across clients"); ax[1].set_ylabel(r"Gradient dissimilarity $\zeta^2$")
    ax[2].scatter([r["spectral_div"] for r in rows], [r["gap"] for r in rows])
    ax[2].set_xlabel("Spectral divergence between client graphs"); ax[2].set_ylabel("Centralised − FedAvg accuracy")
    fig.suptitle("Label and feature distributions identical across clients; only graph structure differs", fontsize=10)
    fig.tight_layout()
    fig.savefig("pilot_figure_sage.png", dpi=200)


if __name__ == "__main__":
    main()
