#!/usr/bin/env python3
"""
Follow-up to the GraphSAGE-style pilot: does keeping the NEIGHBOUR-aggregation weights local
(only self weights are averaged, in the spirit of FedBN's local batch-norm layers) recover the
accuracy a fully shared model loses under structural heterogeneity?

Compares on the same federations and seeds as pilot_v3_sage.py:
  * FedAvg, all weights shared
  * FedAvg, neighbour weights (W1_neigh, W2_neigh) kept local; self weights shared
Each client is evaluated with its own model (for the shared variant that is the global model).

Usage: python3 partial_local_neighbour_weights.py   (needs pilot_v3_sage.py in the same folder)
"""
import csv, importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('m', 'pilot_v3_sage.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
SHARED_IDX = [0, 2]          # self weights
LOCAL_IDX = [1, 3]           # neighbour weights


def client_acc(p, c):
    _, _, s = m.forward(p, c)
    return (s.argmax(1)[c["te"]] == c["y"][c["te"]]).mean()


def fedavg(p0, clients, local_idx):
    models = [[w.copy() for w in p0] for _ in clients]
    for _ in range(m.ROUNDS):
        for k, c in enumerate(clients):
            for _ in range(m.E_LOCAL):
                g = m.grad(models[k], c)
                models[k] = [w - m.LR * gw for w, gw in zip(models[k], g)]
        for i in range(m.NP):
            if i in local_idx:
                continue
            mean = np.mean([q[i] for q in models], axis=0)
            for q in models:
                q[i] = mean.copy()
    return np.array([client_acc(models[k], c) for k, c in enumerate(clients)])


rows = []
for spread in m.SPREADS:
    hs = np.linspace(m.H_MEAN - spread / 2, m.H_MEAN + spread / 2, m.K)
    for seed in m.SEEDS:
        rng = np.random.default_rng(seed)       # identical federations to pilot_v3_sage.py
        class_means = rng.standard_normal((m.C, m.D))
        clients = [m.make_client(rng, h, class_means) for h in hs]
        p0 = m.init_params(rng)
        shared = fedavg(p0, clients, [])
        partial = fedavg(p0, clients, LOCAL_IDX)
        row = dict(spread=spread, seed=seed, acc_fedavg_shared=shared.mean(),
                   acc_fedavg_local_neigh=partial.mean(), gain=partial.mean() - shared.mean(),
                   worst_client_shared=shared.min(), worst_client_local_neigh=partial.min())
        rows.append(row)
        print({k: round(float(v), 4) for k, v in row.items()}, flush=True)

with open('partial_local_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('\nspread | shared | local-neighbour | gain | worst shared | worst local-neighbour')
for s in m.SPREADS:
    R = [r for r in rows if r['spread'] == s]
    f = lambda k: f"{np.mean([r[k] for r in R]):.4f}±{np.std([r[k] for r in R]):.4f}"
    print(s, '|', ' | '.join(f(k) for k in ['acc_fedavg_shared', 'acc_fedavg_local_neigh', 'gain', 'worst_client_shared', 'worst_client_local_neigh']))
