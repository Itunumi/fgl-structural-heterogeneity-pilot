#!/usr/bin/env python3
"""
Control for the pilot: is the accuracy drop under structural heterogeneity caused by
SHARING one model across differently-structured graphs, or simply by some clients
being heterophilous?

For every homophily value h used in the heterogeneous federations, train a centralised
model on K graphs that ALL have homophily h ("matched homogeneous" setting), and record
its test accuracy. The matched baseline for a heterogeneous federation is the mean of
those accuracies over its clients' h values. The cost of structural heterogeneity for a
shared model is (matched baseline) - (accuracy of the shared model on the mixed federation).

Usage: python3 control_matched_homogeneous.py <pilot_module.py> <out.csv>
       (pilot module = pilot_v2_E20.py for GCN, pilot_v3_sage.py for GraphSAGE-style)
"""
import sys, csv, importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('m', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
OUT = sys.argv[2]
STEPS = 300


def train_central(p0, clients):
    p = [w.copy() for w in p0]
    for _ in range(STEPS):
        g = m.avg([m.grad(p, c) for c in clients])
        p = [w - m.LR * gw for w, gw in zip(p, g)]
    return p


def per_client_acc(p, clients):
    out = []
    for c in clients:
        _, _, s = m.forward(p, c)
        out.append((s.argmax(1)[c["te"]] == c["y"][c["te"]]).mean())
    return np.array(out)


rows = []
for seed in m.SEEDS:
    for spread in m.SPREADS:
        hs = np.linspace(m.H_MEAN - spread / 2, m.H_MEAN + spread / 2, m.K)
        rng = np.random.default_rng(seed)
        class_means = rng.standard_normal((m.C, m.D))
        mixed = [m.make_client(rng, h, class_means) for h in hs]
        p0 = m.init_params(rng)
        shared = per_client_acc(train_central(p0, mixed), mixed)
        matched = []
        for k, h in enumerate(hs):
            r2 = np.random.default_rng(10_000 + 97 * seed + k)
            homo = [m.make_client(r2, h, class_means) for _ in range(m.K)]
            matched.append(per_client_acc(train_central(p0, homo), homo).mean())
        matched = np.array(matched)
        row = dict(seed=seed, spread=spread, acc_shared_mixed=shared.mean(),
                   acc_matched_homogeneous=matched.mean(),
                   cost_of_heterogeneity=matched.mean() - shared.mean(),
                   worst_client_cost=float(np.max(matched - shared)))
        rows.append(row)
        print({k: round(float(v), 4) for k, v in row.items()}, flush=True)

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

print('\nspread | shared(mixed) | matched homogeneous | cost | worst-client cost')
for s in m.SPREADS:
    R = [r for r in rows if r['spread'] == s]
    f = lambda k: f"{np.mean([r[k] for r in R]):.4f}±{np.std([r[k] for r in R]):.4f}"
    print(s, '|', f('acc_shared_mixed'), '|', f('acc_matched_homogeneous'), '|', f('cost_of_heterogeneity'), '|', f('worst_client_cost'))
