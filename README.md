# Pilot: structural heterogeneity in federated GNNs

Fully synthetic and numpy-only, so it runs on any laptop with no GPU and no downloads.
Requirements: `numpy`, `scipy`, `matplotlib`.

## Question
Suppose label and feature distributions are identical across clients and only the graph
structure differs (edge homophily). Does FedAvg then (a) drift away from centralised training,
(b) produce a shared model that is worse than models trained on structurally matched clients, and (c) does keeping the neighbour weights local help?

## Setup
- 8 clients, 300 nodes each, 4 balanced classes, 16-d features drawn from the same distribution
  on every client.
- Stochastic block model graphs with expected degree 10. Client homophily values are evenly
  spaced around a mean of 0.5, with spread 0, 0.4 or 0.8 (0.8 means 0.1 to 0.9).
- Full-batch training, plain SGD, 5 seeds.
- `pilot_v2_E20.py`: 2-layer GCN, FedAvg with E = 20 local steps, 80 rounds.
- `pilot_v3_sage.py`: 2-layer GraphSAGE-style model (separate self and mean-neighbour weights),
  E = 10, 100 rounds.
- `control_matched_homogeneous.py <module> <out.csv>`: trains a centralised model on 8 graphs that
  all share client k's homophily, then compares the shared model on the mixed federation with
  these matched baselines.

## Run (about 10–25 minutes each on one CPU core)
```
python3 pilot_v2_E20.py
python3 pilot_v3_sage.py
python3 control_matched_homogeneous.py pilot_v2_E20.py control_gcn.csv
python3 control_matched_homogeneous.py pilot_v3_sage.py control_sage.csv
python3 partial_local_neighbour_weights.py
```

## Results (5 seeds, mean ± sd)
| Spread | GCN: centralised − FedAvg | SAGE: centralised − FedAvg | GCN: cost of sharing | SAGE: cost of sharing | SAGE: gain from local neighbour weights |
|---|---|---|---|---|---|
| 0.0 | +0.7 ± 0.4 | -1.4 ± 0.2 | -0.4 ± 0.8 | +0.1 ± 0.7 | -1.4 ± 0.4 |
| 0.4 | +0.6 ± 0.7 | -0.5 ± 0.8 | +0.0 ± 0.4 | +2.6 ± 1.2 | +0.9 ± 1.7 |
| 0.8 | +0.1 ± 0.3 | +0.2 ± 0.3 | +0.4 ± 0.1 | +6.3 ± 1.4 | +4.0 ± 0.5 |

Values are accuracy points. *Cost of sharing* = matched-homogeneous accuracy minus shared-model accuracy on the mixed federation.

## Limits
- Synthetic SBM graphs, small, with full-batch training: no neighbour sampling, no
  cross-client edges, no partial participation.
- Accuracy saturates within a few rounds, so the "rounds to target" column says little about
  convergence speed.
- In this regime, homophily spread alone produced no FedAvg-specific drift. Its cost showed up
  at the shared optimum, and only for the model that can exploit heterophily.

## Figure
`python3 make_figure.py` rebuilds `pilot_summary.png` (Fig. 1 in the outline) from the CSVs.

## Supporting figures
- `pilot_figure_E20.png` (GCN) and `pilot_figure_sage.png` (GraphSAGE-style): accuracy of
  centralised vs FedAvg training, gradient dissimilarity (zeta^2), and the FedAvg gap against a
  spectral distance between client graphs. They show that zeta^2 did not grow with structural
  heterogeneity, and that the spectral distance barely changed (0.005 to 0.007) while the
  class-compatibility divergence rose from 0.03 to 0.34.

## Author
C. O. Lawal, Covenant University. ORCID 0000-0002-3262-020X.
Pilot study for the research outline "Convergent and Robust Federated Graph Learning under
Client Heterogeneity" (Vector Distinguished Postdoctoral Fellowship application, 2027).
