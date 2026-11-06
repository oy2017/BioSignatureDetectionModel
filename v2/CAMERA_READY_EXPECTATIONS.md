# Camera-ready expectations (ML4PS 2026, submission 10)

Written before each check is run. Outcome recorded beneath each.

## CR1. A non-tree family in the transfer grid (existing file, no new run)

`results/ariel_norm_mlp_shifts.csv` holds the normalised MLP on the same 48 cases.
Expectation: its Spearman rho with the two normalised tree pipelines is at least 0.90,
and its mean rho with the four PCA and raw pipelines is lower than that and within
the "sharing neither" range of the paper (0.76-0.93). If instead the MLP agrees with
the PCA/raw pipelines as well as with its own representation, the paper's claim that
the representation carries the ranking is weakened and the camera-ready must say so.

Outcome (run 5 November, on the 40 re-rendered cases present in every per-pipeline
CSV; the seven injected noise and offset cases that `transfer.py` adds were not in
this pass): met. The MLP (clean 90.09 %) agrees with the normalised trees at rho 0.973
(XGBoost) and 0.978 (random forest), above every tree-tree pair in the paper; with the
PCA pipelines at 0.932 and 0.907 and with the raw pipelines at 0.842 and 0.784, all
within the paper's "sharing neither" range. Magnitudes also track the representation:
under the 3e7 haze the MLP loses 34.8 points against the XGBoost's 28.8 and the PCA
pipeline's 6.1. The full 3 x 3 grid (MLP on PCA and raw bins across the 48 cases,
through `transfer.py`) is expectation CR2, to be written before it is run.
