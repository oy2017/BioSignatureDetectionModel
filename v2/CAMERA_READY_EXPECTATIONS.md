# Camera-ready expectations (ML4PS 2026, submission 10)

Written before each check is run. Outcome recorded beneath each.

## CR1. A non-tree family in the transfer grid (existing file, no new run)

`results/ariel_norm_mlp_shifts.csv` holds the normalised MLP on the same 48 cases.
Expectation: its Spearman rho with the two normalised tree pipelines is at least 0.90,
and its mean rho with the four PCA and raw pipelines is lower than that and within
the "sharing neither" range of the paper (0.76-0.93). If instead the MLP agrees with
the PCA/raw pipelines as well as with its own representation, the paper's claim that
the representation carries the ranking is weakened and the camera-ready must say so.
