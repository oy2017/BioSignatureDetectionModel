# Camera-ready expectations (ML4PS 2026, submission 10)

Written before each check is run. Outcome recorded beneath each.

## CR1. A non-tree family in the transfer grid (existing file, no new run)

`results/ariel_norm_mlp_shifts.csv` holds the normalised MLP on the same 48 cases.
Expectation: its Spearman rho with the two normalised tree pipelines is at least 0.90,
and its mean rho with the four PCA and raw pipelines is lower than that and within
the "sharing neither" range of the paper (0.76-0.93). If instead the MLP agrees with
the PCA/raw pipelines as well as with its own representation, the paper's claim that
the representation carries the ranking is weakened and the camera-ready must say so.

Outcome (run 5 November; the first pass keyed cases by name alone and so merged the
same-named SNR levels of different axes to 40, the rerun in `transfer_mlp.py` keys by
axis and case and uses all 48): met. The MLP (clean 90.09 %) agrees with the normalised trees at rho 0.973
(XGBoost) and 0.978 (random forest), above every tree-tree pair in the paper; with the
PCA pipelines at 0.932 and 0.907 and with the raw pipelines at 0.842 and 0.784, all
within the paper's "sharing neither" range. Magnitudes also track the representation:
under the 3e7 haze the MLP loses 34.8 points against the XGBoost's 28.8 and the PCA
pipeline's 6.1. The full 3 x 3 grid (MLP on PCA and raw bins across the 48 cases,
through `transfer.py`) is expectation CR2, to be written before it is run.

With all 48 cases and all 21 pairs (`results/ariel_transfer_mlp.txt`): same representation
mean rho 0.957 (n = 5, range 0.905-0.978), same model family 0.846 (n = 6), sharing
neither 0.854 (n = 10); one-sided Mann-Whitney p = 0.0006 against all other pairs and
0.004 against the family-sharing pairs; family against neither p = 0.72. The MLP agrees
with the normalised trees at 0.973 and 0.978 and with the PCA and raw pipelines at
0.78-0.93. Magnitude spread over the 28 cases costing some pipeline at least 5 points:
median 10.3x with seven pipelines (8.1x with six).

## CR2. The full 3 x 3 grid (MLP on PCA and raw bins)

Not run. The spectra in `data/` and the simulator stack (MultiREx with the Mie fork,
ExoMolOP, Exo-Transmit) are no longer on the machine; regenerating them is about half
a day with an uncertain install, for a grid whose decisive cell (a non-tree family on
one representation) CR1 already fills. The camera-ready says the grid is 2 x 3 plus one
network.
