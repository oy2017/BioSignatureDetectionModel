"""Camera-ready check CR1: the transfer grid with a non-tree family added.

The submitted paper's grid is two tree families (XGBoost, random forest) crossed with
three representations. Reviewers asked whether "sharing a model family buys nothing"
survives a family that is not a tree ensemble. The normalised-representation MLP was
scored on the same 48 shift cases (results/ariel_norm_mlp_shifts.csv); this script adds
it as a seventh pipeline, recomputes every pairwise Spearman correlation, the group
means and the Mann-Whitney tests of transfer.py, and writes the magnitude table used in
the camera-ready version.

Usage: python transfer_mlp.py --config ariel
Outputs: results/{cfg}_transfer_mlp.txt, {cfg}_transfer_matrix.csv, {cfg}_magnitudes.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from common import RESULTS

KEYS = ["norm_xgb", "norm_rf", "norm_mlp", "pca_xgb", "pca_rf", "raw_xgb", "raw_rf"]
TABLE_CASES = [("exotransmit", "exotransmit", "Independent RT code (Exo-Transmit)"),
               ("exomol", "exomol", "Alternative opacities, 3 non-label gases"),
               ("exomol", "exomol_o3", "and the label-bearing (O3) line list too"),
               ("haze", "haze_3p0e7", "Haze, 3e7 m^-3"),
               ("cloud", "cloud_1e3Pa", "Cloud deck, 1e3 Pa"),
               ("tlse", "tlse_spots20", "Stellar spots, 20% coverage"),
               ("white noise", "snr5", "White noise, SNR 5"),
               ("correlated noise", "snr5", "Correlated noise, SNR 5")]


def load(cfg, key):
    f = f"{cfg}_shifts.csv" if key == "norm_xgb" else f"{cfg}_{key}_shifts.csv"
    return pd.read_csv(os.path.join(RESULTS, f))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel")
    cfg = ap.parse_args().config
    D = {k: load(cfg, k) for k in KEYS}
    L = {}
    for k, d in D.items():
        s = d[(d.axis != "clean") & (d.axis != "extrapolation")]
        L[k] = s.set_index(s.axis + "|" + s.case)["d_acc"]
    cases = sorted(set.intersection(*[set(v.index) for v in L.values()]))
    out = [f"Transfer grid with the normalised MLP added, configuration {cfg}",
           f"{len(cases)} shift cases shared by all {len(KEYS)} pipelines", ""]

    M = pd.DataFrame(index=KEYS, columns=KEYS, dtype=float)
    for a, b in itertools.product(KEYS, KEYS):
        M.loc[a, b] = spearmanr(L[a][cases], L[b][cases]).correlation
    M.to_csv(os.path.join(RESULTS, f"{cfg}_transfer_matrix.csv"))
    out.append("Spearman correlation of the per-case accuracy loss, all pairs:")
    out.append(M.round(3).to_string()); out.append("")

    rep = lambda k: k.split("_")[0]
    fam = lambda k: k.split("_")[1]
    groups = {"same representation": [], "same model family": [], "sharing neither": []}
    for a, b in itertools.combinations(KEYS, 2):
        r = M.loc[a, b]
        g = ("same representation" if rep(a) == rep(b) else
             "same model family" if fam(a) == fam(b) else "sharing neither")
        groups[g].append((a, b, r))
    for g, v in groups.items():
        rs = [x[2] for x in v]
        out.append(f"{g:22s} n={len(v):2d}  mean rho {np.mean(rs):.3f}  range {min(rs):.3f}-{max(rs):.3f}")
        for a, b, r in sorted(v, key=lambda x: -x[2]):
            out.append(f"    {a:9s} {b:9s} {r:.3f}")
    sr = [x[2] for x in groups["same representation"]]
    sf = [x[2] for x in groups["same model family"]]
    sn = [x[2] for x in groups["sharing neither"]]
    p_all = mannwhitneyu(sr, sf + sn, alternative="greater").pvalue
    p_fam = mannwhitneyu(sr, sf, alternative="greater").pvalue
    p_fn = mannwhitneyu(sf, sn, alternative="greater").pvalue
    out += ["", f"same representation vs all other pairs:   one-sided Mann-Whitney p = {p_all:.4f}",
            f"same representation vs same model family: p = {p_fam:.4f}",
            f"same model family vs sharing neither:     p = {p_fn:.4f}", ""]

    # MLP-specific lines
    out.append("The MLP against each other pipeline:")
    for k in KEYS:
        if k != "norm_mlp":
            out.append(f"    norm_mlp vs {k:9s} rho {M.loc['norm_mlp', k]:.3f}")
    out.append("")

    # magnitudes
    rows = []
    for k in KEYS:
        d = D[k]
        clean = 100 * d[d.axis == "clean"].accuracy.values[0]
        row = {"pipeline": k, "clean": clean}
        for ax, case, label in TABLE_CASES:
            row[label] = d[(d.axis == ax) & (d.case == case)].d_acc.values[0]
        rows.append(row)
    mag = pd.DataFrame(rows).set_index("pipeline")
    mag.to_csv(os.path.join(RESULTS, f"{cfg}_magnitudes.csv"))
    out.append("Accuracy change (points) under eight shifts, every pipeline; clean accuracy in %:")
    out.append(mag.round(1).T.to_string()); out.append("")

    # spread of magnitudes, as in transfer.py but with seven pipelines
    A = pd.DataFrame({k: L[k][cases] for k in KEYS})
    big = A[(A.abs() >= 5).any(axis=1)]
    ratio = (big.abs().max(axis=1) / big.abs().min(axis=1).clip(lower=0.1))
    out.append(f"On the {len(big)} cases costing some pipeline at least 5 points, the same shift costs a "
               f"median {ratio.median():.1f}x more for one pipeline than another.")
    text = "\n".join(out)
    open(os.path.join(RESULTS, f"{cfg}_transfer_mlp.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
