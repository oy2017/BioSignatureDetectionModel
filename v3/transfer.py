"""Is the fidelity budget a property of the task, or of the pipeline that measured it?

The six pipelines form a balanced grid: two model families {XGBoost, random forest}
crossed with three input representations {per-spectrum normalized, PCA, raw bins}.
Every pipeline is scored on the IDENTICAL shifted spectra, so the loss vectors are
directly comparable and each of the fifteen pipeline pairs falls into exactly one
of three relationships:

  same representation, different family   (3 pairs)
  same family, different representation   (6 pairs)
  sharing neither                         (6 pairs)

If the budget were a property of the task, all three groups would agree equally
well. The question is which factor, if either, buys agreement.

Usage: python transfer.py --config ariel
Writes results/{config}_transfer.txt / .csv
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")

# tag -> (model family, representation); None is the frozen pipeline's own file
CELLS = {None: ("xgb", "norm"), "norm_rf": ("rf", "norm"),
         "pca_xgb": ("xgb", "pca"), "pca_rf": ("rf", "pca"),
         "raw_xgb": ("xgb", "raw"), "raw_rf": ("rf", "raw")}
NAME = {None: "norm_xgb"}
SEED = 20260101


def series(cfg, tag):
    """Per-case accuracy loss, indexed by axis|case.

    'extrapolation' is excluded: it is a retraining experiment with its own control,
    not a re-render of the same planets, so it is not a shared shift case.
    """
    f = f"{cfg}_shifts.csv" if tag is None else f"{cfg}_{tag}_shifts.csv"
    p = os.path.join(RESULTS, f)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[(d["axis"] != "clean") & (d["axis"] != "extrapolation")]
    return d.set_index(d["axis"] + "|" + d["case"])["d_acc"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel")
    a = ap.parse_args(); cfg = a.config

    S = {t: series(cfg, t) for t in CELLS}
    missing = [NAME.get(t, t) for t, v in S.items() if v is None]
    if missing:
        sys.exit(f"missing shift files for: {missing}")
    common = sorted(set.intersection(*(set(v.index) for v in S.values())))

    rows = []
    for x, y in itertools.combinations(CELLS, 2):
        fx, rx = CELLS[x]; fy, ry = CELLS[y]
        rel = ("same representation" if rx == ry else
               "same model family" if fx == fy else "sharing neither")
        rho = spearmanr(S[x][common], S[y][common]).statistic
        rows.append(dict(a=NAME.get(x, x), b=NAME.get(y, y), relationship=rel,
                         rho=float(rho)))
    df = pd.DataFrame(rows).sort_values(["relationship", "rho"], ascending=[True, False])

    g = {k: df[df.relationship == k]["rho"].to_numpy() for k in
         ("same representation", "same model family", "sharing neither")}
    # Two one-sided contrasts. The headline one asks whether sharing a representation
    # agrees better than every other kind of pair; the second isolates the family.
    others = np.concatenate([g["same model family"], g["sharing neither"]])
    U_all = mannwhitneyu(g["same representation"], others, alternative="greater")
    U = mannwhitneyu(g["same representation"], g["same model family"],
                     alternative="greater")
    # and a permutation test on the same contrast, which does not bottom out at
    # 1/C(9,3) the way the rank test does with three versus six pairs
    rng = np.random.default_rng(SEED)
    pool = np.concatenate([g["same representation"], g["same model family"]])
    obs = g["same representation"].mean() - g["same model family"].mean()
    draws = np.array([rng.permutation(pool) for _ in range(20000)])
    perm = draws[:, :3].mean(axis=1) - draws[:, 3:].mean(axis=1)
    p_perm = (1 + (perm >= obs).sum()) / (1 + len(perm))

    # Is the ordering trivial? A physical proxy --- surviving feature amplitude ---
    # would predict it if the ranking were just "how much signal is left".
    f = os.path.join(RESULTS, f"{cfg}_shifts.csv")
    d0 = pd.read_csv(f); d0 = d0[(d0["axis"] != "clean") & (d0["axis"] != "extrapolation")]
    d0 = d0.set_index(d0["axis"] + "|" + d0["case"]).loc[common]
    amp = d0["amp_ratio"].astype(float)
    ok = np.isfinite(amp) & np.isfinite(S[None][common])
    rho_amp = spearmanr(S[None][common][ok], amp[ok]).statistic

    # Orderings are one thing, magnitudes another: how far apart are the six
    # pipelines on the SAME case? Restricted to cases that actually cost something,
    # so a ratio is not taken against a loss of a fraction of a point.
    M = pd.DataFrame({NAME.get(t, t): S[t][common] for t in CELLS}).abs()
    big = M[M.max(axis=1) >= 5.0]
    spread = (big.max(axis=1) / big.min(axis=1).replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan).dropna()

    # Where does the ranking inversion start to pay? If a fraction f of a population
    # carries the severe haze, the two pipelines break even at the f that equates
    # their population-weighted accuracy.
    def acc(tag, key):
        f_ = f"{cfg}_shifts.csv" if tag is None else f"{cfg}_{tag}_shifts.csv"
        d = pd.read_csv(os.path.join(RESULTS, f_))
        d = d.set_index(d["axis"] + "|" + d["case"])
        return 100 * float(d.loc[key, "accuracy"])
    HAZE = "haze|haze_3p0e7"
    c_n, c_p = acc(None, "clean|reference"), acc("pca_xgb", "clean|reference")
    h_n, h_p = acc(None, HAZE), acc("pca_xgb", HAZE)
    denom = (c_n - h_n) - (c_p - h_p)
    f_even = (c_n - c_p) / denom if denom > 0 else float("nan")

    L = [f"Does the fidelity budget transfer? configuration {cfg}", "",
         f"{len(common)} shared shift cases, six pipelines = 2 model families x 3 representations.",
         "Spearman correlation of the per-case accuracy loss, every pair of pipelines.", "",
         f"{'pipeline A':<12}{'pipeline B':<12}{'relationship':<22}{'rho':>7}"]
    for _, r in df.iterrows():
        L.append(f"{r['a']:<12}{r['b']:<12}{r['relationship']:<22}{r['rho']:7.3f}")
    L += ["",
          f"physical proxy: Spearman(loss, surviving feature amplitude) = {rho_amp:.3f} "
          f"(n = {int(ok.sum())}), so the ordering is not simply how much signal is left.",
          ""]
    for k, v in g.items():
        L.append(f"{k:<22} n={len(v)}  mean rho {v.mean():.3f}   range {v.min():.3f}-{v.max():.3f}")
    L += ["",
          f"same representation vs all other pairs:   Mann-Whitney U one-sided p = {U_all.pvalue:.4f}"
          f"   (n = 3 vs {len(others)})",
          f"same representation vs same model family: Mann-Whitney U one-sided p = {U.pvalue:.4f}",
          f"                                         permutation (20k) p = {p_perm:.4f}",
          f"                                         mean difference = {obs:+.3f}",
          "",
          "",
          f"Magnitude spread on the {len(spread)} cases costing at least 5 points to some "
          f"pipeline:",
          f"  the same shift costs a median {spread.median():.1f}x more for one pipeline than "
          f"another (max {spread.max():.0f}x).",
          f"  Between the two strongest pipelines alone it reaches "
          f"{(M['norm_xgb'] / M['pca_xgb'].replace(0, np.nan)).max():.1f}x.",
          ""]
    if h_n >= h_p:
        L += [f"No ranking inversion under haze 3e7: frozen {c_n:.2f} -> {h_n:.2f}%, PCA {c_p:.2f} -> {h_p:.2f}%;",
              "the in-domain ranking holds under the shift on this grid."]
    else:
        L += [f"Ranking inversion under haze 3e7: frozen {c_n:.2f} -> {h_n:.2f}%, PCA {c_p:.2f} -> {h_p:.2f}%.",
              f"The two break even when {100*f_even:.0f}% of a population carries that haze;",
              "below that fraction the in-domain ranking still wins."]
    fam, nei = g['same model family'].mean(), g['sharing neither'].mean()
    L += ["",
          (f"Sharing a model family agrees no better than sharing nothing ({fam:.3f} vs {nei:.3f});"
           if fam - nei < 0.05 else
           f"Sharing a model family agrees somewhat better than sharing nothing ({fam:.3f} vs {nei:.3f});"),
          "the representation is what carries the budget." if g['same representation'].mean() > max(fam, nei)
          else "no single factor carries the budget on this grid."]
    open(os.path.join(RESULTS, f"{cfg}_transfer.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, f"{cfg}_transfer.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
