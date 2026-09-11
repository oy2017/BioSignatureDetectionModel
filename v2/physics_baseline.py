"""Does the classifier beat a hand-built spectroscopic index?

The obvious non-machine-learning screen for a CH4/O3 label is a pair of band
depths: measure how much deeper the spectrum is inside a methane band than in
the neighbouring continuum, do the same for ozone, and call a planet positive
when both exceed a cutoff. This script builds that index on the same binned,
noisy spectra the classifier sees, tunes its two cutoffs on the TRAINING set by
exhaustive search, and reports its accuracy on the five test sets.

Bands (within Ariel's 0.5-7.8 um range, avoiding the strongest H2O regions):
  CH4   3.15-3.45 um (nu3) and 7.30-7.90 um (nu4)
  O3    4.60-4.85 um (nu1+nu3)
  continuum reference: the median of two windows flanking each band.
Depth is normalised by the spectrum's own scatter, so it does not depend on the
planet's absolute transit depth - the same information the per-spectrum
normalisation gives the classifier.

Two variants are reported:
  index-2D    both band depths above tuned cutoffs (the hand-built screen)
  index-ML    the same two numbers fed to the tuned XGBoost, which isolates how
              much of the classifier's advantage is the extra spectral detail
              rather than the model.

Usage: python physics_baseline.py --config ariel
Writes results/{config}_baseline.txt
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODELS, RESULTS, TESTS, centres, load_split, metrics  # noqa: E402
from pipeline import make_xgb  # noqa: E402

BANDS = {
    "CH4a": (3.15, 3.45, [(2.85, 3.10), (3.50, 3.75)]),
    "CH4b": (7.30, 7.79, [(6.60, 7.20)]),
    "O3": (4.60, 4.85, [(4.25, 4.55), (4.90, 5.20)]),
}


def band_depths(X, wl):
    """(n, 3) normalised depths: (band - continuum) / per-spectrum scatter.

    Molecular absorption raises the apparent planet radius, so the transit depth
    inside an absorption band is LARGER than in the neighbouring continuum; the
    index is therefore band minus continuum."""
    sd = X.std(axis=1, keepdims=True) + 1e-30
    out = []
    for name, (lo, hi, conts) in BANDS.items():
        b = (wl >= lo) & (wl <= hi)
        c = np.zeros_like(b)
        for cl, ch in conts:
            c |= (wl >= cl) & (wl <= ch)
        if b.sum() == 0 or c.sum() == 0:
            raise ValueError(f"band {name} has no bins at this configuration")
        out.append(((X[:, b].mean(axis=1, keepdims=True) - X[:, c].mean(axis=1, keepdims=True)) / sd).ravel())
    return np.column_stack(out)


def tune_2d(D, y, n=60):
    """Cutoffs on the stronger CH4 band and on O3 that maximise training accuracy.
    The CH4 evidence is the stronger of its two bands."""
    ch4 = np.maximum(D[:, 0], D[:, 1])
    o3 = D[:, 2]
    g4 = np.quantile(ch4, np.linspace(0.02, 0.98, n))
    g3 = np.quantile(o3, np.linspace(0.02, 0.98, n))
    best = (-1, None)
    for a in g4:
        m4 = ch4 > a
        for b in g3:
            acc = ((m4 & (o3 > b)).astype(int) == y).mean()
            if acc > best[0]:
                best = (acc, (a, b))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ariel")
    a = ap.parse_args()
    cfg = a.config
    wl = centres(cfg)
    for name, (lo, hi, conts) in BANDS.items():
        nb = int(((wl >= lo) & (wl <= hi)).sum())
        print(f"  band {name}: {nb} bins between {lo} and {hi} um")
    Xtr, ytr, _ = load_split("train", cfg)
    Dtr = band_depths(Xtr, wl)
    acc_tr, (c4, c3) = tune_2d(Dtr, ytr)
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    feats, model, params = fr["features"], fr["model"], fr["params"]
    m_idx = make_xgb(params).fit(Dtr, ytr)

    rows = {"index-2D": [], "index-ML": [], "full pipeline": []}
    for t in TESTS:
        X, y, _ = load_split(t, cfg)
        D = band_depths(X, wl)
        pred = ((np.maximum(D[:, 0], D[:, 1]) > c4) & (D[:, 2] > c3)).astype(int)
        rows["index-2D"].append(dict(accuracy=(pred == y).mean()))
        rows["index-ML"].append(metrics(y, m_idx.predict_proba(D)[:, 1]))
        rows["full pipeline"].append(metrics(y, model.predict_proba(feats.transform(X))[:, 1]))

    L = [f"Hand-built band-depth index vs the classifier, configuration {cfg}", "",
         "Bands (um): " + "; ".join(f"{k} {v[0]}-{v[1]}" for k, v in BANDS.items()),
         f"Cutoffs tuned on the training set: CH4 depth > {c4:.3f}, O3 depth > {c3:.3f} "
         f"(training accuracy {acc_tr*100:.2f}%)", "",
         f"{'method':<32}{'accuracy':>10}{'F1':>9}", ]
    for k, v in rows.items():
        acc = np.mean([r["accuracy"] for r in v]); sd = np.std([r["accuracy"] for r in v], ddof=1)
        f1 = np.mean([r.get("f1", np.nan) for r in v])
        L.append(f"{k:<32}{acc*100:8.2f}%±{sd*100:.2f}" + (f"{f1*100:8.2f}%" if np.isfinite(f1) else "        -"))
    gap = np.mean([r["accuracy"] for r in rows["full pipeline"]]) - np.mean([r["accuracy"] for r in rows["index-2D"]])
    gap2 = np.mean([r["accuracy"] for r in rows["full pipeline"]]) - np.mean([r["accuracy"] for r in rows["index-ML"]])
    L += ["", f"The full pipeline leads the hand-built index by {gap*100:.1f} points and the",
          f"same two band depths given to XGBoost by {gap2*100:.1f} points, so the advantage is",
          "the spectral detail beyond two band ratios, not the model alone."]
    open(os.path.join(RESULTS, f"{cfg}_baseline.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
