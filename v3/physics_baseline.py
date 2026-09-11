"""Does the classifier beat a hand-built spectroscopic index?  (v3, C/O label)

The obvious non-machine-learning screen for a C/O label is a contrast between
carbon-bearing and oxygen-bearing band depths: carbon-rich atmospheres put their
carbon into CH4 and CO and starve H2O, so a planet is called carbon-rich when the
deeper of its carbon bands exceeds its water band by more than a cutoff. The index
is built on the same binned, noisy spectra the classifier sees, its single cutoff
is tuned on the TRAINING set by exhaustive search, and its accuracy is reported on
the five test sets.

Bands (um, within Ariel's 0.5-7.8):
  CH4   3.15-3.45 (nu3) and 7.30-7.79 (nu4)
  CO    4.50-4.85 (fundamental, 4.67) and 2.28-2.40 (first overtone)
  H2O   2.55-2.80 (the 2.7 um band)
  continuum reference: the mean of windows flanking each band.
Depth is normalised by the spectrum's own scatter, so it does not depend on the
planet's absolute transit depth - the same information the per-spectrum
normalisation gives the classifier.

Two variants are reported:
  index-1D    max(carbon band depths) - H2O band depth above a tuned cutoff
  index-ML    the same five band depths fed to the tuned XGBoost, which isolates
              how much of the classifier's advantage is the extra spectral detail
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODELS, RESULTS, TESTS, centres, load_split, metrics  # noqa: E402
from pipeline import make_xgb  # noqa: E402

BANDS = {
    "CH4a": (3.15, 3.45, [(2.85, 3.10), (3.50, 3.75)]),
    "CH4b": (7.30, 7.79, [(6.60, 7.20)]),
    "COa":  (4.50, 4.85, [(4.10, 4.40), (4.95, 5.30)]),
    "COb":  (2.28, 2.40, [(2.10, 2.22), (2.44, 2.52)]),
    "H2O":  (2.55, 2.80, [(2.44, 2.52), (2.85, 3.10)]),
}
CARBON = ("CH4a", "CH4b", "COa", "COb")


def band_depths(X, wl):
    """(n, 5) normalised depths: (band - continuum) / per-spectrum scatter.

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


def co_index(D):
    """Deepest carbon band minus the water band."""
    keys = list(BANDS)
    carbon = np.max(D[:, [keys.index(k) for k in CARBON]], axis=1)
    return carbon - D[:, keys.index("H2O")]


def tune_1d(idx, y, n=300):
    """Single cutoff on the C/O index that maximises training accuracy."""
    best = (-1, None)
    for a in np.quantile(idx, np.linspace(0.01, 0.99, n)):
        acc = ((idx > a).astype(int) == y).mean()
        if acc > best[0]:
            best = (acc, a)
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
    acc_tr, cut = tune_1d(co_index(Dtr), ytr)
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    feats, model, params = fr["features"], fr["model"], fr["params"]
    m_idx = make_xgb(params).fit(Dtr, ytr)

    rows = {"index-1D": [], "index-ML": [], "full pipeline": []}
    for t in TESTS:
        X, y, _ = load_split(t, cfg)
        D = band_depths(X, wl)
        pred = (co_index(D) > cut).astype(int)
        rows["index-1D"].append(dict(accuracy=(pred == y).mean()))
        rows["index-ML"].append(metrics(y, m_idx.predict_proba(D)[:, 1]))
        rows["full pipeline"].append(metrics(y, model.predict_proba(feats.transform(X))[:, 1]))

    L = [f"Hand-built band-depth index vs the classifier, configuration {cfg}, label C/O > 1.0", "",
         "Bands (um): " + "; ".join(f"{k} {v[0]}-{v[1]}" for k, v in BANDS.items()),
         f"Index: max(CH4, CO band depths) - H2O band depth; cutoff tuned on the training set: > {cut:.3f} "
         f"(training accuracy {acc_tr*100:.2f}%)", "",
         f"{'method':<32}{'accuracy':>10}{'F1':>9}", ]
    for k, v in rows.items():
        acc = np.mean([r["accuracy"] for r in v]); sd = np.std([r["accuracy"] for r in v], ddof=1)
        f1 = np.mean([r.get("f1", np.nan) for r in v])
        L.append(f"{k:<32}{acc*100:8.2f}%±{sd*100:.2f}" + (f"{f1*100:8.2f}%" if np.isfinite(f1) else "        -"))
    gap = np.mean([r["accuracy"] for r in rows["full pipeline"]]) - np.mean([r["accuracy"] for r in rows["index-1D"]])
    gap2 = np.mean([r["accuracy"] for r in rows["full pipeline"]]) - np.mean([r["accuracy"] for r in rows["index-ML"]])
    L += ["", f"The full pipeline leads the hand-built index by {gap*100:.1f} points and the",
          f"same five band depths given to XGBoost by {gap2*100:.1f} points, so the advantage is",
          "the spectral detail beyond band contrasts, not the model alone."]
    open(os.path.join(RESULTS, f"{cfg}_baseline.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
