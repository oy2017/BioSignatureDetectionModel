"""Label sensitivity and error structure for the frozen best pipeline, C/O label (v3).

v3 rewrite of v2/analyze_labels.py. The v2 label was a pair of abundance cutoffs; the v3
label is C/O > 1.0, a single chemistry input, so the two label-specific analyses change:

  1. Cut sensitivity: move the cut in dex of log10(C/O) by -0.10 .. +0.10 (C/O 0.79 .. 1.26),
     relabel the SAME spectra, retrain the frozen pipeline's model (features unchanged) and
     report accuracy, majority baseline, gain.
  2. Margin analysis: accuracy of the frozen pipeline against |log10(C/O)|, the dex distance
     of each test planet from the cut, in bins, with the majority baseline per bin and the
     mean predicted probability. C/O ~ U(0.2, 1.8) puts the carbon-rich side within 0.26 dex
     of the cut, so the bins are finer than v2's.
  3. Amplitude analysis: unchanged from v2 (label-agnostic).
Writes results/{config}_labels.txt and results/{config}_labels.parquet.

Usage: python analyze_labels.py --config ariel
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from common import DATA, MODELS, RESULTS, TESTS, base_config, centres, load_split, noise_spec
from noise import sigma_matrix
from pipeline import make_rf, make_xgb

CO_CUT = 1.0
DEX_SHIFTS = (-0.10, -0.05, 0.0, 0.05, 0.10)
MARGIN_EDGES = [0, 0.05, 0.10, 0.20, 0.30, 99]


def margin(P):
    """Unsigned dex distance from the cut; positive class has log10(C/O) > 0."""
    return np.abs(np.log10(P["co_ratio"].to_numpy() / CO_CUT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ariel")
    a = ap.parse_args()
    cfg = a.config
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    feats, model, params = fr["features"], fr["model"], fr["params"]
    kind, mname = best.split("_")
    make = make_xgb if mname == "xgb" else make_rf

    Xtr, ytr, Ptr = load_split("train", cfg)
    tests = [load_split(t, cfg) for t in TESTS]
    Xte = np.vstack([t[0] for t in tests]); yte = np.concatenate([t[1] for t in tests])
    Pte = pd.concat([t[2] for t in tests], ignore_index=True)
    Ztr, Zte = feats.transform(Xtr), feats.transform(Xte)
    lines = [f"Label sensitivity and error structure, configuration {cfg}, pipeline {best}, label C/O > {CO_CUT}", ""]

    # 1. cut sensitivity
    lines.append("1. Cut sensitivity (cut moved in dex of log10 C/O; spectra unchanged; model retrained)")
    lines.append(f"{'shift':>6s} {'C/O cut':>8s} {'train pos':>9s} {'test pos':>8s} {'accuracy':>9s} {'majority':>9s} {'gain':>7s} {'F1':>6s} {'Brier':>7s}")
    from common import metrics
    for d in DEX_SHIFTS:
        cut = CO_CUT * 10 ** d
        ytr_d = (Ptr["co_ratio"] > cut).astype(int).to_numpy()
        yte_d = (Pte["co_ratio"] > cut).astype(int).to_numpy()
        m = make(params).fit(Ztr, ytr_d)
        p = m.predict_proba(Zte)[:, 1]
        mt = metrics(yte_d, p)
        maj = max(yte_d.mean(), 1 - yte_d.mean())
        lines.append(f"{d:+6.2f} {cut:8.3f}  {ytr_d.mean()*100:8.1f}% {yte_d.mean()*100:7.1f}% "
                     f"{mt['accuracy']*100:8.2f}% {maj*100:8.2f}% {(mt['accuracy']-maj)*100:+6.2f}% "
                     f"{mt['f1']:.3f} {mt['brier']:.4f}")

    # 2. margin
    p = model.predict_proba(Zte)[:, 1]
    correct = ((p >= 0.5).astype(int) == yte)
    mg = margin(Pte)
    lines += ["", "2. Accuracy against |log10(C/O)| distance from the cut (dex)"]
    lines.append(f"{'margin':>10s} {'n':>6s} {'pos':>6s} {'accuracy':>9s} {'majority':>9s} {'gain':>7s} {'mean p':>7s}")
    for lo, hi in zip(MARGIN_EDGES[:-1], MARGIN_EDGES[1:]):
        m = (mg >= lo) & (mg < hi)
        if m.sum() == 0:
            continue
        pos = yte[m].mean(); maj = max(pos, 1 - pos)
        lines.append(f"{lo:4.2f}-{hi if hi < 99 else '  ':<4} {m.sum():6d} {pos*100:5.1f}% {correct[m].mean()*100:8.2f}% "
                     f"{maj*100:8.2f}% {(correct[m].mean()-maj)*100:+6.2f}% {p[m].mean():6.3f}")

    # 3. amplitude (unchanged from v2)
    Xclean = np.vstack([np.load(os.path.join(DATA, f"{t}_{base_config(cfg)}.npy")) for t in TESTS]).astype(float)
    amp = Xclean.std(axis=1)
    nshape, nlevel = noise_spec(cfg)
    sig = sigma_matrix(Xclean, Pte["s temperature"].to_numpy(), centres(cfg), shape=nshape, level_ppm=nlevel)
    anr = amp / np.median(sig, axis=1)
    lines += ["", "3. Error rate against feature amplitude (std of the noise-free binned depth) in quintiles"]
    q = pd.qcut(amp, 5, labels=False)
    for i in range(5):
        m = q == i
        lines.append(f"  quintile {i+1}: amplitude {amp[m].min():.2e}-{amp[m].max():.2e}  error {100*(1-correct[m].mean()):5.2f}%  "
                     f"amp/noise median {np.median(anr[m]):.2f}")
    with open(os.path.join(RESULTS, f"{cfg}_labels.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.DataFrame({"margin": mg, "amplitude": amp, "amp_noise": anr, "prob": p, "y": yte,
                  "correct": correct}).to_parquet(os.path.join(RESULTS, f"{cfg}_labels.parquet"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
