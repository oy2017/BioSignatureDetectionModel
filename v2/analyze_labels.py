"""Label sensitivity and error structure for the frozen best pipeline (H3).

  1. Threshold sensitivity: move both abundance cutoffs together by
     -0.5 .. +0.5 dex, relabel the SAME spectra, retrain the frozen pipeline's
     model (features unchanged) and report accuracy, majority baseline, gain.
  2. Margin analysis: accuracy of the frozen pipeline against the dex distance
     of each test planet to the nearest label flip, in bins, with the majority
     baseline per bin and the mean predicted probability.
  3. Amplitude analysis: error rate against the per-spectrum feature amplitude
     (std of the noise-free binned depths) in quintiles, and against the
     amplitude-to-noise ratio.
Writes results/{config}_labels.txt and results/{config}_labels.parquet (per
planet: margin, amplitude, prob, correct) for the figure script.

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

BIO_CH4, BIO_O3 = -6.0, -7.0


def margin(P):
    ch4, o3 = P["atm CH4"].to_numpy(), P["atm O3"].to_numpy()
    pos = (ch4 > BIO_CH4) & (o3 > BIO_O3)
    m = np.where(pos, np.minimum(ch4 - BIO_CH4, o3 - BIO_O3),
                 np.maximum(BIO_CH4 - ch4, BIO_O3 - o3))
    return m


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
    lines = [f"Label sensitivity and error structure, configuration {cfg}, pipeline {best}", ""]

    # 1. threshold sensitivity
    lines.append("1. Threshold sensitivity (both cutoffs moved together; spectra unchanged; model retrained)")
    lines.append(f"{'shift':>6s} {'CH4/O3 cutoff':>14s} {'train pos':>9s} {'test pos':>8s} {'accuracy':>9s} {'majority':>9s} {'gain':>7s} {'F1':>6s} {'Brier':>7s}")
    from common import metrics
    for d in (-0.5, -0.25, 0.0, 0.25, 0.5):
        c4, c3 = BIO_CH4 + d, BIO_O3 + d
        ytr_d = ((Ptr["atm CH4"] > c4) & (Ptr["atm O3"] > c3)).astype(int).to_numpy()
        yte_d = ((Pte["atm CH4"] > c4) & (Pte["atm O3"] > c3)).astype(int).to_numpy()
        m = make(params).fit(Ztr, ytr_d)
        p = m.predict_proba(Zte)[:, 1]
        mt = metrics(yte_d, p)
        maj = max(yte_d.mean(), 1 - yte_d.mean())
        lines.append(f"{d:+6.2f} {c4:6.2f}/{c3:6.2f}  {ytr_d.mean()*100:8.1f}% {yte_d.mean()*100:7.1f}% "
                     f"{mt['accuracy']*100:8.2f}% {maj*100:8.2f}% {(mt['accuracy']-maj)*100:+6.2f}% "
                     f"{mt['f1']:.3f} {mt['brier']:.4f}")

    # 2. margin
    p = model.predict_proba(Zte)[:, 1]
    correct = ((p >= 0.5).astype(int) == yte)
    mg = margin(Pte)
    lines += ["", "2. Accuracy against margin to the nearest label flip (dex)"]
    lines.append(f"{'margin':>10s} {'n':>6s} {'pos':>6s} {'accuracy':>9s} {'majority':>9s} {'gain':>7s} {'mean p':>7s}")
    edges = [0, 0.25, 0.5, 1.0, 2.0, 99]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mg >= lo) & (mg < hi)
        if m.sum() == 0:
            continue
        pos = yte[m].mean(); maj = max(pos, 1 - pos)
        lines.append(f"{lo:4.2f}-{hi if hi < 99 else '  ':<4} {m.sum():6d} {pos*100:5.1f}% {correct[m].mean()*100:8.2f}% "
                     f"{maj*100:8.2f}% {(correct[m].mean()-maj)*100:+6.2f}% {p[m].mean():6.3f}")

    # 3. amplitude
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
