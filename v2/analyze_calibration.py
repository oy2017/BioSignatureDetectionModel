"""Calibration and operating-threshold transfer for the frozen pipeline (H1, H6).

  1. Reliability curves (equal-count bins, Wilson intervals) for every model
     in results/{config}_probs.parquet, plus ECE and Brier.
  2. Precision-recall curve of the frozen best pipeline on the pooled clean
     test sets, and on the same planets re-rendered under a 10^4 Pa cloud deck
     (data/{split}_native_cloud_1e4Pa.npy, same noise seeds), with the decision
     threshold and precision required for 90 / 95 / 99 % recall in each case.
Writes results/{config}_calibration.txt, results/{config}_reliability.csv,
results/{config}_pr.parquet (curves for the figure script).

Usage: python analyze_calibration.py --config ariel [--cloud cloud_1e4Pa]
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from bin_spectra import bin_native
from common import DATA, MODELS, RESULTS, SNR, TESTS, base_config, centres, configs, ece, load_split, noise_spec
from noise import sigma_matrix


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def reliability(p, y, n_bins=10):
    order = np.argsort(p); p, y = p[order], y[order]
    out = []
    for b in np.array_split(np.arange(len(p)), n_bins):
        lo, hi = wilson(y[b].sum(), len(b))
        out.append(dict(mean_pred=p[b].mean(), observed=y[b].mean(), n=len(b), lo=lo, hi=hi))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ariel")
    ap.add_argument("--cloud", default="cloud_1e4Pa")
    a = ap.parse_args()
    cfg = a.config
    probs = pd.read_parquet(os.path.join(RESULTS, f"{cfg}_probs.parquet"))
    y = probs["y"].to_numpy()
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    lines = [f"Calibration, configuration {cfg} (pooled test sets, n={len(y)}, equal-count bins)", ""]
    rel_rows = []
    for k in [c for c in probs.columns if c != "y"]:
        p = probs[k].to_numpy()
        r = reliability(p, y); r["model"] = k; rel_rows.append(r)
        from sklearn.metrics import brier_score_loss, roc_auc_score
        lines.append(f"  {k:10s} ECE {ece(p, y):.4f}  Brier {brier_score_loss(y, p):.4f}  AUC {roc_auc_score(y, p):.4f}  "
                     f"max |gap| {np.abs(r['mean_pred'] - r['observed']).max():.3f}")
    pd.concat(rel_rows).to_csv(os.path.join(RESULTS, f"{cfg}_reliability.csv"), index=False)

    # PR: clean vs cloud deck, frozen best pipeline
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    feats, model = fr["features"], fr["model"]
    edges = np.array(configs()[base_config(cfg)]["edges"]); cen = centres(cfg); nshape, nlevel = noise_spec(cfg)
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    p_clean, p_cloud, ys = [], [], []
    have_cloud = all(os.path.exists(os.path.join(DATA, f"{t}_native_{a.cloud}.npy")) for t in TESTS)
    for k, t in enumerate(TESTS):
        X, yt, P = load_split(t, cfg)
        p_clean.append(model.predict_proba(feats.transform(X))[:, 1]); ys.append(yt)
        if have_cloud:
            Xn = np.load(os.path.join(DATA, f"{t}_native_{a.cloud}.npy")).astype(float)
            v = np.all(np.isfinite(Xn), axis=1)
            Xb = np.full((len(Xn), len(cen)), np.nan); Xb[v] = bin_native(Xn[v], wl, edges)
            Xs = np.full_like(Xb, np.nan)
            # sigma comes from the CLEAN spectrum, exactly as evaluate_shifts.py
            # builds this case: a cloud that suppresses features must not also be
            # rewarded with less noise. Passing the cloudy spectrum here made the
            # deck look 0.7 points cheaper than Table 3 reports for it.
            Xnf = np.load(os.path.join(DATA, f"{t}_{cfg}.npy")).astype(float)
            sig = sigma_matrix(Xnf[v], P[v]["s temperature"].to_numpy(), cen, SNR, nshape, nlevel)
            rng = np.random.default_rng(2000 + k + 1)
            Xs[v] = Xb[v] + rng.normal(0.0, 1.0, sig.shape) * sig
            pc = np.full(len(Xn), np.nan); pc[v] = model.predict_proba(feats.transform(Xs[v]))[:, 1]
            p_cloud.append(pc)
    pc, yy = np.concatenate(p_clean), np.concatenate(ys)
    curves = {}
    lines += ["", f"Operating thresholds, frozen {best}"]

    def pr_block(label, p, yv):
        prec, rec, thr = precision_recall_curve(yv, p)
        ap_ = average_precision_score(yv, p)
        acc = ((p >= 0.5) == yv).mean()
        i5 = np.argmin(np.abs(thr - 0.5))
        lines.append(f"  {label}: AP {ap_:.3f}, at threshold 0.5 precision {prec[i5]*100:.1f}% recall {rec[i5]*100:.1f}% accuracy {acc*100:.2f}%")
        for target in (0.90, 0.95, 0.99):
            ok = np.where(rec[:-1] >= target)[0]
            j = ok[-1] if len(ok) else 0
            lines.append(f"      recall >= {int(target*100)}%: threshold {thr[j]:.3f}, precision {prec[j]*100:.1f}%")
        curves[label] = pd.DataFrame({"precision": prec, "recall": rec})

    pr_block("clean", pc, yy)
    if have_cloud:
        pcl = np.concatenate(p_cloud); v = np.isfinite(pcl)
        pr_block(f"{a.cloud}", pcl[v], yy[v])
        lines.append(f"  (cloud: {(~v).sum()} planets with failed re-render excluded)")
    with open(os.path.join(RESULTS, f"{cfg}_calibration.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.concat([c.assign(case=k) for k, c in curves.items()]).to_parquet(os.path.join(RESULTS, f"{cfg}_pr.parquet"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
