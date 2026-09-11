"""Retire three stated limitations by measuring what they are worth.

  train-variance   The five-set standard deviations measure test-sample scatter
                   only. Retrain the primary pipeline on bootstrap resamples of
                   the training set and report the spread of the five-set mean,
                   so the paper can quote a training-draw uncertainty too.
  spot-contrast    The stellar-contamination result adopts T_spot = 0.85 T_eff.
                   Recompute the contamination at 0.80, 0.85 and 0.90 and report
                   the loss at each, so the reader sees whether the conclusion
                   depends on the adopted contrast.
  corr-length      The correlated-noise family smooths white noise with a fixed
                   kernel. Sweep the correlation length and report the loss, so
                   the reader sees whether "correlated noise costs more than
                   white" depends on the kernel width.

Usage: python sensitivity.py --config ariel [--parts train-variance spot-contrast corr-length]
Writes results/{config}_sensitivity.txt
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bin_spectra import bin_native  # noqa: E402
from common import (DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features, centres,  # noqa: E402
                    configs, load_split, metrics)
from noise import add_noise, sigma_matrix  # noqa: E402
from pipeline import make_xgb  # noqa: E402


def pooled_test(cfg):
    Xs, ys = [], []
    for t in TESTS:
        X, y, _ = load_split(t, cfg)
        Xs.append(X); ys.append(y)
    return Xs, ys


def noise_floor(cfg, cen):
    """The per-spectrum noise level these planets were actually observed at.

    Must match evaluate_shifts.noise_floor. An earlier version of this function
    estimated the level from the scatter of adjacent bins, which assumes a
    locally flat spectrum; the Ariel layout has three photometric bands and hard
    resolution jumps, so that estimator came out 1.46x too large and every
    injected level was mislabelled by that factor. The generating sigma is
    available directly from the noise-free spectra.
    """
    sig = []
    for t in TESTS:
        _, _, P = load_split(t, cfg, noisy=False)
        Xnf = np.load(os.path.join(DATA, f"{t}_{cfg}.npy")).astype(float)
        sig.append(np.median(sigma_matrix(Xnf, P["s temperature"].to_numpy(), cen),
                             axis=1, keepdims=True))
    return np.vstack(sig)


def train_variance(cfg, feats_kind, params, n_boot=10):
    Xtr, ytr, _ = load_split("train", cfg)
    Xs, ys = pooled_test(cfg)
    rng = np.random.default_rng(SEED)
    means = []
    for b in range(n_boot):
        idx = rng.choice(len(ytr), len(ytr), replace=True)
        f = Features(feats_kind).fit(Xtr[idx])
        mdl = make_xgb(params)
        mdl.set_params(random_state=SEED + b)
        m = mdl.fit(f.transform(Xtr[idx]), ytr[idx])
        accs = [metrics(y, m.predict_proba(f.transform(X))[:, 1])["accuracy"] for X, y in zip(Xs, ys)]
        means.append(np.mean(accs))
        print(f"  bootstrap {b+1}/{n_boot}: {means[-1]*100:.2f}%", flush=True)
    return np.array(means)


def spot_contrast(cfg, feats, model, fracs=(0.80, 0.85, 0.90), coverages=(0.05, 0.20)):
    import shift_tlse as T
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    T.set_grid(wl)
    edges = np.array(configs()[cfg]["edges"]); cen = centres(cfg)
    out = []
    for k, t in enumerate(TESTS):
        P = pd.read_parquet(os.path.join(DATA, f"{t}_params.parquet"))
        X = np.load(os.path.join(DATA, f"{t}_native.npy")).astype(np.float64)
        y = P["label_co"].to_numpy()
        Tst = P["s temperature"].to_numpy()
        logg = np.log10(T.G_SUN * P["s mass"].to_numpy() / P["s radius"].to_numpy() ** 2 * 100)
        for frac in fracs:
            old = T.T_SPOT_FRAC
            T.T_SPOT_FRAC = frac
            for cov in coverages:
                eps = T.contamination(Tst, logg, cov, 0.0)
                Xb = bin_native(X * eps, wl, edges)
                Xn, _ = add_noise(Xb, P, cen, snr=SNR, shape="ariel", seed=2000 + k + 1)
                acc = metrics(y, model.predict_proba(feats.transform(Xn))[:, 1])["accuracy"]
                out.append(dict(split=t, frac=frac, coverage=cov, accuracy=acc))
            T.T_SPOT_FRAC = old
    return pd.DataFrame(out).groupby(["frac", "coverage"])["accuracy"].mean().reset_index()


def corr_length(cfg, feats, model, sigmas=(1.0, 3.0, 8.0, 20.0), snr_eff=8):
    Xs, ys = pooled_test(cfg)
    X = np.vstack(Xs); y = np.concatenate(ys)
    sig_n = noise_floor(cfg, centres(cfg))
    m = np.sqrt((SNR / snr_eff) ** 2 - 1.0)
    rng = np.random.default_rng(SEED)
    rows = []
    base = metrics(y, model.predict_proba(feats.transform(X))[:, 1])["accuracy"]
    raw = rng.normal(0, 1, X.shape)
    white = X + raw * sig_n * m
    rows.append(dict(kernel="white (sigma 0)", accuracy=metrics(y, model.predict_proba(feats.transform(white))[:, 1])["accuracy"]))
    for s in sigmas:
        z = gaussian_filter1d(raw, sigma=s, axis=1)
        z /= z.std(axis=1, keepdims=True) + 1e-12
        Xp = X + z * sig_n * m
        rows.append(dict(kernel=f"sigma {s:g} bins",
                         accuracy=metrics(y, model.predict_proba(feats.transform(Xp))[:, 1])["accuracy"]))
    return base, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ariel")
    ap.add_argument("--parts", nargs="*", default=["train-variance", "spot-contrast", "corr-length"])
    a = ap.parse_args()
    cfg = a.config
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    feats, model, params = fr["features"], fr["model"], fr["params"]
    kind = best.split("_")[0]
    L = [f"Sensitivity of three stated limitations, configuration {cfg}, pipeline {best}", ""]

    if "train-variance" in a.parts:
        print("train variance:", flush=True)
        mu = train_variance(cfg, kind, params)
        L += ["1. Training-draw uncertainty (10 bootstrap resamples of the training set,",
              "   each retrained and scored on the same five test sets)",
              f"   five-set mean accuracy: {mu.mean()*100:.2f}% ± {mu.std(ddof=1)*100:.2f} across training draws",
              f"   range {mu.min()*100:.2f}-{mu.max()*100:.2f}%; the paper's single-draw value is the reference model", ""]

    if "spot-contrast" in a.parts:
        print("spot contrast:", flush=True)
        df = spot_contrast(cfg, feats, model)
        L += ["2. Stellar-contamination result against the adopted spot contrast",
              "   (T_spot / T_eff; the paper adopts 0.85)", "",
              f"   {'contrast':>9} {'5% coverage':>13} {'20% coverage':>13}"]
        for frac in sorted(df["frac"].unique()):
            s = df[df["frac"] == frac]
            a5 = s[s["coverage"] == 0.05]["accuracy"].iloc[0] * 100
            a20 = s[s["coverage"] == 0.20]["accuracy"].iloc[0] * 100
            L.append(f"   {frac:>9.2f} {a5:12.2f}% {a20:12.2f}%")
        L.append("")

    if "corr-length" in a.parts:
        print("correlation length:", flush=True)
        base, df = corr_length(cfg, feats, model)
        L += ["3. Correlated-noise cost against the smoothing kernel width, at effective SNR 8",
              f"   (clean baseline {base*100:.2f}%; the paper uses sigma 3 bins)", ""]
        for _, r in df.iterrows():
            L.append(f"   {r['kernel']:<18} {r['accuracy']*100:6.2f}%  ({(r['accuracy']-base)*100:+.2f})")
        L.append("")

    open(os.path.join(RESULTS, f"{cfg}_sensitivity.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
