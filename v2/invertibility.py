"""Does a cheap invertibility measure predict what augmentation will buy?

The repair rule says deterministic shifts are learnable and stochastic ones are
not. That is a claim about whether the shifted spectrum determines the clean one,
so it should be measurable WITHOUT retraining a classifier: fit a map from the
shifted spectrum back to its clean counterpart and record how much of the clean
signal it recovers.

For each axis we report, on held-out planets:
  R2_inv   variance of the clean spectrum explained by a ridge map from the
           shifted spectrum (per-spectrum normalized, so it measures shape
           recovery rather than scale)
  dCorr    median per-planet correlation between shifted and clean spectrum,
           a zero-parameter baseline for the same idea
and compare them against the augmentation gain already measured, so the question
"does invertibility predict repairability?" is answered on four axes.

Usage: python invertibility.py --config ariel
Writes results/{config}_invertibility.txt / .csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bin_spectra import bin_native  # noqa: E402
from common import DATA, RESULTS, SEED, SNR, TESTS, centres, configs, load_split  # noqa: E402
from noise import sigma_matrix  # noqa: E402
from augment import corr_noise  # noqa: E402

# axis label -> how to build the shifted spectra for the pooled test planets
RERENDER = {"stellar spots 20%": "tlse_spots20", "stellar spots 10%": "tlse_spots10",
            "haze 3e7": "haze_3p0e7", "haze 2e6": "haze_2p0e6"}
INJECTED = {"correlated noise SNR 8": ("correlated", 8), "correlated noise SNR 5": ("correlated", 5),
            "white noise SNR 8": ("white", 8), "white noise SNR 5": ("white", 5)}


def norm(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-30
    return (X - mu) / sd


def invertibility(Xshift, Xclean, seed=SEED):
    """Ridge from shifted -> clean on half the planets, R2 on the other half."""
    good = np.all(np.isfinite(Xshift), axis=1) & np.all(np.isfinite(Xclean), axis=1)
    Xshift, Xclean = Xshift[good], Xclean[good]
    A, B = norm(Xshift), norm(Xclean)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(A)); h = len(A) // 2
    tr, te = idx[:h], idx[h:]
    def r2_of(model):
        model.fit(A[tr], B[tr]); P = model.predict(A[te])
        return 1 - ((B[te] - P) ** 2).sum() / ((B[te] - B[te].mean(axis=0)) ** 2).sum()
    r2 = r2_of(Ridge(alpha=1.0))
    r2nl = r2_of(MLPRegressor(hidden_layer_sizes=(128,), max_iter=250, random_state=SEED,
                              early_stopping=True, n_iter_no_change=8))
    dcorr = np.median([np.corrcoef(A[i], B[i])[0, 1] for i in te[:1500]])
    return float(r2), float(r2nl), float(dcorr)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel")
    a = ap.parse_args(); cfg = a.config
    cen = centres(cfg); edges = np.array(configs()[cfg]["edges"])
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    Xc, ys, Ps, Xnf = [], [], [], []
    for t in TESTS:
        X, y, P = load_split(t, cfg)
        Xc.append(X); ys.append(y); Ps.append(P)
        Xnf.append(np.load(os.path.join(DATA, f"{t}_{cfg}.npy")).astype(float))
    Xc = np.vstack(Xc); Xnf = np.vstack(Xnf); P = pd.concat(Ps, ignore_index=True)

    rows = []
    for label, case in RERENDER.items():
        Xn = np.vstack([np.load(os.path.join(DATA, f"{t}_native_{case}.npy")).astype(float) for t in TESTS])
        Xb = bin_native(Xn, wl, edges)
        sig = sigma_matrix(Xnf, P["s temperature"].to_numpy(), cen, SNR, "ariel", None)
        rng = np.random.default_rng(4242)
        Xs = Xb + rng.normal(0, 1, sig.shape) * sig
        r2, r2nl, dc = invertibility(Xs, Xc)
        rows.append(dict(axis=label, kind="deterministic", r2_lin=r2, r2_nonlin=r2nl, corr=dc))
        print(f"  {label:<22} linear {r2:6.3f}  nonlinear {r2nl:6.3f}", flush=True)
    for label, (kind, snr) in INJECTED.items():
        rng = np.random.default_rng(4243)
        Xs = corr_noise(Xc, rng, snr, kind=kind, Xnf=Xnf, params=P, cen=cen)
        r2, r2nl, dc = invertibility(Xs, Xc)
        rows.append(dict(axis=label, kind="stochastic", r2_lin=r2, r2_nonlin=r2nl, corr=dc))
        print(f"  {label:<22} linear {r2:6.3f}  nonlinear {r2nl:6.3f}", flush=True)

    df = pd.DataFrame(rows)
    # join against the measured augmentation gain
    aug = {}
    for f in ("ariel_augment.csv", "ariel_augment_white.csv"):
        p = os.path.join(RESULTS, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            for _, r in d[d.case != "clean"].iterrows():
                aug[f"{r['axis']}|{r['case']}"] = r.get("pct_of_gap", np.nan)
    KEY = {"stellar spots 20%": "stellar spots|tlse_spots20", "stellar spots 10%": "stellar spots|tlse_spots10",
           "haze 3e7": "haze|haze_3p0e7", "haze 2e6": "haze|haze_2p0e6",
           "correlated noise SNR 8": "correlated noise|snr8", "correlated noise SNR 5": "correlated noise|snr5",
           "white noise SNR 8": "white noise|snr8", "white noise SNR 5": "white noise|snr5"}
    df["pct_recovered"] = [aug.get(KEY[a], np.nan) for a in df.axis]
    L = [f"Does invertibility predict repairability? configuration {cfg}", "",
         "R2_inv: variance of the clean spectrum recovered by a ridge map from the shifted",
         "spectrum, fitted and scored on disjoint halves of the pooled test planets.",
         "pct_recovered: the augmentation gain already measured for the same case.", "",
         f"{'axis':<24}{'kind':<15}{'linear':>8}{'nonlin':>8}{'recovered':>11}"]
    for _, r in df.iterrows():
        pr = f"{r['pct_recovered']:.0f}%" if np.isfinite(r["pct_recovered"]) else "n/a"
        L.append(f"{r['axis']:<24}{r['kind']:<15}{r['r2_lin']:8.3f}{r['r2_nonlin']:8.3f}{pr:>11}")
    ok = df.dropna(subset=["pct_recovered"])
    if len(ok) > 2:
        from scipy.stats import spearmanr, pearsonr
        L += ["", f"Spearman(R2_inv, recovered) = {spearmanr(ok.r2_nonlin, ok.pct_recovered).statistic:.3f}"
                  f"   Pearson = {pearsonr(ok.r2_nonlin, ok.pct_recovered)[0]:.3f}   (n = {len(ok)})"]
        d = ok[ok.kind == "deterministic"]; s = ok[ok.kind == "stochastic"]
        L += [f"deterministic: linear {d.r2_lin.min():.3f}-{d.r2_lin.max():.3f}, nonlinear {d.r2_nonlin.min():.3f}-{d.r2_nonlin.max():.3f}, recovered {d.pct_recovered.min():.0f}-{d.pct_recovered.max():.0f}%",
              f"stochastic:    linear {s.r2_lin.min():.3f}-{s.r2_lin.max():.3f}, nonlinear {s.r2_nonlin.min():.3f}-{s.r2_nonlin.max():.3f}, recovered {s.pct_recovered.min():.0f}-{s.pct_recovered.max():.0f}%"]
    open(os.path.join(RESULTS, f"{cfg}_invertibility.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, f"{cfg}_invertibility.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
