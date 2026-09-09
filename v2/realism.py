"""Three checks a referee asked for, each answering a specific attack.

1. per-channel offsets   Per-spectrum normalisation annihilates a GLOBAL additive
                         offset, which is the one additive error a six-channel
                         payload will not make. Ariel has three photometers and
                         three spectrometers with independent zero points. This
                         injects an independent offset per channel.
2. floors                The paper reports no non-machine-learning comparator
                         beyond the majority class. This adds a logistic
                         regression on the same normalised bins (a linear floor)
                         and a band-depth index built on windows chosen by their
                         measured correlation with the labelling gases rather
                         than by textbook band centres.
3. realistic subset      The grid spans transit depths to tens of percent and
                         feature amplitudes far above anything Ariel will see.
                         This re-scores the frozen pipeline on the subset whose
                         depth and amplitude fall in the range of real targets.

Usage: python realism.py --config ariel
Writes results/{config}_realism.txt
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
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features, centres, configs, load_split, metrics  # noqa: E402
from noise import sigma_matrix  # noqa: E402
from pipeline import make_xgb  # noqa: E402

CHANNELS = ["VISPhot", "FGS1", "FGS2", "NIRSpec", "AIRS-CH0", "AIRS-CH1"]


def channel_index(cfg):
    lab = configs()[cfg]["channel"]
    return np.array([CHANNELS.index(c) for c in lab])


def per_channel_offset(cfg, feats, model, strengths=(0.25, 0.5, 1.0, 2.0)):
    ch = channel_index(cfg)
    cen = centres(cfg)
    rows = []
    Xs, ys, Ps, Xnf = [], [], [], []
    for t in TESTS:
        X, y, P = load_split(t, cfg)
        Xs.append(X); ys.append(y); Ps.append(P)
        Xnf.append(np.load(os.path.join(DATA, f"{t}_ariel.npy")).astype(float))
    X = np.vstack(Xs); y = np.concatenate(ys); P = pd.concat(Ps, ignore_index=True); Xnf = np.vstack(Xnf)
    sig = np.median(sigma_matrix(Xnf, P["s temperature"].to_numpy(), cen), axis=1, keepdims=True)
    base = metrics(y, model.predict_proba(feats.transform(X))[:, 1])["accuracy"]
    rng = np.random.default_rng(SEED)
    for s in strengths:
        off = rng.normal(0, 1, (len(X), len(CHANNELS))) * sig * s
        Xp = X + off[:, ch]
        acc = metrics(y, model.predict_proba(feats.transform(Xp))[:, 1])["accuracy"]
        rows.append(dict(strength=s, accuracy=acc, delta=(acc - base) * 100))
    return base, pd.DataFrame(rows)


def band_corr(cfg):
    """Correlation of each bin's depth with log CH4 and log O3, to choose windows
    from the data rather than from a textbook."""
    X, y, P = load_split("train", cfg, noisy=False)
    Z = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-30)
    out = {}
    for g in ("CH4", "O3"):
        v = P[f"atm {g}"].to_numpy()
        out[g] = np.array([np.corrcoef(Z[:, i], v)[0, 1] for i in range(Z.shape[1])])
    return centres(cfg), out


def floors(cfg, feats, model):
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, _ = load_split("train", cfg)
    tests = [load_split(t, cfg) for t in TESTS]
    res = {}
    # linear floor on the same normalised features
    f = Features("norm").fit(Xtr)
    lr = LogisticRegression(max_iter=4000, C=1.0).fit(f.transform(Xtr), ytr)
    res["logistic regression, normalized bins"] = [metrics(y, lr.predict_proba(f.transform(X))[:, 1])["accuracy"] for X, y, _ in tests]
    # data-chosen band index
    wl, cc = band_corr(cfg)
    def idx(g, k=6, sign=1):
        return np.argsort(sign * cc[g])[-k:]
    feat_tr, feat_te = [], []
    def make(X):
        Z = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-30)
        return np.column_stack([Z[:, idx("CH4")].mean(1) - Z[:, np.argsort(cc["CH4"])[:6]].mean(1),
                                Z[:, idx("O3")].mean(1) - Z[:, np.argsort(cc["O3"])[:6]].mean(1)])
    Dtr = make(Xtr)
    best, cut = -1, None
    q = np.linspace(0.02, 0.98, 60)
    g4, g3 = np.quantile(Dtr[:, 0], q), np.quantile(Dtr[:, 1], q)
    for a in g4:
        ma = Dtr[:, 0] > a
        for b in g3:
            acc = ((ma & (Dtr[:, 1] > b)).astype(int) == ytr).mean()
            if acc > best:
                best, cut = acc, (a, b)
    res["two-band index, data-chosen windows"] = [
        (((make(X)[:, 0] > cut[0]) & (make(X)[:, 1] > cut[1])).astype(int) == y).mean() for X, y, _ in tests]
    res["full pipeline"] = [metrics(y, model.predict_proba(feats.transform(X))[:, 1])["accuracy"] for X, y, _ in tests]
    return res, wl, cc, best


def realistic_subset(cfg, feats, model, depth_max=0.03, amp_max=3e-4):
    rows = []
    for t in TESTS:
        X, y, P = load_split(t, cfg)
        Xnf = np.load(os.path.join(DATA, f"{t}_ariel.npy")).astype(float)
        depth = Xnf.mean(axis=1); amp = Xnf.max(axis=1) - Xnf.min(axis=1)
        m = (depth < depth_max) & (amp < amp_max)
        if m.sum() < 50:
            continue
        p = model.predict_proba(feats.transform(X[m]))[:, 1]
        rows.append(dict(n=int(m.sum()), frac=float(m.mean()), accuracy=metrics(y[m], p)["accuracy"],
                         pos=float(y[m].mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ariel")
    a = ap.parse_args()
    cfg = a.config
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    feats, model = fr["features"], fr["model"]
    L = [f"Realism checks, configuration {cfg}, pipeline {best}", ""]

    print("per-channel offsets", flush=True)
    base, df = per_channel_offset(cfg, feats, model)
    L += ["1. Additive offsets drawn INDEPENDENTLY PER CHANNEL (three photometers and three",
          "   spectrometers with their own zero points), against the global offset in Table 3",
          f"   which costs exactly nothing by construction. Clean baseline {base*100:.2f}%.", "",
          f"   {'offset (x noise)':>17} {'accuracy':>10} {'delta':>8}"]
    for _, r in df.iterrows():
        L.append(f"   {r['strength']:>17.2f} {r['accuracy']*100:9.2f}% {r['delta']:+7.2f}")
    L.append("")

    print("floors", flush=True)
    res, wl, cc, tr_acc = floors(cfg, feats, model)
    L += ["2. Non-machine-learning and linear floors on the same five test sets.", "",
          f"   {'method':<40}{'accuracy':>10}"]
    for k, v in res.items():
        L.append(f"   {k:<40}{np.mean(v)*100:8.2f}%±{np.std(v, ddof=1)*100:.2f}")
    top4 = wl[np.argsort(cc['CH4'])[-4:]]; top3 = wl[np.argsort(cc['O3'])[-4:]]
    L += ["", f"   Bins most correlated with log CH4: {', '.join(f'{x:.2f}' for x in sorted(top4))} um"
              f" (max r = {cc['CH4'].max():.2f})",
          f"   Bins most correlated with log O3:  {', '.join(f'{x:.2f}' for x in sorted(top3))} um"
          f" (max r = {cc['O3'].max():.2f})", ""]

    print("realistic subset", flush=True)
    sub = realistic_subset(cfg, feats, model)
    L += ["3. The frozen pipeline on the subset with mean transit depth below 3 % and",
          "   peak-to-peak feature amplitude below 300 ppm, the range of real Ariel targets.", "",
          f"   planets retained: {sub['n'].sum()} of 9072 ({sub['frac'].mean()*100:.1f} %), positive rate {sub['pos'].mean():.3f}",
          f"   accuracy on that subset: {sub['accuracy'].mean()*100:.2f}% ± {sub['accuracy'].std(ddof=1)*100:.2f}",
          f"   (full test sets: {np.mean(res['full pipeline'])*100:.2f}%)", ""]
    open(os.path.join(RESULTS, f"{cfg}_realism.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
