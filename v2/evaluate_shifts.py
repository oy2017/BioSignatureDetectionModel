"""Stage 5: score the frozen best pipeline on every shift axis, paired per planet.

Two kinds of axis:
  * re-rendered: a file data/{split}_native_{axis}.npy exists for each test
    split (Exo-Transmit, ExoMol / +HITRAN O3, cloud decks, hazes, stellar
    contamination). It is binned to the configuration and given the SAME noise
    realisation seed as the clean test set, so the only difference from the
    clean spectrum of the same planet is the physics.
  * injected: a perturbation applied to the clean binned spectra (white noise to
    a target effective SNR; time-correlated noise; gain ramp; baseline offset;
    Ariel-coloured vs white noise at matched sigma).
For each case: accuracy, Brier, ECE, predicted-positive rate, flips relative
to the clean prediction (clean-right/shifted-wrong and the reverse), and the
median surviving feature amplitude relative to clean.

Also: out-of-envelope extrapolation (radius split, size-matched control),
which retrains by construction.

Usage: python evaluate_shifts.py --config ariel [--axes ...] [--no-injected] [--no-extrapolation]
Writes results/{config}_shifts.txt and results/{config}_shifts.csv
"""
import argparse
import glob
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from bin_spectra import bin_native
from common import (DATA, MODELS, RESULTS, SEED, SNR, TESTS, base_config, centres,
                    configs, load_split, metrics, noise_spec)
from noise import add_noise, add_correlated_noise, sigma_matrix
from pipeline import make_rf, make_xgb

SWEEP_SNR = [15, 12, 10, 8, 5]
SWEEP_SYS = [0.25, 0.5, 1.0, 2.0]


def noise_floor(Xnf, params, cen, snr=SNR, shape="ariel", level_ppm=None):
    """The per-spectrum noise level actually used to observe these planets.

    An earlier version estimated this from the scatter of adjacent bins, which
    assumes a locally flat spectrum. The Ariel layout is not flat: it has three
    photometric bands and hard resolution jumps at the channel boundaries, and
    that estimator came out 1.46x too large, so every injected level was
    mislabelled by that factor. The generating sigma is available directly."""
    from noise import sigma_matrix
    return np.median(sigma_matrix(Xnf, params["s temperature"].to_numpy(), cen, snr, shape, level_ppm),
                     axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ariel")
    ap.add_argument("--axes", nargs="*", default=None)
    ap.add_argument("--no-injected", action="store_true")
    ap.add_argument("--no-extrapolation", action="store_true")
    ap.add_argument("--model", default=None,
                    help="pipeline key to score (default: the config's best). Use to test "
                         "whether the fidelity ordering is a property of the task or of the model.")
    ap.add_argument("--tag", default=None, help="suffix for the output files")
    a = ap.parse_args()
    cfg = a.config
    best = a.model or json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    feats = fr["features"]
    if best.endswith("_mlp"):
        import tensorflow as tf
        _k = tf.keras.models.load_model(fr["keras_path"])
        class _W:
            def predict_proba(self, Z):
                q = _k.predict(Z, verbose=0).ravel()
                return np.c_[1 - q, q]
        model = _W()
    else:
        model = fr["model"]
    tag = a.tag or ("" if a.model is None else "_" + best)
    edges = np.array(configs()[base_config(cfg)]["edges"])
    cen = centres(cfg)
    nshape, nlevel = noise_spec(cfg)
    wl = np.load(os.path.join(DATA, "native_wl.npy"))

    def predict(X):
        return model.predict_proba(feats.transform(X))[:, 1]

    # clean, paired baseline: same noise seeds as load_split
    clean = []
    for t in TESTS:
        X, y, P = load_split(t, cfg)
        Xnf = np.load(os.path.join(DATA, f"{t}_{base_config(cfg)}.npy")).astype(float)
        clean.append((X, y, P, Xnf))
    ypool = np.concatenate([c[1] for c in clean])
    Ppool = pd.concat([c[2] for c in clean], ignore_index=True)
    Xpool = np.vstack([c[0] for c in clean])
    Xnf_pool = np.vstack([c[3] for c in clean])
    p0 = predict(Xpool); pred0 = (p0 >= 0.5).astype(int)
    m0 = metrics(ypool, p0)
    amp0 = np.median(Xnf_pool.std(1))
    rows = [dict(axis="clean", case="reference", n=len(ypool), **m0, pos_rate=pred0.mean(),
                 flips_bad=0, flips_good=0, amp_ratio=1.0)]

    def record(axis, case, Xs, valid=None, amp_of=None):
        yv, pv0 = ypool, pred0
        Xv = Xs
        if valid is not None:
            Xv, yv, pv0 = Xs[valid], ypool[valid], pred0[valid]
        if amp_of is not None and valid is not None:
            amp_of = amp_of[valid]
        p = predict(Xv); pred = (p >= 0.5).astype(int)
        m = metrics(yv, p)
        rows.append(dict(axis=axis, case=case, n=len(yv), **m, pos_rate=pred.mean(),
                         flips_bad=int(((pv0 == yv) & (pred != yv)).sum()),
                         flips_good=int(((pv0 != yv) & (pred == yv)).sum()),
                         amp_ratio=float(np.median(amp_of.std(1)) / amp0) if amp_of is not None else np.nan))
        print(f"{axis:14s} {case:22s} acc {m['accuracy']*100:6.2f} ({(m['accuracy']-m0['accuracy'])*100:+6.2f}) "
              f"brier {m['brier']:.4f} pos {pred.mean():.3f} amp {rows[-1]['amp_ratio']:.3f} n={len(yv)}", flush=True)

    # ---- re-rendered axes: any data/test1_native_<axis>.npy present
    axes = a.axes
    if axes is None:
        axes = sorted({re.sub(r"^test1_native_", "", os.path.basename(f))[:-4]
                       for f in glob.glob(os.path.join(DATA, "test1_native_*.npy"))})
    for ax in axes:
        parts = []
        ok = True
        for t in TESTS:
            f = os.path.join(DATA, f"{t}_native_{ax}.npy")
            if not os.path.exists(f):
                ok = False; break
            parts.append(np.load(f).astype(np.float64))
        if not ok:
            print(f"skip {ax}: missing splits"); continue
        Xn = np.vstack(parts)
        valid = np.all(np.isfinite(Xn), axis=1)
        Xb = np.full((Xn.shape[0], len(cen)), np.nan)
        Xb[valid] = bin_native(Xn[valid], wl, edges)
        # Same noise realisation as the clean set of the same planet. Two points:
        # (1) the noise is drawn for the FULL array before masking, because a fixed
        #     seed on a different array shape gives a different realisation and would
        #     silently unpair the axes that lost a planet to a failed re-render;
        # (2) sigma is computed from the CLEAN spectrum, so a cloud that suppresses
        #     features is not also rewarded with less noise. Observing conditions are
        #     set by the star and the instrument, not by the planet's atmosphere.
        Xs = np.empty_like(Xb)
        i = 0
        for k, t in enumerate(TESTS):
            n = len(clean[k][1]); sl = slice(i, i + n)
            Pk = clean[k][2]
            sig = sigma_matrix(clean[k][3], Pk["s temperature"].to_numpy(), cen, SNR, nshape, nlevel)
            rng_k = np.random.default_rng(2000 + k + 1)
            filled = np.where(np.isfinite(Xb[sl]), Xb[sl], clean[k][3])
            Xs[sl] = filled + rng_k.normal(0.0, 1.0, sig.shape) * sig
            i += n
        family = ax.split("_")[0]
        # amplitude is always measured on the noise-free spectrum, so the column
        # means the same thing in every row (an injected axis leaves it at 1.00)
        record(family, ax, Xs, valid=valid if not valid.all() else None, amp_of=Xb)

    # ---- injected families on the clean noisy spectra
    if not a.no_injected:
        rng = np.random.default_rng(SEED)
        sig_n = noise_floor(Xnf_pool, Ppool, cen, SNR, nshape, nlevel)
        for s in SWEEP_SNR[1:]:
            m = np.sqrt((SNR / s) ** 2 - 1.0)
            record("white noise", f"snr{s}", Xpool + rng.normal(0, 1, Xpool.shape) * sig_n * m, amp_of=Xnf_pool)
        for s in SWEEP_SNR[1:]:
            m = np.sqrt((SNR / s) ** 2 - 1.0)
            raw = rng.normal(0, 1, Xpool.shape)
            sm = gaussian_filter1d(raw, sigma=3.0, axis=1)
            sm /= sm.std(axis=1, keepdims=True) + 1e-12
            record("correlated noise", f"snr{s}", Xpool + sm * sig_n * m, amp_of=Xnf_pool)
        for s in SWEEP_SYS:
            amp = s * sig_n / np.abs(Xpool).mean(axis=1, keepdims=True)
            tilt = np.linspace(-1, 1, Xpool.shape[1]); signs = rng.choice([-1.0, 1.0], size=(len(Xpool), 1))
            record("gain ramp", f"x{s}", Xpool * (1 + signs * amp * tilt), amp_of=Xnf_pool * (1 + signs * amp * tilt))
        for s in SWEEP_SYS:
            record("baseline offset", f"x{s}", Xpool + rng.normal(0, 1, (len(Xpool), 1)) * sig_n * s, amp_of=Xnf_pool)
        # absolute noise floor (realistic convention): Ariel-shaped sigma with a fixed median level
        for lvl in (20, 50, 100, 200):
            Xa, _ = add_noise(Xnf_pool, Ppool, cen, shape="ariel_abs", level_ppm=lvl, seed=900 + lvl)
            record("absolute noise", f"{lvl} ppm", Xa, amp_of=Xnf_pool)
        # noise colouring at matched median sigma: white vs Ariel-shaped, from the noise-free spectra
        for s in (15, 10, 7, 5):
            Xw, _ = add_noise(Xnf_pool, Ppool, cen, snr=s, shape="white", seed=777 + s)
            Xa, _ = add_noise(Xnf_pool, Ppool, cen, snr=s, shape="ariel", seed=777 + s)
            record("noise colouring", f"white snr{s}", Xw, amp_of=Xnf_pool)
            record("noise colouring", f"ariel snr{s}", Xa, amp_of=Xnf_pool)

    # ---- extrapolation (retrain by construction)
    lines = []
    if not a.no_extrapolation:
        from common import Features
        Xtr, ytr, Ptr = load_split("train", cfg)
        r = Ptr["p_radius"].to_numpy(); cut = 15.0
        tr, te = r <= cut, r > cut
        kind, mname = best.split("_"); make = make_xgb if mname == "xgb" else make_rf
        params = fr["params"]

        def fit_eval(Xa, ya, Xb, yb):
            f = Features(kind, n_components=getattr(feats, "k", None)).fit(Xa)
            mm = make(params).fit(f.transform(Xa), ya)
            return metrics(yb, mm.predict_proba(f.transform(Xb))[:, 1])

        ext = fit_eval(Xtr[tr], ytr[tr], Xtr[te], ytr[te])
        # The control must change ONLY the training distribution. An earlier version
        # also resampled the test planets, which meant it compared two different tasks:
        # large planets have deeper envelopes and are intrinsically easier, so the
        # extrapolation penalty was masked by an easier control test set.
        rng = np.random.default_rng(SEED); ctl = []
        pool = np.where(tr)[0]
        for rep in range(5):
            s_tr = rng.choice(len(ytr), int(tr.sum()), replace=False)
            ctl.append(fit_eval(Xtr[s_tr], ytr[s_tr], Xtr[te], ytr[te]))
        a_ctl = np.mean([c["accuracy"] for c in ctl]); b_ctl = np.mean([c["brier"] for c in ctl])
        lines += ["", f"Out-of-envelope extrapolation, radius split at {cut} R_earth (train n={tr.sum()}, test n={te.sum()})",
                  "  The control trains on a random subset of the SAME size drawn from the whole grid",
                  "  and is tested on the SAME large planets, so only the training distribution differs.",
                  f"  extrapolation: acc {ext['accuracy']*100:.2f}%  Brier {ext['brier']:.4f}",
                  f"  control (random training draw, same n, same test planets, mean of 5): acc {a_ctl*100:.2f}%  Brier {b_ctl:.4f}",
                  f"  penalty net of sample size: {(ext['accuracy']-a_ctl)*100:+.2f} points, Brier {ext['brier']-b_ctl:+.4f}"]
        rows.append(dict(axis="extrapolation", case="radius>15 vs control", n=int(te.sum()), **ext,
                         pos_rate=np.nan, flips_bad=0, flips_good=0, amp_ratio=np.nan,
                         control_accuracy=a_ctl, control_brier=b_ctl))

    df = pd.DataFrame(rows)
    df["d_acc"] = (df["accuracy"] - m0["accuracy"]) * 100
    df["d_brier"] = df["brier"] - m0["brier"]
    df.to_csv(os.path.join(RESULTS, f"{cfg}{tag}_shifts.csv"), index=False)
    hdr = [f"Domain shift, configuration {cfg}, frozen pipeline {best}; clean paired baseline "
           f"{m0['accuracy']*100:.2f}% (Brier {m0['brier']:.4f}, n={len(ypool)})", "",
           f"{'axis':16s} {'case':22s} {'acc':>7s} {'d_acc':>7s} {'brier':>7s} {'ece':>6s} {'pos':>6s} {'flip-':>6s} {'flip+':>6s} {'amp':>6s}"]
    for _, r in df.iterrows():
        hdr.append(f"{r['axis']:16s} {r['case']:22s} {r['accuracy']*100:7.2f} {r['d_acc']:+7.2f} {r['brier']:7.4f} "
                   f"{r['ece']:6.3f} {r['pos_rate']:6.3f} {int(r['flips_bad']):6d} {int(r['flips_good']):6d} {r['amp_ratio']:6.3f}")
    with open(os.path.join(RESULTS, f"{cfg}{tag}_shifts.txt"), "w") as f:
        f.write("\n".join(hdr + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
