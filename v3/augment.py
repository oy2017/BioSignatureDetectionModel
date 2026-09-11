"""Can the domain-shift losses be repaired by training on the shifted physics?

For each of the three most expensive axes, build an AUGMENTED training set in
which every planet is independently assigned a random strength of that shift
(including "none" for a third of the planets, so the clean regime is retained),
retrain the primary pipeline on it, and score it on the same shifted test sets
the frozen pipeline was scored on. The comparison is like-for-like: identical
planets, identical noise seeds, only the training distribution differs.

  spots       stellar contamination, coverage drawn from {0, 2, 5, 10, 20} %
  haze        Lee-Mie haze, density drawn from {0, 2e5, 2e6, 3e7, 2.4e8} m^-3
  correlated  time-correlated noise at effective SNR drawn from {15, 12, 10, 8}

Stage 1 (--render) writes the augmented native training spectra; the haze pass
re-runs the forward model and takes ~15 min, the spots pass is a multiplication
and takes seconds. Stage 2 (--fit) retrains and evaluates.

Usage:
    python augment.py --render          # build train_native_aug_{spots,haze}.npy
    python augment.py --fit             # retrain + evaluate, write results/ariel_augment.txt
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

from bin_spectra import bin_native  # noqa: E402
from common import (DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features, centres,  # noqa: E402
                    configs, load_split, metrics)
from noise import add_noise  # noqa: E402
from pipeline import make_xgb  # noqa: E402

SPOT_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]
HAZE_LEVELS = [0.0, 2e5, 2e6, 3e7, 2.4e8]
CORR_SNR = [15, 12, 10, 8]
# test cases each augmentation is scored against
SPOT_CASES = ["tlse_spots02", "tlse_spots05", "tlse_spots10", "tlse_spots20"]
HAZE_CASES = ["haze_2p0e5", "haze_2p0e6", "haze_3p0e7", "haze_2p4e8"]


def render_spots():
    """Contamination is a multiplicative factor: apply per planet at a random level."""
    import shift_tlse as T
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    T.set_grid(wl)
    P = pd.read_parquet(os.path.join(DATA, "train_params.parquet"))
    X = np.load(os.path.join(DATA, "train_native.npy")).astype(np.float64)
    rng = np.random.default_rng(SEED)
    lev = rng.choice(SPOT_LEVELS, size=len(P))
    Tst = P["s temperature"].to_numpy()
    logg = np.log10(T.G_SUN * P["s mass"].to_numpy() / P["s radius"].to_numpy() ** 2 * 100)
    out = np.empty_like(X, dtype=np.float32)
    for f in SPOT_LEVELS:
        m = lev == f
        out[m] = (X[m] * (1.0 if f == 0 else T.contamination(Tst[m], logg[m], f, 0.0))).astype(np.float32)
        print(f"  spots {f:.2f}: {m.sum()} planets", flush=True)
    np.save(os.path.join(DATA, "train_native_aug_spots.npy"), out)
    np.save(os.path.join(DATA, "train_aug_spots_levels.npy"), lev)


def render_haze(jobs):
    """Haze needs the forward model: one pass, random density per planet."""
    import shift_aerosol as A
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    P = pd.read_parquet(os.path.join(DATA, "train_params.parquet"))
    X = np.load(os.path.join(DATA, "train_native.npy")).astype(np.float32)
    rng = np.random.default_rng(SEED + 1)
    lev = rng.choice(HAZE_LEVELS, size=len(P))
    out = X.copy()
    for h in HAZE_LEVELS:
        if h == 0.0:
            continue
        m = np.where(lev == h)[0]
        sub = P.iloc[m].reset_index(drop=True)
        Y = A.render(sub, wl, jobs=jobs, haze_density=h)
        ok = np.all(np.isfinite(Y), axis=1)
        out[m[ok]] = Y[ok].astype(np.float32)
        print(f"  haze {h:.1e}: {len(m)} planets, {int((~ok).sum())} failures", flush=True)
    np.save(os.path.join(DATA, "train_native_aug_haze.npy"), out)
    np.save(os.path.join(DATA, "train_aug_haze_levels.npy"), lev)


def binned(path_or_arr, cfg):
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    edges = np.array(configs()[cfg]["edges"])
    X = np.load(path_or_arr) if isinstance(path_or_arr, str) else path_or_arr
    return bin_native(np.asarray(X, dtype=np.float64), wl, edges)


def shifted_test(case, cfg):
    """Pooled shifted test spectra, noised exactly as evaluate_shifts.py does:
    the same seed as the clean set of the same planet, sigma taken from the CLEAN
    spectrum so a suppressed atmosphere is not also given less noise, and the
    configuration's own noise convention rather than a hardcoded one."""
    from common import noise_spec, base_config
    from noise import sigma_matrix
    cen = centres(cfg); nshape, nlevel = noise_spec(cfg)
    Xs, ys = [], []
    for k, t in enumerate(TESTS):
        Xn = np.load(os.path.join(DATA, f"{t}_native_{case}.npy")).astype(np.float64)
        _, y, P = load_split(t, cfg, noisy=False)
        clean_b = np.load(os.path.join(DATA, f"{t}_{base_config(cfg)}.npy")).astype(np.float64)
        Xb = binned(Xn, cfg)
        Xb = np.where(np.isfinite(Xb), Xb, clean_b)          # failed re-renders fall back to clean
        sig = sigma_matrix(clean_b, P["s temperature"].to_numpy(), cen, SNR, nshape, nlevel)
        rng = np.random.default_rng(2000 + k + 1)
        Xs.append(Xb + rng.normal(0.0, 1.0, sig.shape) * sig); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def corr_noise(X, rng, snr_eff, kind="correlated", Xnf=None, params=None, cen=None):
    """Inject noise to a target effective SNR. sigma is the GENERATING sigma of
    the clean spectra, not an adjacent-difference estimate of it: on the Ariel
    layout, whose bins jump at channel boundaries, that estimator runs 1.46x high
    and mislabels every level (evaluate_shifts.py carries the same fix)."""
    from scipy.ndimage import gaussian_filter1d
    from noise import sigma_matrix
    if Xnf is not None:
        sig = np.median(sigma_matrix(Xnf, params["s temperature"].to_numpy(), cen), axis=1, keepdims=True)
    else:
        sig = np.diff(X, axis=1).std(axis=1, keepdims=True) / np.sqrt(2.0)
    m = np.sqrt((SNR / snr_eff) ** 2 - 1.0) if snr_eff < SNR else 0.0
    if m == 0:
        return X.copy()
    if kind == "white":
        return X + rng.normal(0, 1, X.shape) * sig * m
    z = gaussian_filter1d(rng.normal(0, 1, X.shape), sigma=3.0, axis=1)
    z /= z.std(axis=1, keepdims=True) + 1e-12
    return X + z * sig * m


def fit(cfg="ariel"):
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    frozen_feats, frozen_model, params = fr["features"], fr["model"], fr["params"]
    kind = best.split("_")[0]
    ytr = pd.read_parquet(os.path.join(DATA, "train_params.parquet"))["label_co"].to_numpy()
    Xtr_clean, _, _ = load_split("train", cfg)
    cen = centres(cfg)
    Ptr = pd.read_parquet(os.path.join(DATA, "train_params.parquet"))
    # v3: the clean reference is the frozen pipeline's own accuracy on the pooled clean test sets,
    # not a number carried over from another grid
    _Xc, _yc, _ = load_split("test1", cfg)
    for _t in TESTS[1:]:
        _X2, _y2, _ = load_split(_t, cfg); _Xc = np.vstack([_Xc, _X2]); _yc = np.concatenate([_yc, _y2])
    clean_ref = metrics(_yc, frozen_model.predict_proba(frozen_feats.transform(_Xc))[:, 1])["accuracy"]
    print(f"clean reference (frozen pipeline, pooled tests): {clean_ref*100:.2f}%", flush=True)
    lines = [f"Recovery by training on the shifted physics, configuration {cfg}, pipeline {best}",
             "Frozen: trained on clean spectra only (the paper's primary pipeline).",
             "Augmented: retrained on a training set where each planet carries a random strength",
             "of that axis (a third of planets left clean). Identical test planets and noise seeds.", ""]
    rows = []

    def train_on(Xaug_native):
        Xb = binned(Xaug_native, cfg)
        Xn, _ = add_noise(Xb, Ptr, cen, snr=SNR, shape="ariel", seed=1000)
        f = Features(kind).fit(Xn)
        m = make_xgb(params).fit(f.transform(Xn), ytr)
        return f, m

    # ---- spots and haze: re-rendered augmentation
    for axis, fname, cases in (("stellar spots", "train_native_aug_spots.npy", SPOT_CASES),
                               ("haze", "train_native_aug_haze.npy", HAZE_CASES)):
        p = os.path.join(DATA, fname)
        if not os.path.exists(p):
            print(f"skip {axis}: {fname} missing"); continue
        f_aug, m_aug = train_on(np.load(p))
        lines.append(f"{axis}:")
        lines.append(f"{'case':<16}{'frozen':>9}{'augmented':>11}{'gain':>8}{'% of gap':>10}")
        for case in cases:
            Xs, ys = shifted_test(case, cfg)
            a_fr = metrics(ys, frozen_model.predict_proba(frozen_feats.transform(Xs))[:, 1])["accuracy"]
            a_au = metrics(ys, m_aug.predict_proba(f_aug.transform(Xs))[:, 1])["accuracy"]
            frac = (a_au - a_fr) / max(clean_ref - a_fr, 1e-9) * 100
            lines.append(f"{case:<16}{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}{frac:9.0f}%")
            rows.append(dict(axis=axis, case=case, frozen=a_fr, augmented=a_au, pct_of_gap=frac))
        # cost on clean data of having trained with the shift
        Xc, yc, _ = load_split("test1", cfg)
        for t in TESTS[1:]:
            X2, y2, _ = load_split(t, cfg); Xc = np.vstack([Xc, X2]); yc = np.concatenate([yc, y2])
        a_fr = metrics(yc, frozen_model.predict_proba(frozen_feats.transform(Xc))[:, 1])["accuracy"]
        a_au = metrics(yc, m_aug.predict_proba(f_aug.transform(Xc))[:, 1])["accuracy"]
        lines.append(f"{'clean':<16}{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}{'':>10}")
        rows.append(dict(axis=axis, case="clean", frozen=a_fr, augmented=a_au))
        lines.append("")

    # ---- correlated noise: injected augmentation, no re-render
    Xtr_nf = np.load(os.path.join(DATA, f"train_{cfg}.npy")).astype(float)
    rng = np.random.default_rng(SEED + 2)
    lev = rng.choice(CORR_SNR, size=len(ytr))
    Xa = Xtr_clean.copy()
    for s in CORR_SNR:
        m = lev == s
        if s < SNR:
            Xa[m] = corr_noise(Xtr_clean[m], rng, s, Xnf=Xtr_nf[m], params=Ptr[m].reset_index(drop=True), cen=cen)
    f_aug = Features(kind).fit(Xa)
    m_aug = make_xgb(params).fit(f_aug.transform(Xa), ytr)
    lines.append("time-correlated noise:")
    lines.append(f"{'case':<16}{'frozen':>9}{'augmented':>11}{'gain':>8}{'% of gap':>10}")
    Xc, yc, _ = load_split("test1", cfg)
    for t in TESTS[1:]:
        X2, y2, _ = load_split(t, cfg); Xc = np.vstack([Xc, X2]); yc = np.concatenate([yc, y2])
    Xc_nf = np.vstack([np.load(os.path.join(DATA, f"{t}_{cfg}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    rng2 = np.random.default_rng(SEED + 3)
    for s in (12, 10, 8, 5):
        Xs = corr_noise(Xc, rng2, s, Xnf=Xc_nf, params=Pc, cen=cen)
        a_fr = metrics(yc, frozen_model.predict_proba(frozen_feats.transform(Xs))[:, 1])["accuracy"]
        a_au = metrics(yc, m_aug.predict_proba(f_aug.transform(Xs))[:, 1])["accuracy"]
        lines.append(f"{'SNR '+str(s):<16}{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}"
                     f"{(a_au-a_fr)/max(clean_ref-a_fr,1e-9)*100:9.0f}%")
        rows.append(dict(axis="correlated noise", case=f"snr{s}", frozen=a_fr, augmented=a_au))
    a_fr = metrics(yc, frozen_model.predict_proba(frozen_feats.transform(Xc))[:, 1])["accuracy"]
    a_au = metrics(yc, m_aug.predict_proba(f_aug.transform(Xc))[:, 1])["accuracy"]
    lines.append(f"{'clean':<16}{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}{'':>10}")
    rows.append(dict(axis="correlated noise", case="clean", frozen=a_fr, augmented=a_au))

    with open(os.path.join(RESULTS, "ariel_augment.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "ariel_augment.csv"), index=False)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--config", default="ariel")
    a = ap.parse_args()
    if a.render:
        print("spots:"); render_spots()
        print("haze:"); render_haze(a.jobs)
    if a.fit:
        fit(a.config)


if __name__ == "__main__":
    main()
