"""Compound mismatch: do single-axis costs add, or worse?

Real planets are off on every axis at once; the budget prices one axis at a time. The
compound test planets are built from renders that already exist: the stellar-contamination
factor (spots render / clean render, a multiplicative term in the forward model) applied to
the haze render of the same planet, then noised at a reduced effective SNR. Scores the
frozen clean-trained screen and reports each compound loss against the sum of its parts.

Usage: python compound.py --config ariel
Writes data/test*_native_compound_<tag>.npy (so trust_randomized.py can reuse them) and
results/ariel_compound.txt / .csv
"""
import argparse, json, os, sys
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, centres, load_split, metrics  # noqa: E402
from augment import binned, shifted_test, corr_noise  # noqa: E402

COMPOUNDS = [  # tag, spots case, haze case, effective SNR at test (None = the training SNR)
    ("spots10_haze3e7",       "tlse_spots10", "haze_3p0e7", None),
    ("spots20_haze3e7",       "tlse_spots20", "haze_3p0e7", None),
    ("spots10_haze3e7_snr8",  "tlse_spots10", "haze_3p0e7", 8),
    ("spots20_haze3e7_snr8",  "tlse_spots20", "haze_3p0e7", 8),
    ("spots10_haze2e6_snr10", "tlse_spots10", "haze_2p0e6", 10),   # the "everything mild" case
]


def build(tag, spots, haze):
    """Native compound render per split: haze render x (spots render / clean render)."""
    for t in TESTS:
        out = os.path.join(DATA, f"{t}_native_compound_{tag}.npy")
        if os.path.exists(out):
            continue
        clean = np.load(os.path.join(DATA, f"{t}_native.npy")).astype(np.float64)
        sp = np.load(os.path.join(DATA, f"{t}_native_{spots}.npy")).astype(np.float64)
        hz = np.load(os.path.join(DATA, f"{t}_native_{haze}.npy")).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            fac = np.where(clean != 0, sp / clean, 1.0)
        X = hz * fac
        np.save(out, X.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel"); a = ap.parse_args(); cfg = a.config
    cen = centres(cfg)
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib")); ff, fm = fr["features"], fr["model"]
    acc = lambda X, y: metrics(y, fm.predict_proba(ff.transform(X))[:, 1])["accuracy"]

    Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
    Xc_nf = np.vstack([np.load(os.path.join(DATA, f"{t}_{cfg}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    clean = acc(Xc, yc)

    # single-axis costs on the same planets, same construction as evaluate_shifts / augment
    single = {}
    for case in ("tlse_spots10", "tlse_spots20", "haze_3p0e7", "haze_2p0e6"):
        Xs, ys = shifted_test(case, cfg); single[case] = clean - acc(Xs, ys)
    rng = np.random.default_rng(SEED + 51)
    for s in (10, 8):
        single[f"snr{s}"] = clean - acc(corr_noise(Xc, rng, s, Xnf=Xc_nf, params=Pc, cen=cen), yc)

    rows, L = [], [f"Compound mismatch on the frozen clean-trained screen, configuration {cfg}, pipeline {best}; clean {clean*100:.2f}%", "",
                   f"{'compound':<24}{'accuracy':>10}{'loss':>8}{'sum of parts':>14}{'excess':>9}"]
    for tag, spots, haze, snr in COMPOUNDS:
        build(tag, spots, haze)
        Xs, ys = shifted_test(f"compound_{tag}", cfg)
        if snr is not None:
            rng = np.random.default_rng(SEED + 52)
            Xs = corr_noise(Xs, rng, snr, Xnf=Xc_nf, params=Pc, cen=cen)
        a = acc(Xs, ys); loss = clean - a
        parts = single[spots] + single[haze] + (single[f"snr{snr}"] if snr else 0.0)
        L.append(f"{tag:<24}{a*100:9.2f}%{loss*100:8.2f}{parts*100:14.2f}{(loss-parts)*100:+9.2f}")
        rows.append(dict(compound=tag, accuracy=a, loss=loss, sum_of_parts=parts, excess=loss - parts))
    L += ["", "single-axis losses (points): " + ", ".join(f"{k} {v*100:.2f}" for k, v in single.items()),
          "", "excess > 0: the axes compound super-additively on this screen; excess < 0: they partly mask each other."]
    open(os.path.join(RESULTS, f"{cfg}_compound.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"{cfg}_compound.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
