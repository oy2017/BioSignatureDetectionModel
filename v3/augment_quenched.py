"""Axis 8 repairability: does training on quenched spectra recover the quenched loss?

The pre-registered prediction (v3/PREREGISTERED_PREDICTIONS.md, "Axis 8") is that this
axis is REPAIRABLE (> 60 % of the gap), because a quenched re-render is a deterministic
function of the planet -- zero random draws per spectrum -- even though it changes which
molecules are present rather than distorting a fixed spectrum. This script is the test.

Mirrors augment.py: the augmented training set gives each planet one version -- a third
left at equilibrium, two thirds quenched (the v2 "a third clean" convention) -- binned
and noised with the same seed the frozen pipeline's training used; the frozen and
augmented pipelines are then scored on the identical quenched test spectra, built the way
evaluate_shifts builds a re-rendered case. The recovered fraction is measured against
the frozen pipeline's own clean accuracy, and reported per temperature band because the
chemistry predicts the cost to sit below ~1000 K.

Needs: data/train_native_quenched.npy (generate_grid.py --mode quenched --splits train)
       data/test*_native_quenched.npy, results/ariel_best.json, models/ariel_<best>.joblib
Usage: python augment_quenched.py --config ariel
Writes results/ariel_augment_quenched.txt / .csv
"""
import argparse, json, os, sys
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features, centres, load_split, metrics  # noqa: E402
from noise import add_noise  # noqa: E402
from pipeline import make_xgb  # noqa: E402
from augment import binned, shifted_test  # noqa: E402

T_BANDS = [(500, 1000), (1000, 1500), (1500, 2500)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel"); a = ap.parse_args(); cfg = a.config
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    ff, fm, params = fr["features"], fr["model"], fr["params"]; kind = best.split("_")[0]
    Ptr = pd.read_parquet(os.path.join(DATA, "train_params.parquet")); ytr = Ptr["label_co"].to_numpy()
    cen = centres(cfg)

    # augmented training set: one version per planet, a third equilibrium, two thirds quenched
    Xeq = np.load(os.path.join(DATA, "train_native.npy")).astype(float)
    Xq = np.load(os.path.join(DATA, "train_native_quenched.npy")).astype(float)
    assert Xeq.shape == Xq.shape, "quenched train render is not aligned with the equilibrium one"
    rng = np.random.default_rng(SEED + 21)
    use_q = rng.random(len(ytr)) < 2 / 3
    ok_q = np.all(np.isfinite(Xq), axis=1)
    Xaug = np.where((use_q & ok_q)[:, None], Xq, Xeq)
    Xb = binned(Xaug, cfg)
    Xn, _ = add_noise(Xb, Ptr, cen, snr=SNR, shape="ariel", seed=1000)
    f_aug = Features(kind).fit(Xn); m_aug = make_xgb(params).fit(f_aug.transform(Xn), ytr)

    # clean reference and the quenched test case
    Xc, yc, _ = load_split("test1", cfg)
    for t in TESTS[1:]:
        X2, y2, _ = load_split(t, cfg); Xc = np.vstack([Xc, X2]); yc = np.concatenate([yc, y2])
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    Tc = Pc["atm temperature"].to_numpy()
    clean_ref = metrics(yc, fm.predict_proba(ff.transform(Xc))[:, 1])["accuracy"]
    Xs, ys = shifted_test("quenched", cfg)
    assert len(ys) == len(yc)
    p_fr = fm.predict_proba(ff.transform(Xs))[:, 1]; p_au = m_aug.predict_proba(f_aug.transform(Xs))[:, 1]
    c_fr = ((p_fr >= 0.5).astype(int) == ys); c_au = ((p_au >= 0.5).astype(int) == ys)
    c_cl = ((fm.predict_proba(ff.transform(Xc))[:, 1] >= 0.5).astype(int) == yc)

    L = [f"Axis 8 (quenched composition): recovery by training on quenched spectra, configuration {cfg}, pipeline {best}",
         f"Frozen clean reference {clean_ref*100:.2f}%. Augmented: {use_q.mean()*100:.0f}% of training planets quenched, rest equilibrium.", "",
         f"{'subset':<16}{'n':>6}{'clean':>8}{'frozen':>9}{'augmented':>11}{'gain':>8}{'% of gap':>10}"]
    rows = []
    def line(name, m):
        a_cl, a_fr, a_au = c_cl[m].mean(), c_fr[m].mean(), c_au[m].mean()
        frac = (a_au - a_fr) / max(a_cl - a_fr, 1e-9) * 100
        L.append(f"{name:<16}{m.sum():>6}{a_cl*100:7.2f}%{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}{frac:9.0f}%")
        rows.append(dict(subset=name, n=int(m.sum()), clean=a_cl, frozen=a_fr, augmented=a_au, pct_of_gap=frac))
    line("all planets", np.ones(len(ys), bool))
    for lo, hi in T_BANDS:
        line(f"T {lo}-{hi} K", (Tc >= lo) & (Tc < hi))
    a_cl_au = metrics(yc, m_aug.predict_proba(f_aug.transform(Xc))[:, 1])["accuracy"]
    L += ["", f"cost on clean data of training with the shift: {clean_ref*100:.2f}% -> {a_cl_au*100:.2f}% ({(a_cl_au-clean_ref)*100:+.2f})",
          "", "Pre-registered prediction: > 60 % recovered (deterministic re-render, zero draws per spectrum).",
          "The rule is falsified on this axis if the all-planets recovery is below 40 %."]
    open(os.path.join(RESULTS, f"{cfg}_augment_quenched.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"{cfg}_augment_quenched.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
