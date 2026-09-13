"""Two referee-proofing baselines for the trust envelope (stress-test items S12 and 'calibration').

1. Split-conformal prediction: calibrate a threshold on clean held-out scores so that 90 % of
   clean planets receive a prediction set containing the true label; under each mismatch report
   the realised coverage (should be 90 % if the guarantee held) and the share of planets given an
   ambiguous set {0,1} (the conformal way of declining). Shows what the standard guarantee is
   worth when the deployment distribution is not the calibration one.
2. Calibration under mismatch: Brier score and expected calibration error of the frozen screen's
   probabilities per case — does a 0.9 still mean 90 %?

Usage: python conformal_calibration.py --config ariel
Writes results/ariel_conformal_calibration.txt / .csv
"""
import argparse, json, os, sys
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, TESTS, centres, load_split  # noqa: E402
from augment import shifted_test, corr_noise  # noqa: E402

CASES = ["cloud_1e3Pa", "haze_3p0e7", "haze_2p4e8", "tlse_spots10", "tlse_spots20", "exotransmit", "exomol", "quenched",
         "compound_spots20_haze3e7", "absorbers", "absorbers_quenched"]


def ece(p, y, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel"); a = ap.parse_args(); cfg = a.config
    cen = centres(cfg)
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib")); ff, fm = fr["features"], fr["model"]
    proba = lambda X: fm.predict_proba(ff.transform(X))[:, 1]
    Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
    Xc_nf = np.vstack([np.load(os.path.join(DATA, f"{t}_{cfg}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    # split conformal: calibrate on test1+test2 (clean), evaluate on test3-5 and their shifted versions
    n12 = sum(len(load_split(t, cfg)[1]) for t in TESTS[:2]); cal, ev = slice(0, n12), slice(n12, None)
    pc = proba(Xc); s_cal = 1 - np.where(yc[cal] == 1, pc[cal], 1 - pc[cal])          # nonconformity: 1 - p(true class)
    q = np.quantile(s_cal, np.ceil(0.9 * (len(s_cal) + 1)) / len(s_cal))
    def conformal(p, y):
        in1 = (1 - p) <= q; in0 = p <= q                                                     # label 1 in set if 1-p <= q, label 0 if p <= q
        covered = np.where(y == 1, in1, in0); ambiguous = in1 & in0; empty = ~in1 & ~in0
        return covered.mean(), ambiguous.mean(), empty.mean()
    sets = [("clean", Xc, yc)]
    for c in CASES:
        if all(os.path.exists(os.path.join(DATA, f"{t}_native_{c}.npy")) for t in TESTS):
            Xs, ys = shifted_test(c, cfg); sets.append((c, Xs, ys))
    rng = np.random.default_rng(SEED + 81)
    for kind, s in (("white", 5), ("correlated", 5)):
        sets.append((f"{kind}_snr{s}", corr_noise(Xc, rng, s, kind=kind, Xnf=Xc_nf, params=Pc, cen=cen), yc))
    rows, L = [], [f"Split-conformal prediction (90 % target, calibrated on clean test1+test2) and calibration under mismatch; {cfg}, {best}", "",
                   f"{'case':<28}{'accuracy':>9}{'coverage':>10}{'ambiguous':>11}{'empty':>7}{'Brier':>8}{'ECE':>7}{'mean p|y=1':>12}"]
    for name, X, y in sets:
        p = proba(X)[ev]; yy = y[ev]
        cov, amb, emp = conformal(p, yy); acc = ((p >= .5) == yy).mean()
        rows.append(dict(case=name, accuracy=acc, conformal_coverage=cov, ambiguous=amb, empty=emp, brier=np.mean((p - yy) ** 2), ece=ece(p, yy), mean_p_pos=p[yy == 1].mean()))
        L.append(f"{name:<28}{acc*100:8.2f}%{cov*100:9.1f}%{amb*100:10.1f}%{emp*100:6.1f}%{rows[-1]['brier']:8.3f}{rows[-1]['ece']:7.3f}{rows[-1]['mean_p_pos']:12.3f}")
    L += ["", "Reading: coverage below 90 % means the conformal guarantee, valid on the calibration distribution, is broken by the mismatch;",
          "'ambiguous' is the conformal decline rate. ECE well above the clean value means the probabilities no longer mean what they say."]
    open(os.path.join(RESULTS, f"{cfg}_conformal_calibration.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"{cfg}_conformal_calibration.csv"), index=False); print("\n".join(L))


if __name__ == "__main__":
    main()
