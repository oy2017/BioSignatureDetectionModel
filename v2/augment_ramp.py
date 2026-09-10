"""Break the confound in the repair rule.

The rule is established on four axes: stellar contamination and haze, which are
deterministic maps AND physics re-renders, against correlated and white noise,
which are stochastic AND injected. Deterministic/stochastic is therefore perfectly
confounded with physics/instrument, and the four axes cannot tell the proposed
criterion apart from the duller claim that re-rendered physics repairs and injected
noise does not.

The gain ramp breaks it. It is an instrument systematic, injected exactly like the
noise axes, but its effect on a spectrum is a fixed multiplicative tilt rather than
a fresh draw per bin: the whole perturbation is one number and a sign. If the
criterion is about determinism, the ramp should repair like the physics axes. If it
is really about physics versus instrument, the ramp should repair like the noise.

The test spectra are built here rather than reused from evaluate_shifts.py, so the
frozen and augmented pipelines are compared on identical arrays; the ramp itself is
constructed exactly as evaluate_shifts.py constructs it.

Usage: python augment_ramp.py
Writes results/ariel_augment_ramp.txt / .csv
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features,  # noqa: E402
                    centres, load_split, metrics)
from noise import sigma_matrix  # noqa: E402
from pipeline import make_xgb  # noqa: E402

CFG = "ariel"
CLEAN = 0.9033
STRENGTHS = [0.0, 0.5, 1.0, 2.0]      # multiples of the noise level, as in the budget
TEST_STRENGTHS = [0.5, 1.0, 2.0]


def ramp(X, Xnf, params, cen, s, rng):
    """One multiplicative tilt across wavelength, sign drawn per planet.

    Identical construction to evaluate_shifts.py: the amplitude is s noise levels
    relative to the mean absolute depth, and the tilt runs linearly from -1 to +1
    across the bins.
    """
    if s == 0:
        return X.copy()
    sig_n = np.median(sigma_matrix(Xnf, params["s temperature"].to_numpy(), cen),
                      axis=1, keepdims=True)
    amp = s * sig_n / np.abs(X).mean(axis=1, keepdims=True)
    tilt = np.linspace(-1, 1, X.shape[1])
    signs = rng.choice([-1.0, 1.0], size=(len(X), 1))
    return X * (1 + signs * amp * tilt)


def main():
    cen = centres(CFG)
    best = json.load(open(os.path.join(RESULTS, f"{CFG}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{CFG}_{best}.joblib"))
    ff, fm, params = fr["features"], fr["model"], fr["params"]
    kind = best.split("_")[0]

    Xtr, ytr, Ptr = load_split("train", CFG)
    Xtr_nf = np.load(os.path.join(DATA, f"train_{CFG}.npy")).astype(float)

    Xc = np.vstack([load_split(t, CFG)[0] for t in TESTS])
    yc = np.concatenate([load_split(t, CFG)[1] for t in TESTS])
    Xc_nf = np.vstack([np.load(os.path.join(DATA, f"{t}_{CFG}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, CFG, noisy=False)[2] for t in TESTS], ignore_index=True)

    # augmented training set: each planet at a random ramp strength, a quarter clean
    rng = np.random.default_rng(SEED + 11)
    lev = rng.choice(STRENGTHS, size=len(ytr))
    Xa = Xtr.copy()
    for s in STRENGTHS:
        m = lev == s
        if s > 0:
            Xa[m] = ramp(Xtr[m], Xtr_nf[m], Ptr[m].reset_index(drop=True), cen, s, rng)
    f = Features(kind).fit(Xa)
    m_aug = make_xgb(params).fit(f.transform(Xa), ytr)

    L = ["Gain-ramp augmentation: a DETERMINISTIC shift that is not a physics re-render.",
         "",
         "The repair rule was established on two physics re-renders against two injected",
         "noise axes, which confounds deterministic/stochastic with physics/instrument.",
         "The gain ramp is injected like the noise but deterministic in form, so it",
         "separates the two explanations.",
         "",
         f"{'case':<14}{'frozen':>9}{'augmented':>11}{'gain':>8}{'% of gap':>10}"]
    rows = []
    rng2 = np.random.default_rng(SEED + 12)
    for s in TEST_STRENGTHS:
        Xs = ramp(Xc, Xc_nf, Pc, cen, s, rng2)
        a_fr = metrics(yc, fm.predict_proba(ff.transform(Xs))[:, 1])["accuracy"]
        a_au = metrics(yc, m_aug.predict_proba(f.transform(Xs))[:, 1])["accuracy"]
        pct = (a_au - a_fr) / max(CLEAN - a_fr, 1e-9) * 100
        cost = (CLEAN - a_fr) * 100
        L.append(f"{'x'+str(s):<14}{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}"
                 f"{pct:9.0f}%   (costs {cost:.1f} points)")
        rows.append(dict(axis="gain ramp", case=f"x{s}", frozen=a_fr, augmented=a_au,
                         pct_of_gap=pct, cost=cost))
    a_fr = metrics(yc, fm.predict_proba(ff.transform(Xc))[:, 1])["accuracy"]
    a_au = metrics(yc, m_aug.predict_proba(f.transform(Xc))[:, 1])["accuracy"]
    L.append(f"{'clean':<14}{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}")
    rows.append(dict(axis="gain ramp", case="clean", frozen=a_fr, augmented=a_au,
                     pct_of_gap=np.nan, cost=0.0))

    big = [r for r in rows if r["case"] != "clean" and r["cost"] > 5.0]
    L += [""]
    if big:
        lo = min(r["pct_of_gap"] for r in big)
        hi = max(r["pct_of_gap"] for r in big)
        L.append(f"On the >5-point criterion the paper uses: {len(big)} case(s), "
                 f"{lo:.0f}-{hi:.0f}% recovered.")
        L.append("Deterministic physics axes recover 79-89%; stochastic noise axes 29-35%.")
        verdict = ("with the deterministic axes: determinism, not physics, is what predicts repair"
                   if lo >= 60 else
                   "with the stochastic axes: the split tracks physics vs instrument, not determinism"
                   if hi <= 45 else
                   "between the two bands: the criterion is not clean on this axis")
        L.append(f"The gain ramp lands {verdict}.")
    else:
        L.append("No case cleared the >5-point criterion, so this axis cannot settle it.")

    open(os.path.join(RESULTS, "ariel_augment_ramp.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "ariel_augment_ramp.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
