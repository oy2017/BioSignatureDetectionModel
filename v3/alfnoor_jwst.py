"""The consortium classifier, as re-implemented, applied to converted JWST spectra (REVISION_PLAN.md, A4).

Training is exactly alfnoor_faithful.do_fit's: the faithful POP-III natives binned to the 104-bin
Tier-3 layout, scattered with the radiometric Tier-1 noise of each planet (same seed, so the same
noise draw as the note's reproduction), per-spectrum normalized, four scikit-learn learners at their
defaults per molecule and abundance threshold. Three input variants, as in alfnoor_haze_mechanism.py:

  all          all 104 bins (the published design)
  no_optical   the three photometric points dropped (bins above 1.1 um), normalized over the rest
  airs_only    the AIRS bins only (above 1.95 um), normalized over those

A converted spectrum (104 transit depths from jwst_to_ariel) is scored three ways:
  (i)   as delivered (JWST precision): verdict and probability per learner, molecule, threshold
  (ii)  under Ariel noise: n Gaussian draws with the target's per-bin sigma for a tier; the fraction
        of draws in which each learner calls the molecule present
  (iii) the Mugnai band metric M_mol on the same bins (their non-ML method)

Learners are cached under data/alfnoor_jwst/ (git-ignored); delete the cache after any change to the
training populations.

Usage (library): see verdict_table() and tier_frequencies(); `python alfnoor_jwst.py --demo` trains the
three variants and scores a simulated WASP-39 b-like planet end to end (no real data).
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, SEED  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import alfnoor_screen as AS  # noqa: E402
import jwst_to_ariel as J  # noqa: E402

CACHE = os.path.join(DATA, "alfnoor_jwst"); os.makedirs(CACHE, exist_ok=True)
# ALFNOOR_TRAIN=haze|hazedeck|union trains on the departure-aware POP-III sets of alfnoor_retrain.py instead of the faithful one
TRAIN = os.environ.get("ALFNOOR_TRAIN", "faithful")
VARIANTS = {"all": lambda cen: np.ones_like(cen, bool), "no_optical": lambda cen: cen > 1.1, "airs_only": lambda cen: cen > 1.95}
MOLS, THRESH = AS.MOLS, AS.THRESH


def _makers():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    return {"KNN": lambda: KNeighborsClassifier(), "MLP": lambda: MLPClassifier(random_state=SEED),
            "RFC": lambda: RandomForestClassifier(random_state=SEED, n_jobs=8), "SVC": lambda: SVC()}


def znorm(X, cols):
    X = np.atleast_2d(np.asarray(X, float))[:, cols]
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def training_set():
    """Binned, noised POP-III exactly as do_fit builds it (natives 'faithful', layout tier3_r20, noise radiometric)."""
    src = AF.DIRS["faithful"]; wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = AF.layout_edges("tier3_r20")
    Ptr = pd.read_parquet(os.path.join(src, "pop3_params.parquet"))
    aug = "hazedeck" if TRAIN == "union" else TRAIN
    Xn = np.load(os.path.join(src, "pop3_native.npy" if TRAIN == "faithful" else f"pop3_native_aug_{aug}.npy")).astype(np.float64)
    ok = np.all(np.isfinite(Xn), axis=1); Ptr = Ptr[ok].reset_index(drop=True); Xn = Xn[ok]; nf = 1.0
    if TRAIN == "union":
        import alfnoor_retrain as AR
        Xn, nf = AR.union_natives(Xn, Ptr, wl); nf = nf[:, None]
    rng = np.random.default_rng(SEED + 405)
    Xtr = bin_native(Xn, wl, edges); Xtr = Xtr + rng.normal(0, 1, Xtr.shape) * AF.sigma(Ptr, edges, "radiometric") * nf
    return Xtr, Ptr, edges


def learners(variant):
    """{(threshold, molecule, learner): fitted model} for the input variant, with its column mask; cached."""
    f = os.path.join(CACHE, f"learners_{variant}.joblib" if TRAIN == "faithful" else f"learners_{variant}_{TRAIN}.joblib")
    if os.path.exists(f):
        return joblib.load(f)
    Xtr, Ptr, edges = training_set(); cen = 0.5 * (edges[1:] + edges[:-1]); cols = VARIANTS[variant](cen)
    Ztr = znorm(Xtr, cols); out = {"cols": cols, "models": {}}
    for th_name, th in THRESH.items():
        for mol in MOLS:
            y = (Ptr[f"atm {mol}"] > th).astype(int).to_numpy()
            for name, mk in _makers().items():
                out["models"][(th_name, mol, name)] = mk().fit(Ztr, y)
    joblib.dump(out, f); return out


def _present_prob(m, Z):
    """P(present) where the learner offers one; SVC gives its signed decision value instead."""
    if hasattr(m, "predict_proba"):
        return m.predict_proba(Z)[:, 1]
    return m.decision_function(Z)


def verdict_table(depth104, variants=("all", "no_optical", "airs_only")):
    """Verdicts at the delivered precision: one row per variant, threshold, molecule, learner."""
    rows = []
    for v in variants:
        L = learners(v); Z = znorm(depth104, L["cols"])
        for (th, mol, name), m in L["models"].items():
            rows.append(dict(variant=v, threshold=th, molecule=mol, learner=name, present=int(m.predict(Z)[0]), score=float(_present_prob(m, Z)[0])))
    return pd.DataFrame(rows)


def tier_frequencies(depth104, sigma104, n=1000, seed=0, variants=("all", "no_optical", "airs_only"), threshold="1e-4"):
    """Fraction of n noise draws in which each learner calls each molecule present, at one threshold."""
    X = J.noise_draws(np.asarray(depth104, float), np.asarray(sigma104, float), n, seed); rows = []
    for v in variants:
        L = learners(v); Z = znorm(X, L["cols"])
        for (th, mol, name), m in L["models"].items():
            if th != threshold:
                continue
            rows.append(dict(variant=v, threshold=th, molecule=mol, learner=name, frequency=float(m.predict(Z).mean())))
    return pd.DataFrame(rows)


def band_metrics(depth104, edges):
    return {mol: J.band_metric(np.asarray(depth104, float), edges, mol) for mol in ("H2O", "CH4", "CO2")}


def summarize(name, depth104, edges, sigma_by_tier, n=1000):
    """Text summary of (i)-(iii) for one converted spectrum; sigma_by_tier = {'tier1': sigma104, ...}."""
    V = verdict_table(depth104); L = [f"== {name}: verdicts at the delivered precision (1 = present)"]
    for th in THRESH:
        t = V[V.threshold == th].pivot_table(index=["variant", "molecule"], columns="learner", values="present")
        L.append(f"-- threshold {th}"); L.append(t.to_string())
    for tier, s in sigma_by_tier.items():
        F = tier_frequencies(depth104, s, n=n).pivot_table(index=["variant", "molecule"], columns="learner", values="frequency")
        L.append(f"== {name}: fraction of {n} Ariel {tier} noise draws called present (threshold 1e-4)"); L.append((F * 100).round(0).to_string())
    bm = band_metrics(depth104, edges)
    L.append(f"== band metric (Mugnai et al. 2021, Eq. 3): " + ", ".join(f"M_{k} = {v:.2f}" for k, v in bm.items())
             + "  (their flags: CH4 >= 0.22, CO2 >= 0.26, H2O >= 0.80)")
    return "\n".join(L), V


def demo():
    """A simulated WASP-39 b-like planet (10x-solar-like abundances, gray deck absent) through the whole chain."""
    import generate_grid as G
    edges, labels = J.tier3_layout(); wl = np.load(os.path.join(DATA, "native_wl.npy"))
    M = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet")); r = M[M.name == "WASP-39b"].iloc[0].to_dict()
    r.update({"atm top_pressure": 1e-4, "atm fill_gas": "H2+He", "atm H2O": -2.3, "atm CO2": -4.5, "atm CH4": -7.0, "atm NH3": -7.0, "atm CO": -3.5, "atm O3": -15.0})
    G.FILL_GAS = AF.FILL
    y = G.one_native(r, wl).astype(float); direct = bin_native(y[None, :], wl, edges)[0]
    sig = {t: J.ariel_sigma("WASP-39b", edges, c) for t, c in (("tier1", "tier1_transits"), ("tier3", "tier3_transits"))}
    text, _ = summarize("simulated WASP-39b-like planet (truth: H2O -2.3, CO2 -4.5, CH4 -7, NH3 -7)", direct, edges, sig, n=500)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true"); a = ap.parse_args()
    if a.demo:
        demo()
