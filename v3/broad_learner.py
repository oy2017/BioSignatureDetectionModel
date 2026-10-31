"""The second-round learners on a grid (REVISION_EXPECTATIONS.md, section L): per-reduction, as matched+amp, with the broad population B, and optionally balanced classes.

Usage: python broad_learner.py [--jobs N] [--cond rainout]. Reads results/goyal_test/grid_<cond>.npz; writes results/goyal_test/votes_broad_<cond>.csv.gz.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from bin_spectra import bin_native  # noqa: E402
from common import DATA, RESULTS, SEED  # noqa: E402
import matched_learner as ML  # noqa: E402

OUT = os.path.join(RESULTS, "goyal_test"); MOLS = ML.MOLS


def broad():
    from jwst_to_ariel import tier3_layout
    wl = np.load(os.path.join(DATA, "native_wl.npy")); e, _ = tier3_layout(); Xn = np.load(os.path.join(DATA, "broad", "broad_native.npy")).astype(float); P = pd.read_parquet(os.path.join(DATA, "broad", "broad_params.parquet")); ok = np.isfinite(Xn).all(1)
    return bin_native(Xn[ok], wl, e), P[ok].reset_index(drop=True)


def learners_for(j, i, prec, Xtr, Ptr):
    """Training features for a reduction at a precision."""
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; rng = np.random.default_rng(SEED + 3700 + i + (0 if prec == "JWST" else 5000)); e = np.nan_to_num(err[meas]); e = np.sqrt(e ** 2 + s[meas] ** 2) if prec == "Tier-1" else e
    D = Xtr - Xtr[:, meas].mean(1, keepdims=True); f = md / Ptr["modulation_5H"].to_numpy() * np.where(rng.uniform(size=len(Xtr)) < 0.5, 10 ** rng.uniform(-1, 0, len(Xtr)), 1.0); T = D * f[:, None]; T[:, meas] += rng.normal(0, 1, (len(T), meas.sum())) * e
    noise = float(np.sqrt(np.mean(e ** 2))); return ML.features(T, meas, noise, True), noise, e, meas, rng


def fit_votes(Ztr, y, Zte, bal, rng):
    if bal:
        p, n = np.where(y == 1)[0], np.where(y == 0)[0]; k = rng.choice(p if len(p) < len(n) else n, abs(len(p) - len(n))); ix = np.r_[np.arange(len(y)), k]; Ztr, y = Ztr[ix], y[ix]
    v = np.zeros(len(Zte))
    for mk in ML.MAKERS().values(): v += mk().fit(Ztr, y).predict(Zte)
    return v


def run(j, i, X, Xtr, Ptr):
    batch, pl = j[0], j[1]; out = []
    for prec in ("JWST", "Tier-1"):
        Ztr, noise, e, meas, rng = learners_for(j, i, prec, Xtr, Ptr); r2 = np.random.default_rng(SEED + 1700 + i + (0 if prec == "JWST" else 5000)); Te = X.copy(); Te[:, meas] += r2.normal(0, 1, (len(X), meas.sum())) * e; Zte = ML.features(Te, meas, noise, True)
        for m in MOLS:
            y = (Ptr[f"atm {m}"] > -4).astype(int).to_numpy()
            for name, bal in (("B", False), ("B+bal", True)):
                v = fit_votes(Ztr, y, Zte, bal, rng); out.append(pd.DataFrame(dict(batch=batch, planet=pl, precision=prec, molecule=m, design=name, row=np.arange(len(v)), present=(v >= 3).astype(float), absent=(v <= 1).astype(float))))
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--cond", default="rainout"); a = ap.parse_args(); Xtr, Ptr = broad(); L = ML.jobs_list(); G = np.load(os.path.join(OUT, f"grid_{a.cond}.npz"))
    R = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run)(j, i, G[j[1]], Xtr, Ptr) for i, j in enumerate(L)); pd.concat(R).to_csv(os.path.join(OUT, f"votes_broad_{a.cond}.csv.gz"), index=False)


if __name__ == "__main__":
    main()
