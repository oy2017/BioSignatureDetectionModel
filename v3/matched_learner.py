"""A method built for each reduction: learners trained on that reduction's measured bins at its real errors (REVISION_EXPECTATIONS.md, section I).

Usage: python matched_learner.py [--jobs N]. Writes results/matched_learner/{matched_learner.csv, matched_learner.txt}.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from bin_spectra import bin_native  # noqa: E402
from common import DATA, RESULTS, SEED  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import jwst_panel as JP  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import router_real as RR  # noqa: E402

OUT = os.path.join(RESULTS, "matched_learner"); os.makedirs(OUT, exist_ok=True); MOLS = ["CH4", "H2O", "CO2", "NH3"]; DRAWS = 200
BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
MAKERS = {"KNN": lambda: KNeighborsClassifier(), "MLP": lambda: MLPClassifier(random_state=SEED), "RFC": lambda: RandomForestClassifier(random_state=SEED, n_jobs=1), "SVC": lambda: SVC()}


def training():
    src = AF.DIRS["faithful"]; wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = AF.layout_edges("tier3_r20"); P = pd.read_parquet(os.path.join(src, "pop3_params.parquet"))
    Xs, Ps = [], []
    for f in ("pop3_native", "pop3_native_aug_hazedeck"):
        Xn = np.load(os.path.join(src, f + ".npy")).astype(np.float64); ok = np.all(np.isfinite(Xn), axis=1); Xs.append(bin_native(Xn[ok], wl, edges)); Ps.append(P[ok])
    return np.vstack(Xs), pd.concat(Ps, ignore_index=True)


def features(X, meas, noise, amp):
    Y = X[:, meas]; Z = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
    return np.column_stack([Z, np.log10(Y.std(1) / noise)]) if amp else Z


def fit_one(pl, labs, x, err, filled, sig_t1, Xtr, Ptr, md, prec, design, seed):
    meas = ~filled; rng = np.random.default_rng(seed); amp = design == "matched+amp"; e = np.nan_to_num(err[meas]); e = np.sqrt(e ** 2 + sig_t1[meas] ** 2) if prec == "Tier-1" else e
    D = Xtr - Xtr[:, meas].mean(1, keepdims=True); f = md / Ptr["modulation_5H"].to_numpy()
    if amp: f = f * np.where(rng.uniform(size=len(f)) < 0.5, 10 ** rng.uniform(-1, 0, len(f)), 1.0)
    T = D * f[:, None]; T[:, meas] = T[:, meas] + rng.normal(0, 1, (len(T), meas.sum())) * e
    noise = float(np.sqrt(np.mean(e ** 2))); Ztr = features(T, meas, noise, amp); xm = x.copy()
    Xte = xm[None, :] if prec == "JWST" else J.noise_draws(xm, sig_t1, DRAWS, seed=405); Zte = features(Xte, meas, noise, amp)
    rows = []
    for m in MOLS:
        if labs.get(m) is None: continue
        y = (Ptr[f"atm {m}"] > -4).astype(int).to_numpy(); v = np.zeros(len(Zte))
        for k, mk in MAKERS.items(): v += mk().fit(Ztr, y).predict(Zte)
        pr, ab = v >= 3, v <= 1; sp = ~(pr | ab); lab = labs[m]; cr = float(np.mean((pr if lab == "present" else ab) + 0.5 * sp))
        rows.append(dict(precision=prec, planet=pl, molecule=m, design=design, label=lab, credit=cr, present=float(pr.mean()), split=float(sp.mean()), absent=float(ab.mean()), batch=None))
    return rows


def jobs_list():
    edges, _ = J.tier3_layout(); M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name"); L = []
    for batch, items, folder in (("panel", {p: d["labels"] for p, d in RR.PANEL.items()}, "jwst_panel"), ("fourth", JP.FOURTH, "jwst_panel_fourth")):
        for pl, labs in items.items():
            f = RR.PANEL[pl]["coverage"] if batch == "panel" else os.path.join(RESULTS, folder, f"{pl}_coverage.csv"); cov = pd.read_csv(f)
            filled = cov.filled.to_numpy(bool); x = cov.depth.to_numpy().copy(); x[filled] = np.nanmean(x[~filled]); L.append((batch, pl, labs, x, cov.err.to_numpy(), filled, J.ariel_sigma(BASE(pl), edges), float(M.loc[BASE(pl), "modulation_5H"])))
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=3); a = ap.parse_args(); Xtr, Ptr = training(); print(Xtr.shape, flush=True)
    def run(i, j, prec, design):
        batch, pl, labs, x, err, filled, s, md = j; r = fit_one(pl, labs, x, err, filled, s, Xtr, Ptr, md, prec, design, SEED + 700 + i)
        for q in r: q["batch"] = batch
        return r
    J_ = jobs_list(); out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run)(i, j, p, d) for i, j in enumerate(J_) for p in ("JWST", "Tier-1") for d in ("matched", "matched+amp"))
    df = pd.DataFrame([r for x in out for r in x]); df.to_csv(os.path.join(OUT, "matched_learner.csv"), index=False)


if __name__ == "__main__":
    main()
