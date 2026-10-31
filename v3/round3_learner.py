"""Third-round learners (REVISION_EXPECTATIONS.md, section M): one network ensemble, or the four-learner vote, per reduction and precision, trained on population C or C plus B.

Usage: python round3_learner.py --cand C-mlp [--cond rainout local] [--jobs N]. Reads results/goyal_test/grid_<cond>.npz; writes results/round3/{votes_<cand>_<cond>.csv.gz, real_<cand>.csv, thresholds_<cand>.csv}.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from bin_spectra import bin_native  # noqa: E402
from common import DATA, RESULTS, SEED  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402

OUT = os.path.join(RESULTS, "round3"); os.makedirs(OUT, exist_ok=True); GRIDS = os.path.join(RESULTS, "goyal_test"); MOLS = ML.MOLS; DRAWS = 200; MEMBERS = 5; COPIES = 3; FMAX = 0.10


def population(cand):
    wl = np.load(os.path.join(DATA, "native_wl.npy")); e, _ = J.tier3_layout(); Xs, Ps = [], []
    for f, a in (("correlated", "correlated"), ("broad", "broad")):
        if (f == "broad" and "B" not in cand.split("-")[0]) or (f == "correlated" and "C" not in cand.split("-")[0]): continue
        X = np.load(os.path.join(DATA, f, f"{a}_native.npy")).astype(float); P = pd.read_parquet(os.path.join(DATA, f, f"{a}_params.parquet")); ok = np.isfinite(X).all(1); Xs.append(bin_native(X[ok], wl, e)); Ps.append(P[ok][["modulation_5H"] + [f"atm {m}" for m in MOLS]])
    return np.vstack(Xs), pd.concat(Ps, ignore_index=True)


def draw(X, mod, meas, e, md, noise, rng):
    """Training features: scale to the planet, mute half the rows, add the reduction's noise."""
    D = X - X[:, meas].mean(1, keepdims=True); f = md / mod * np.where(rng.uniform(size=len(X)) < 0.5, 10 ** rng.uniform(-1, 0, len(X)), 1.0); T = D * f[:, None]; T[:, meas] += rng.normal(0, 1, (len(T), meas.sum())) * e
    return ML.features(T, meas, noise, True)


def thresholds(p, Y):
    out = []
    for k in range(Y.shape[1]):
        y = Y[:, k].astype(bool); best, tb = -1.0, 0.99
        for t in np.linspace(0.01, 0.99, 99):
            sens, fpr = (p[y, k] >= t).mean() if y.any() else 0.0, (p[~y, k] >= t).mean() if (~y).any() else 0.0
            if fpr <= FMAX and 0.5 * (sens + 1 - fpr) > best: best, tb = 0.5 * (sens + 1 - fpr), t
        out.append(tb)
    return np.array(out)


def fit_mlp(X, P, meas, e, md, noise, seed):
    Y = (P[[f"atm {m}" for m in MOLS]].to_numpy() > -4).astype(int); mod = P["modulation_5H"].to_numpy(); r0 = np.random.default_rng(SEED + 77); hold = r0.uniform(size=len(X)) < 0.15; nets = []
    for k in range(MEMBERS):
        rng = np.random.default_rng(seed + k); Z = np.vstack([draw(X[~hold], mod[~hold], meas, e, md, noise, rng) for _ in range(COPIES)]); y = np.tile(Y[~hold], (COPIES, 1))
        nets.append(MLPClassifier((256, 128), batch_size=256, early_stopping=True, n_iter_no_change=6, max_iter=60, random_state=seed + k).fit(Z, y))
    prob = lambda Z: np.mean([n.predict_proba(Z) for n in nets], axis=0); rng = np.random.default_rng(seed + 99); t = thresholds(prob(draw(X[hold], mod[hold], meas, e, md, noise, rng)), Y[hold])
    def decide(Z):
        p = prob(Z); return {m: ((p[:, i] >= t[i]).astype(float), (p[:, i] < t[i]).astype(float), p[:, i]) for i, m in enumerate(MOLS)}
    return decide, dict(zip(MOLS, t))


def fit_vote(X, P, meas, e, md, noise, seed, bal=False):
    rng = np.random.default_rng(seed); Z = draw(X, P["modulation_5H"].to_numpy(), meas, e, md, noise, rng)
    def fit(mk, y):
        ix = np.arange(len(y))
        if bal:
            p, n = np.where(y == 1)[0], np.where(y == 0)[0]; ix = np.r_[ix, rng.choice(p if len(p) < len(n) else n, abs(len(p) - len(n)))]
        return mk().fit(Z[ix], y[ix])
    learners = {m: [fit(mk, (P[f"atm {m}"] > -4).astype(int).to_numpy()) for mk in ML.MAKERS().values()] for m in MOLS}
    def decide(Zt):
        out = {}
        for m in MOLS:
            v = np.sum([c.predict(Zt) for c in learners[m]], axis=0); out[m] = ((v >= 3).astype(float), (v <= 1).astype(float), v / 4.0)
        return out
    return decide, {}


def independent():
    import independent_test as IT
    e, _ = J.tier3_layout(); M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name"); L = []
    for name, (folder, files, mcs, labs) in IT.LABELS.items():
        x, err, filled = IT.spectrum(folder, files, e); b = mcs or name.replace("-JWST", "").replace("-HST", ""); md = float(M.loc[b, "modulation_5H"]) if mcs else float(IT.modulation(*IT.EXTRA[b])); L.append(("independent", name, labs, x, err, filled, np.zeros_like(x), md))
    return L


def run(j, i, prec, cand, X, P, grids):
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; e = np.nan_to_num(err[meas]); e = np.sqrt(e ** 2 + s[meas] ** 2) if prec == "Tier-1" else e; noise = float(np.sqrt(np.mean(e ** 2))); seed = SEED + 5000 + 10 * i + (0 if prec == "JWST" else 3)
    decide, t = fit_mlp(X, P, meas, e, md, noise, seed) if cand.endswith("mlp") else fit_vote(X, P, meas, e, md, noise, seed, "+bal" in cand); votes, real = [], []
    Xr = x[None, :] if prec == "JWST" else J.noise_draws(x, s, DRAWS, seed=405 + i); res = decide(ML.features(Xr, meas, noise, True))
    for m in MOLS:
        if labs.get(m) is None: continue
        pr, ab, p = res[m]; real.append(dict(batch=batch, planet=pl, precision=prec, molecule=m, design=cand, label=labs[m], present=pr.mean(), absent=ab.mean(), prob=p.mean(), credit=(pr if labs[m] == "present" else ab).mean() + 0.5 * (1 - pr - ab).mean(), threshold=t.get(m, np.nan)))
    for cond, G in grids.items():
        if pl not in G: continue
        r2 = np.random.default_rng(SEED + 1700 + i + (0 if prec == "JWST" else 5000)); Te = G[pl].copy(); Te[:, meas] += r2.normal(0, 1, (len(Te), meas.sum())) * e; res = decide(ML.features(Te, meas, noise, True))
        votes.append(pd.DataFrame([dict(cond=cond, batch=batch, planet=pl, precision=prec, molecule=m, design=cand, row=q, present=res[m][0][q], absent=res[m][1][q], prob=round(float(res[m][2][q]), 4)) for m in MOLS for q in range(len(Te))]))
    print(pl, prec, flush=True); return votes, real, dict(planet=pl, precision=prec, **t)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cand", required=True, choices=["C-mlp", "C+B-mlp", "C-vote", "C+B-vote", "B+bal-vote", "C+bal-vote"]); ap.add_argument("--cond", nargs="*", default=["rainout"]); ap.add_argument("--jobs", type=int, default=6); a = ap.parse_args()
    X, P = population(a.cand); print(X.shape, flush=True); grids = {c: dict(np.load(os.path.join(GRIDS, f"grid_{c}.npz"))) for c in a.cond}; L = ML.jobs_list(); n = len(L); L = L + independent()
    task = [(j, i, "JWST") for i, j in enumerate(L)] + [(j, i, "Tier-1") for i, j in enumerate(L[:n])]; R = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run)(j, i, p, a.cand, X, P, grids) for j, i, p in task)
    V = [v for r in R for v in r[0]]
    if V:
        V = pd.concat(V)
        for c, q in V.groupby("cond"): q.drop(columns="cond").to_csv(os.path.join(OUT, f"votes_{a.cand}_{c}.csv.gz"), index=False)
    pd.DataFrame([x for r in R for x in r[1]]).to_csv(os.path.join(OUT, f"real_{a.cand}.csv"), index=False); pd.DataFrame([r[2] for r in R]).to_csv(os.path.join(OUT, f"thresholds_{a.cand}.csv"), index=False)


if __name__ == "__main__":
    main()
