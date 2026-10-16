"""Calibrate the dchi2 identifiability test against nested-sampling posteriors (REVISION_PLAN.md, B1).

On a subset of POP-I planets, a full Bayesian retrieval of the noise-free Tier-3 spectrum with the
planet's Tier-1 per-bin sigma as the likelihood width gives, for each molecule X, the posterior
probability of the TRUE class, p_true = P(log X > t | data) for a present molecule and
1 - P(...) for an absent one. This is the expected posterior, the information-content measure
used by tier studies; --noisy adds one noise draw instead. The retrieval frees the same quantities
the dchi2 test searches (T, gray cloud-top pressure, the four abundances) plus an additive depth
offset, with uniform priors on the population's ranges, nestle's multi-ellipsoid sampler, and a
hard cap on likelihood calls. The forward model is built once per planet and its fitting
parameters updated per call (TauREx's tm[name] = value), which is several times faster than
rebuilding.

After the subset: the Spearman correlation between dchi2 (from identifiability.py) and
logit(p_true) per molecule, and the dchi2 at which a logistic fit of p_true crosses 0.95 and
0.84: the calibrated thresholds for "decidable".

Usage:
  python identifiability_calibrate.py --n 200 --jobs 2            # retrievals (resumable)
  python identifiability_calibrate.py --n 1 --jobs 1 --npoints 50  # timing
  python identifiability_calibrate.py --summarize                  # the mapping, once both CSVs exist
Writes results/identifiability_calibration.csv / .txt
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import alfnoor_screen as AS  # noqa: E402
import forward_model as FM  # noqa: E402
import generate_grid as G  # noqa: E402
from identifiability import THRESH, T_RANGE, LOGP_RANGE, AB_RANGE  # noqa: E402

MOLS = AS.MOLS
OFFSET = 5e-3            # +- fractional depth, uniform prior on the additive offset


def select(Pte, n, seed=SEED):
    """Half random, half among planets with at least one label within 0.7 dex of the threshold."""
    rng = np.random.default_rng(seed)
    near = np.where(np.min(np.abs(Pte[[f"atm {m}" for m in MOLS]].to_numpy() - THRESH), axis=1) < 0.7)[0]
    far = np.setdiff1d(np.arange(len(Pte)), near)
    a = rng.choice(near, min(n // 2, len(near)), replace=False); b = rng.choice(far, n - len(a), replace=False)
    return np.sort(np.concatenate([a, b]))


def retrieve_one(i, row, obs, sig, wl, edges, npoints, maxcall, seed):
    import warnings; warnings.filterwarnings("ignore")
    import nestle
    G.FILL_GAS = AF.FILL
    t0 = time.time()
    try:
        r = dict(row); tm = FM.build_model(r, list(G.GASES), AF.FILL, cloud_pressure=float(r["cloud_pressure"]))
        T_p = float(r["T_p"])
        lo = np.array([T_RANGE[0] * T_p, LOGP_RANGE[0]] + [AB_RANGE[0]] * 4 + [-OFFSET])
        hi = np.array([T_RANGE[1] * T_p, LOGP_RANGE[1]] + [AB_RANGE[1]] * 4 + [OFFSET])

        def prior(u):
            return lo + u * (hi - lo)

        def loglike(x):
            tm["T"] = x[0]; tm["clouds_pressure"] = 10.0 ** x[1]
            for k, m in enumerate(MOLS):
                tm[m] = 10.0 ** x[2 + k]
            try:
                wn, d = tm.model()[:2]
            except Exception:
                return -1e10
            w = 1e4 / np.asarray(wn, float); o = np.argsort(w)
            d = np.asarray(d, float)[o]; w = w[o]; m = (w >= FM.WL_MIN) & (w <= FM.WL_MAX)
            y = bin_native(d[m][None, :], w[m], edges)[0] + x[6]
            if not np.all(np.isfinite(y)):
                return -1e10
            ll = -0.5 * np.sum(((obs - y) / sig) ** 2)
            return ll if np.isfinite(ll) else -1e10

        rng = np.random.RandomState(seed)
        res = nestle.sample(loglike, prior, 7, method="multi", npoints=npoints, maxcall=maxcall, rstate=rng)
        w = res.weights / res.weights.sum(); out = dict(index=int(i), name=r["name"], ncall=int(res.ncall), minutes=(time.time() - t0) / 60, logz=float(res.logz))
        for k, m in enumerate(MOLS):
            s = res.samples[:, 2 + k]; p = float(w[s > THRESH].sum()); lab = int(r[f"atm {m}"] > THRESH)
            q = lambda f: float(np.interp(f, np.cumsum(w[np.argsort(s)]), np.sort(s)))
            out.update({f"{m}_label": lab, f"{m}_p_present": p, f"{m}_p_true": p if lab else 1 - p, f"{m}_med": q(0.5), f"{m}_lo": q(0.16), f"{m}_hi": q(0.84), f"{m}_true": float(r[f"atm {m}"])})
        return out
    except Exception as e:
        return dict(index=int(i), name=row.get("name", "?"), ncall=0, minutes=(time.time() - t0) / 60, error=str(e)[:120])


def summarize():
    from scipy.stats import spearmanr
    from scipy.optimize import brentq
    cal = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv"))
    idf = pd.read_csv(os.path.join(RESULTS, "identifiability_none_tier1.csv"))
    L = [f"dchi2 test against nested-sampling posteriors on {cal['index'].nunique()} POP-I planets (noise-free spectra, Tier-1 sigma)", ""]
    rows = []
    for m in MOLS:
        a = idf[idf.molecule == m][["index", "dchi2"]].merge(cal[["index", f"{m}_p_true", f"{m}_label"]], on="index")
        if len(a) < 10:
            L.append(f"{m}: only {len(a)} planets in both files"); continue
        p = np.clip(a[f"{m}_p_true"].to_numpy(), 1e-4, 1 - 1e-4); d = a.dchi2.to_numpy()
        rho = spearmanr(d, np.log(p / (1 - p))).correlation
        # logistic fit of p_true against log10(dchi2): p = 1 / (1 + exp(-(a + b log10 d)))
        x = np.log10(np.clip(d, 1e-3, None)); y = (p > 0.5).astype(float)
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6).fit(x[:, None], (a[f"{m}_p_true"] > 0.95).astype(int))
        f = lambda t, q: lr.predict_proba([[np.log10(t)]])[0, 1] - q
        thr = {}
        for q in (0.95, 0.84):
            try:
                thr[q] = brentq(f, 1e-3, 1e4, args=(0.5,)) if q == 0.95 else np.nan
            except ValueError:
                thr[q] = np.nan
        L.append(f"{m:<4} n={len(a)}  Spearman(dchi2, logit p_true) = {rho:.2f};  P(p_true > 0.95) crosses 1/2 at dchi2 = {thr[0.95]:.2f}  "
                 f"(nested-model 95 %: 3.84; 3 sigma: 9)")
        rows.append(dict(molecule=m, n=len(a), spearman=rho, dchi2_at_95=thr[0.95]))
    L.append(""); L.append("decidable in the retrieval sense (p_true > 0.95) by molecule: " + ", ".join(
        f"{m} {(cal[f'{m}_p_true'] > 0.95).mean()*100:.0f} %" for m in MOLS if f"{m}_p_true" in cal))
    open(os.path.join(RESULTS, "identifiability_calibration.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--npoints", type=int, default=100); ap.add_argument("--maxcall", type=int, default=20000)
    ap.add_argument("--noisy", action="store_true"); ap.add_argument("--summarize", action="store_true"); a = ap.parse_args()
    if a.summarize:
        summarize(); return
    src = AF.DIRS["faithful"]; wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = AF.layout_edges("tier3_r20")
    Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet")); Y = np.load(os.path.join(src, "pop1_native.npy")).astype(np.float64)
    ok = np.all(np.isfinite(Y), axis=1); Pte = Pte[ok].reset_index(drop=False).rename(columns={"index": "_index"}); Y = Y[ok]
    mcs = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet"))[["name", "atm temperature"]].rename(columns={"atm temperature": "T_p"})
    Pte = Pte.merge(mcs.drop_duplicates("name"), on="name", how="left")
    sig = AF.sigma(Pte, edges, "radiometric"); Xb = bin_native(Y, wl, edges)
    if a.noisy:
        Xb = Xb + np.random.default_rng(SEED + 7).normal(0, 1, Xb.shape) * sig
    sel = select(Pte, a.n)
    out = os.path.join(RESULTS, "identifiability_calibration.csv")
    done = pd.read_csv(out) if os.path.exists(out) and a.n > 1 else pd.DataFrame()
    todo = [i for i in sel if done.empty or int(Pte.iloc[i]["_index"]) not in set(done["index"])]
    print(f"{len(sel)} planets selected, {len(todo)} to retrieve, {a.jobs} jobs, {a.npoints} live points, cap {a.maxcall} calls", flush=True)
    from joblib import Parallel, delayed
    t0 = time.time()
    for b in range(0, len(todo), a.jobs):
        batch = todo[b:b + a.jobs]
        rows = Parallel(n_jobs=a.jobs)(delayed(retrieve_one)(int(Pte.iloc[i]["_index"]), Pte.iloc[i].to_dict(), Xb[i], sig[i], wl, edges, a.npoints, a.maxcall, SEED + int(i)) for i in batch)
        new = pd.DataFrame(rows); done = pd.concat([done, new], ignore_index=True)
        if a.n > 1:
            done.to_csv(out, index=False)
        print(f"  {min(b + a.jobs, len(todo))}/{len(todo)}  {time.time()-t0:.0f} s  median {new.minutes.median():.1f} min, {new.ncall.median():.0f} calls per planet", flush=True)
        if a.n == 1:
            print(new.T.to_string())


if __name__ == "__main__":
    main()
