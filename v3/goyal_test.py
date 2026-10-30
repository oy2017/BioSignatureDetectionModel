"""Confirmation on the Goyal et al. (2019) generic ATMO grid (REVISION_EXPECTATIONS.md, section J). Grid: Google Drive folder 1ZFbkPdqg37_Om7ECSspSpEp5QrUMfA9J, the two rainout tars, extracted under data/goyal2019/x/.

Usage: python goyal_test.py [--jobs N] [--cond rainout|local]. Writes results/goyal_test/{goyal_<cond>.csv, goyal_<cond>.txt}.
"""
import argparse
import gzip
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402

OUT = os.path.join(RESULTS, "goyal_test"); os.makedirs(OUT, exist_ok=True); MOLS = ML.MOLS; N = 200; COL = {"H2O": 10, "CH4": 13, "CO2": 16, "NH3": 69}
KB, MH, RSUN, RJUP, GRAV = 1.380658e-16, 1.6726e-24, 69580000000.0, 6991100000.0, np.array([5.0, 10.0, 20.0, 50.0])


def dirs(cond):
    r = os.path.join("data", "goyal2019", "x"); return (os.path.join(r, "transfiles_rainout"), os.path.join(r, "chemfiles_txt")) if cond == "rainout" else (os.path.join(r, "transfiles_local"), os.path.join(r, "chemfiles_local"))


def table(cond):
    t = sorted(f for f in os.listdir(dirs(cond)[0]) if f.endswith(".gz") and not f.startswith("._")); rows = [f.split("_") for f in t]
    return pd.DataFrame(dict(file=t, T=[float(r[1]) for r in rows], g=[float(r[2]) for r in rows], met=[float(r[3]) for r in rows], co=[float(r[4]) for r in rows], haze=[float(r[5]) for r in rows], cloud=[float(r[6]) for r in rows]))


def chem(cond, f):
    with gzip.open(os.path.join(dirs(cond)[1], f.replace("trans-", "chem-")), "rt") as h: A = np.loadtxt(h)
    k = (A[:, 0] >= 1e-4) & (A[:, 0] <= 0.1); return {m: float(np.log10(np.clip(A[k, c], 1e-30, None).mean())) for m, c in COL.items()}, float(A[k, 1].mean())


def rescale(cond, row, mmm, rs, rp, gp, tp):
    """The grid's gg_rescale: depth for a planet of radius rp (RJ, the model's bulk radius being 1 RJ), gravity gp (cgs), temperature tp."""
    with gzip.open(os.path.join(dirs(cond)[0], row.file), "rt") as h: A = np.loadtxt(h)
    o = np.argsort(A[:, 0]); w, d = A[o, 0], A[o, 1]; mu = MH * mmm; t1, g1 = row["T"], row.g * 100; h1 = KB * t1 / (mu * g1); z1 = np.sqrt(d) * RSUN - RJUP; rp2 = rp * RJUP; h2 = KB * tp / (mu * gp)
    z2 = h2 * (z1 / h1 + np.log(np.sqrt(KB * t1 * mu * g1 / (2 * np.pi * RJUP)) * np.sqrt(2 * np.pi * rp2 / (KB * tp * mu * gp)))); return w, ((z2 + rp2) / (rs * RSUN)) ** 2


def grid_for(j, i, cond, T, M, edges):
    batch, pl, labs, x, err, filled, s, md = j; m = M.loc[ML.BASE(pl)]; rp = m.p_radius * 0.08921; tp = float(m["atm temperature"])
    gp = 2479.0 * (m.p_mass / 317.8) / rp ** 2; g0 = GRAV[np.argmin(np.abs(np.log(GRAV) - np.log(gp / 100)))]; q = T[(np.abs(T["T"] - tp) <= 200) & (T.g == g0)]
    if len(q) == 0: q = T[(T["T"] == T["T"].iloc[np.argmin(np.abs(T["T"] - tp))]) & (T.g == g0)]
    q = q.sample(min(N, len(q)), random_state=SEED + 900 + i); X, Y = [], []
    for _, r in q.iterrows():
        c, mmm = chem(cond, r.file); w, d = rescale(cond, r, mmm, float(m["s radius"]), rp, gp, tp); X.append(ML.bin_native(d[None, :], w, edges)[0]); Y.append(dict(file=r.file, T=r["T"], g=r.g, met=r.met, co=r.co, haze=r.haze, cloud=r.cloud, mmm=mmm, **{f"v_{k}": v for k, v in c.items()}))
    return np.array(X), pd.DataFrame(Y)


def run_matched(j, i, X, Y, Xtr, Ptr):
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; out = []
    for prec in ("JWST", "Tier-1"):
        rng = np.random.default_rng(SEED + 700 + i + (0 if prec == "JWST" else 5000)); e = np.nan_to_num(err[meas]); e = np.sqrt(e ** 2 + s[meas] ** 2) if prec == "Tier-1" else e
        D = Xtr - Xtr[:, meas].mean(1, keepdims=True); f = md / Ptr["modulation_5H"].to_numpy() * np.where(rng.uniform(size=len(Xtr)) < 0.5, 10 ** rng.uniform(-1, 0, len(Xtr)), 1.0); Tr = D * f[:, None]; Tr[:, meas] += rng.normal(0, 1, (len(Tr), meas.sum())) * e
        noise = float(np.sqrt(np.mean(e ** 2))); Ztr = ML.features(Tr, meas, noise, True); r2 = np.random.default_rng(SEED + 1700 + i + (0 if prec == "JWST" else 5000)); Te = X.copy(); Te[:, meas] += r2.normal(0, 1, (len(X), meas.sum())) * e; Zte = ML.features(Te, meas, noise, True)
        for m in MOLS:
            y = (Ptr[f"atm {m}"] > -4).astype(int).to_numpy(); v = np.zeros(len(Zte))
            for mk in ML.MAKERS().values(): v += mk().fit(Ztr, y).predict(Zte)
            out.append(pd.DataFrame(dict(batch=batch, planet=pl, precision=prec, molecule=m, design="matched+amp", row=np.arange(len(v)), present=(v >= 3).astype(float), absent=(v <= 1).astype(float))))
    return pd.concat(out)


def run_frozen(ctx, j, i, X, F, Fl):
    import panel_tier1 as PT
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; hi, lo = (F.hi_clear_max[pl], F.lo_own_p01[pl]) if batch == "panel" else (Fl.hi[pl], Fl.lo[pl]); out = []; e = np.nan_to_num(err)
    j2 = float(np.nanmean(err[meas] ** 2)) * 1e12; t2 = float(np.mean(s[meas] ** 2)) * 1e12
    for prec, noise2 in (("JWST", j2), ("Tier-1", j2 + t2)):
        r = np.random.default_rng(SEED + 1700 + i + (0 if prec == "JWST" else 5000)); B = X.copy(); B[:, filled] = B[:, meas].mean(1, keepdims=True); B = B.copy(); ee = np.zeros(X.shape); ee[:, meas] = r.normal(0, 1, (len(X), meas.sum())) * (np.sqrt(e[meas] ** 2 + s[meas] ** 2) if prec == "Tier-1" else e[meas]); B = B + ee
        if prec == "Tier-1": B[:, filled] += r.normal(0, 1, (len(X), filled.sum())) * s[filled]
        res = PT.score(ctx, B, meas, ML.BASE(pl), noise2, hi, lo)
        for m in MOLS:
            votes, low, up = res[m]
            for d in ("published", "AIRS-only", "shape flag", "haze+deck-retrained", "the method"):
                pr = votes[d] >= 3; ab = votes[d] <= 1
                if d == "the method": pr = pr & ~low; ab = ab & ~low
                out.append(pd.DataFrame(dict(batch=batch, planet=pl, precision=prec, molecule=m, design=d, row=np.arange(len(X)), present=pr.astype(float), absent=ab.astype(float))))
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--cond", default="rainout"); a = ap.parse_args(); tag = a.cond
    edges, _ = J.tier3_layout(); M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name"); T = table(a.cond); L = ML.jobs_list(); Xtr, Ptr = ML.training()
    cache = os.path.join(OUT, f"grid_{tag}.npz")
    G = Parallel(n_jobs=a.jobs)(delayed(grid_for)(j, i, a.cond, T, M, edges) for i, j in enumerate(L)); print("grid read", flush=True)
    Y = pd.concat([g[1].assign(planet=j[1], row=np.arange(len(g[1]))) for g, j in zip(G, L)], ignore_index=True); Y.to_csv(os.path.join(OUT, f"truth_{tag}.csv"), index=False); np.savez_compressed(cache, **{j[1]: g[0] for g, j in zip(G, L)})
    import panel_tier1 as PT
    ctx = PT.Context(); F = pd.read_csv(os.path.join(RESULTS, "jwst_panel", "amplitude_flag.csv")).set_index("planet"); Fl = pd.read_csv(os.path.join(RESULTS, "jwst_panel_fourth", "fourth_batch_flags.csv")).set_index("planet")
    fro = [run_frozen(ctx, j, i, g[0], F, Fl) for i, (j, g) in enumerate(zip(L, G))]; print("frozen done", flush=True); del ctx
    pd.concat(fro).to_csv(os.path.join(OUT, f"votes_frozen_{tag}.csv.gz"), index=False)
    mat = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_matched)(j, i, g[0], g[1], Xtr, Ptr) for i, (j, g) in enumerate(zip(L, G)))
    pd.concat(mat).to_csv(os.path.join(OUT, f"votes_matched_{tag}.csv.gz"), index=False); print("done", flush=True)


if __name__ == "__main__":
    main()
