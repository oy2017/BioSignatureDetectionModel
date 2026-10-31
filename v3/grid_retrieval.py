"""U7 of REVISION_EXPECTATIONS.md: a retrieval of photochemical-grid spectra through the real masks, beside the matched ensemble and the routing method on the same labels. Writes results/round3/grid_retrieval_<prec>.csv and grid_retrieval.txt.

Usage: python grid_retrieval.py [jobs], and --summarize."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import DATA, SEED  # noqa: E402
import identifiability_calibrate as IC  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402

OUT = os.path.join("results", "round3"); GR = os.path.join("results", "goyal_test"); N = 90; MOLS = ML.MOLS; AB = (-9.0, -1.0)


def _retrieve(*a, **k):
    IC.OFFSET = 0.02
    return IC.retrieve_one(*a, **k)


def pairs(Y):
    return Y.sample(N, random_state=SEED + 6100)[["planet", "row"]].reset_index(drop=True)


def main():
    if "--summarize" in sys.argv: return summarize()
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 5; Y = pd.read_csv(os.path.join(GR, "truth_crossfield.csv")); Z = np.load(os.path.join(GR, "grid_crossfield.npz")); L = {j[1]: (i, j) for i, j in enumerate(ML.jobs_list())}; M = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet")).drop_duplicates("name").set_index("name"); wl = np.load(os.path.join(DATA, "native_wl.npy")); edges, _ = J.tier3_layout(); P = pairs(Y)
    for prec in ("jwst", "tier1"):
        out = os.path.join(OUT, f"grid_retrieval_{prec}.csv"); done = set(pd.read_csv(out).key) if os.path.exists(out) else set(); T = []
        for k, q in P.iterrows():
            key = f"{q.planet}:{q.row}"
            if key in done: continue
            i, j = L[q.planet]; _, pl, _, _, err, filled, s, _ = j; meas = ~filled; e = np.nan_to_num(err); r_ = np.random.default_rng(SEED + 1700 + i + (0 if prec == "jwst" else 5000)); B = Z[pl].copy(); B[:, meas] += r_.normal(0, 1, (len(B), meas.sum())) * (e[meas] if prec == "jwst" else np.sqrt(e[meas] ** 2 + s[meas] ** 2)); x = B[q.row].copy(); x[filled] = x[meas].mean(); sig = np.where(filled, 1.0, e if prec == "jwst" else np.sqrt(e ** 2 + s ** 2)); t = Y[(Y.planet == pl) & (Y.row == q.row)].iloc[0]; r = M.loc[ML.BASE(pl)].to_dict(); r["name"] = key; r["T_p"] = float(t["T"]); r["cloud_pressure"] = 1e6; r["atm top_pressure"] = 1e-4; r["atm CO"] = -15.0; r["atm O3"] = -15.0
            for m in MOLS: r[f"atm {m}"] = float(t[f"v_{m}"])
            T.append((k, r, x, sig))
        print(prec, len(T), flush=True)
        for res in Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(_retrieve)(k, r, x, s, wl, edges, 100, 20000, SEED + 7000 + k, None, 10800, False, AB) for k, r, x, s in T):
            res["key"] = res["name"]; old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(); pd.concat([old, pd.DataFrame([res])], ignore_index=True).to_csv(out, index=False); print(res["key"], round(res.get("minutes", 0), 1), flush=True)
    summarize()


def summarize():
    Y = pd.read_csv(os.path.join(GR, "truth_crossfield.csv")); P = pairs(Y); L = []; Vf = pd.read_csv(os.path.join(GR, "votes_frozen_crossfield.csv.gz")).query('design=="the method"'); Ve = pd.read_csv(os.path.join(OUT, "votes_C+B-mlp_crossfield.csv.gz"))
    for prec, pp in (("jwst", "JWST"), ("tier1", "Tier-1")):
        f = os.path.join(OUT, f"grid_retrieval_{prec}.csv")
        if not os.path.exists(f): continue
        R = pd.read_csv(f); R = R[R[[f"{m}_p_present" for m in MOLS]].notna().all(1)]; R[["planet", "row"]] = R.key.str.split(":", expand=True); R["row"] = R.row.astype(int); rows = []
        for _, q in R.iterrows():
            t = Y[(Y.planet == q.planet) & (Y.row == q.row)].iloc[0]
            for m in MOLS: rows.append(dict(planet=q.planet, row=q.row, molecule=m, y=t[f"v_{m}"] > -4, present=float(q[f"{m}_p_present"] >= 0.5), minutes=q.minutes))
        A = pd.DataFrame(rows); bal = lambda g: 0.5 * (g[g.y].c.mean() + g[~g.y].c.mean())
        B = A.assign(c=np.where(A.y, A.present, 1 - A.present)); E = A[["planet", "row", "molecule", "y"]].merge(Ve[Ve.precision == pp], on=["planet", "row", "molecule"]); E["c"] = np.where(E.y, E.present, E.absent); F = A[["planet", "row", "molecule", "y"]].merge(Vf[Vf.precision == pp], on=["planet", "row", "molecule"]); F["c"] = np.where(F.y, F.present, F.absent) + 0.5 * (1 - F.present - F.absent)
        L.append(f"{pp}: {len(R)} spectra, {len(A)} labels, {A.y.mean() * 100:.0f} % present; median {R.minutes.median():.1f} min per spectrum, {R.minutes.sum() / 60:.0f} core-hours")
        for n, g in (("retrieval", B), ("matched ensemble", E), ("routing method", F)):
            L.append(f"  {n:<18} balanced {bal(g):.3f}  present {g[g.y].c.mean():.3f}  false-present {g[~g.y].present.mean():.3f}"); L += [f"     {m} {bal(g[g.molecule == m]):.3f}" for m in MOLS if g[g.molecule == m].y.nunique() == 2]
    t = "\n".join(L); open(os.path.join(OUT, "grid_retrieval.txt"), "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
