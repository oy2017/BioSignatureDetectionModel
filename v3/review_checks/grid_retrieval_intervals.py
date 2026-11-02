"""Intervals for U7 (REVISION_EXPECTATIONS.md): retrieval against the matched ensemble and the routing method on the photochemical grid, bootstrap over planets, on the labels the retrieval finished and with each timed-out spectrum counted as no verdict (0.5 a label). Writes results/round3/grid_retrieval_intervals.txt."""
import os, sys, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = "_tp" if "--tfix" in sys.argv else ""; MOLS = ["CH4", "H2O", "CO2", "NH3"]; G = "results/goyal_test"; O = "results/round3"
Y = pd.read_csv(f"{G}/truth_crossfield.csv"); Ve = pd.read_csv(f"{O}/votes_C+B-mlp_crossfield.csv.gz"); Vf = pd.read_csv(f"{G}/votes_frozen_crossfield.csv.gz").query('design=="the method"')
def table(prec, pp):
    R = pd.read_csv(f"{O}/grid_retrieval{TAG}_{prec}.csv"); R[["planet", "row"]] = R.key.str.split(":", expand=True); R["row"] = R.row.astype(int); rows = []
    for _, q in R.iterrows():
        t = Y[(Y.planet == q.planet) & (Y.row == q.row)].iloc[0]; done = bool(pd.notna(q[[f"{m}_p_present" for m in MOLS]]).all())
        for m in MOLS: rows.append(dict(planet=q.planet, row=q.row, molecule=m, y=t[f"v_{m}"] > -4, done=done, ret=float(q[f"{m}_p_present"] >= 0.5) if done else np.nan))
    A = pd.DataFrame(rows); A["c_ret"] = np.where(A.done, np.where(A.y, A.ret, 1 - A.ret), 0.5)
    E = Ve[Ve.precision == pp][["planet", "row", "molecule", "present", "absent"]]; A = A.merge(E, on=["planet", "row", "molecule"]); A["c_ens"] = np.where(A.y, A.present, A.absent)
    F = Vf[Vf.precision == pp][["planet", "row", "molecule", "present", "absent"]].rename(columns={"present": "fp", "absent": "fa"}); A = A.merge(F, on=["planet", "row", "molecule"]); A["c_rou"] = np.where(A.y, A.fp, A.fa) + 0.5 * (1 - A.fp - A.fa)
    return A
def bal(g, c): return 0.5 * (g[g.y][c].mean() + g[~g.y][c].mean())
L = []; rng = np.random.default_rng(411)
for prec, pp in (("jwst", "JWST"), ("tier1", "Tier-1")):
    A0 = table(prec, pp)
    for tag, A in (("finished spectra", A0[A0.done]), ("all 90, timeouts as no verdict", A0)):
        units = A.planet.unique(); idx = {u: A.index[A.planet == u].to_numpy() for u in units}; nsp = A[["planet", "row"]].drop_duplicates().shape[0]
        pt = {c: bal(A, c) for c in ("c_ret", "c_ens", "c_rou")}; bs = []
        for _ in range(2000):
            s = A.loc[np.concatenate([idx[u] for u in rng.choice(units, len(units))])]; bs.append([bal(s, "c_ens") - bal(s, "c_ret"), bal(s, "c_rou") - bal(s, "c_ret")])
        bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        L.append(f"{pp:<7}{tag:<32}{nsp} spectra, {len(units)} planets: retrieval {pt['c_ret']:.3f}  ensemble {pt['c_ens']:.3f}  routing {pt['c_rou']:.3f}  ensemble - retrieval {pt['c_ens'] - pt['c_ret']:+.3f} ({lo[0]:+.3f}, {hi[0]:+.3f})  routing - retrieval {pt['c_rou'] - pt['c_ret']:+.3f} ({lo[1]:+.3f}, {hi[1]:+.3f})")
    T = A0[~A0.done]
    if len(T): L.append(f"{pp:<7}timed-out spectra only: {T[['planet','row']].drop_duplicates().shape[0]}, ensemble {bal(T, 'c_ens'):.3f}, routing {bal(T, 'c_rou'):.3f}")
t = "\n".join(L); open(f"{O}/grid_retrieval{TAG}_intervals.txt", "w").write(t + "\n"); print(t)
