"""AD8 (REVISION_EXPECTATIONS.md): does pinning the tested molecule at the class boundary inflate dchi2?

identifiability.py's cheap fitting test holds molecule X at the boundary 1e-4 and re-optimises the nuisance
parameters (T, gray cloud-top pressure, the other three abundances). The objection: with the nuisance free, the best
opposite-class fit might lie away from the boundary. Here, on a fixed random 100 POP-I test planets (clean spectra,
Tier-1 radiometric sigma, the same planet set / edges / sigma as `identifiability.py --departure none`):

  1. the boundary minimisation is rerun with exactly the starts identifiability.py used (its per-chunk RNG streams
     are reconstructed, including the resume after the first 16 planets), and compared with the stored dchi2;
  2. the minimisation is repeated over the full opposite class: log10 X free over [-9, -4] for a present label,
     [-4, -2] for an absent one, jointly with the same nuisance box. Starts: the true parameters projected into the
     box (X lands on the boundary), the two random nuisance starts of step 1 with a random X inside the class, and
     the step-1 boundary optimum. Nelder-Mead with the same settings and maxfev 400 per start (4 starts, 1600 calls,
     against 3 x 400 = 1200 for the boundary). Since the boundary-optimum start evaluates the boundary solution
     first and Nelder-Mead returns its best point, full <= boundary(rerun) holds by construction.

Writes results/dchi2_full_class.csv (one row per planet x molecule; resumable) and results/dchi2_full_class.txt.
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/fanyang/tfenv/bin/python review_checks/dchi2_full_class.py --jobs 3
"""
import argparse
import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import minimize

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, V3)
import identifiability as ID  # noqa: E402
from common import DATA, RESULTS, SEED  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import generate_grid as G  # noqa: E402

SEL_SEED = 808                     # selection of the 100 planets
MOLRNG_SEED = 8080                 # random X coordinate of the full-class random starts
N_PLANETS, STARTS, MAXFEV = 100, 3, 400
RESUME_AT = 16                     # identifiability_none_tier1 was restarted after its first 16 planets (see its log)
THR = {"CH4": 0.45, "H2O": 0.98, "CO2": 0.90, "NH3": 0.56}
OUT_CSV = os.path.join(RESULTS, "dchi2_full_class.csv"); OUT_TXT = os.path.join(RESULTS, "dchi2_full_class.txt")


def load():
    """Same planet set, sigma and edges as identifiability.py main() with --departure none --noise tier1."""
    src = AF.DIRS["faithful"]; wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = AF.layout_edges("tier3_r20")
    Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet"))
    Y = np.load(os.path.join(src, "pop1_native.npy")).astype(np.float64)
    ok = np.all(np.isfinite(Y), axis=1)
    Pte = Pte[ok].reset_index(drop=False).rename(columns={"index": "_index"}); Y = Y[ok]
    mcs = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet"))[["name", "atm temperature"]].rename(columns={"atm temperature": "T_p"})
    Pte = Pte.merge(mcs.drop_duplicates("name"), on="name", how="left"); assert Pte.T_p.notna().all()
    sig = AF.sigma(Pte, edges, "radiometric")
    return Pte, Y, sig, wl, edges


def stored_rng(p):
    """The RNG state identifiability.py's worker had when it reached Pte position p (chunk 2, per-run todo offsets)."""
    j = p if p < RESUME_AT else p - RESUME_AT
    c0 = j - j % 2; rng = np.random.default_rng(SEED + c0)
    if j % 2:                                       # the chunk's first planet consumed 4 molecules x (STARTS-1) draws
        for _ in range(len(ID.MOLS) * (STARTS - 1)):
            rng.uniform(0, 1, 5)
    return rng


def full_class(pl, mol, rand_u5, bx, mrng):
    """Minimum offset-free chi^2 over the whole opposite class (X free inside it), nuisance re-optimised."""
    others = [m for m in ID.MOLS if m != mol]; row = pl.row
    present = float(row[f"atm {mol}"]) > ID.THRESH
    mlo, mhi = (ID.AB_RANGE[0], ID.THRESH) if present else (ID.THRESH, ID.AB_RANGE[1])
    t0 = float(row["atm temperature"]); lp0 = np.log10(float(row["cloud_pressure"]))
    x_true = np.array([t0, lp0] + [float(row[f"atm {m}"]) for m in others] + [float(row[f"atm {mol}"])])
    lo = np.array([ID.T_RANGE[0] * pl.T_p, ID.LOGP_RANGE[0]] + [ID.AB_RANGE[0]] * 3 + [mlo])
    hi = np.array([ID.T_RANGE[1] * pl.T_p, ID.LOGP_RANGE[1]] + [ID.AB_RANGE[1]] * 3 + [mhi])
    scale = hi - lo

    def f(u):
        x = lo + np.clip(u, 0, 1) * scale
        ab = {m: x[2 + k] for k, m in enumerate(others)}; ab[mol] = x[5]
        y = pl.render(x[0], x[1], ab)
        return 1e12 if y is None else ID.chi2_offset(pl.truth - y, pl.sigma)
    u_b = 1.0 if present else 0.0                    # the boundary in the X coordinate
    u0s = [("true", np.clip((x_true - lo) / scale, 0, 1))]
    u0s += [(f"rand{k}", np.append(u, mrng.uniform(0, 1))) for k, u in enumerate(rand_u5)]
    u0s += [("boundary_opt", np.append((bx - lo[:5]) / scale[:5], u_b))]
    best, bxf, bstart, per = np.inf, None, None, {}
    for name, u0 in u0s:
        u0 = np.clip(u0, 0, 1)
        res = minimize(f, u0, method="Nelder-Mead",
                       options=dict(xatol=1e-3, fatol=1e-3, maxfev=MAXFEV, initial_simplex=ID.Planet._simplex(u0)))
        per[name] = float(res.fun)
        if res.fun < best:
            best, bxf, bstart = float(res.fun), lo + np.clip(res.x, 0, 1) * scale, name
    return best, bxf, bstart, per


def run_planet(p, row, truth, sig, wl, edges):
    import warnings; warnings.filterwarnings("ignore")
    G.FILL_GAS = AF.FILL
    pl = ID.Planet(row, truth, wl, edges, sig, "none", list(G.GASES))
    rng = stored_rng(p); mrng = np.random.default_rng(MOLRNG_SEED + p); out = []
    for mol in ID.MOLS:
        peek = copy.deepcopy(rng); rand_u5 = [peek.uniform(0, 1, 5) for _ in range(STARTS - 1)]
        t0 = time.time(); pl.calls = 0
        db, bx = pl.dchi2(mol, STARTS, rng); cb = pl.calls; tb = time.time() - t0
        t0 = time.time(); pl.calls = 0
        df_, xf, start, per = full_class(pl, mol, rand_u5, bx, mrng); cf = pl.calls; tf = time.time() - t0
        out.append(dict(pos=p, index=int(row["_index"]), name=row["name"], molecule=mol, log_true=float(row[f"atm {mol}"]),
                        label=int(float(row[f"atm {mol}"]) > ID.THRESH), dchi2_boundary=db, dchi2_full=df_,
                        full_logX=xf[5], full_T=xf[0], full_logP=xf[1], boundary_T=bx[0], boundary_logP=bx[1],
                        full_best_start=start, **{f"full_{k}": v for k, v in per.items()},
                        calls_boundary=cb, calls_full=cf, sec_boundary=tb, sec_full=tf))
    return out


def summarize(df, stored, wall=None, write=True):
    L = [f"AD8: boundary vs full-opposite-class dchi2, POP-I clean, Tier-1 radiometric sigma, {df['index'].nunique()} planets "
         f"({len(df)} labels), selection seed {SEL_SEED}", ""]
    m = df.merge(stored[["index", "molecule", "dchi2"]].rename(columns={"dchi2": "dchi2_stored"}), on=["index", "molecule"])
    r = m.dchi2_boundary - m.dchi2_stored; rel = r / np.maximum(m.dchi2_stored, 1e-9)
    L += ["== 1. reproduction of the stored boundary dchi2 (results/identifiability_none_tier1.csv), same starts",
          f"  rerun - stored: median {r.median():+.2e}, |diff| max {r.abs().max():.3g}; |diff| < 1e-6 on {(r.abs() < 1e-6).mean()*100:.1f} %, "
          f"< 0.01 on {(r.abs() < 0.01).mean()*100:.1f} %, < 0.1 on {(r.abs() < 0.1).mean()*100:.1f} %; |rel| < 1 % on {(rel.abs() < 0.01).mean()*100:.1f} %",
          f"  Spearman(rerun, stored) {m.dchi2_boundary.corr(m.dchi2_stored, method='spearman'):.4f}", ""]
    worst = m.assign(diff=r).reindex(r.abs().sort_values(ascending=False).index).head(5)
    L += ["  largest |rerun - stored|:"] + ["    " + s for s in worst[["name", "molecule", "dchi2_stored", "dchi2_boundary", "diff"]].to_string(index=False).split("\n")] + [""]
    d = df.dchi2_boundary - df.dchi2_full
    L += ["== 2. boundary - full-class dchi2 (>= 0 by construction)",
          f"  min {d.min():.3g}  median {d.median():.3g}  mean {d.mean():.3g}  90th pct {d.quantile(.9):.3g}  99th pct {d.quantile(.99):.3g}  max {d.max():.3g}",
          f"  fraction > 0.1: {(d > 0.1).mean()*100:.1f} %   > 1: {(d > 1).mean()*100:.1f} %   (AD8 expects > 0.1 on at most 10 %)",
          f"  relative (boundary - full)/boundary: median {(d/np.maximum(df.dchi2_boundary,1e-9)).median():.3g}, > 10 % on "
          f"{((d/np.maximum(df.dchi2_boundary,1e-9)) > 0.1).mean()*100:.1f} %"]
    for mol in ID.MOLS:
        g = df.molecule == mol; dd = d[g]
        L.append(f"  {mol:<4} n {g.sum():3d}  median {dd.median():.3g}  max {dd.max():8.3g}  > 0.1 {(dd > 0.1).mean()*100:5.1f} %  > 1 {(dd > 1).mean()*100:5.1f} %"
                 f"  | present > 0.1 {(dd[df.label[g] == 1] > 0.1).mean()*100:5.1f} %  absent > 0.1 {(dd[df.label[g] == 0] > 0.1).mean()*100:5.1f} %")
    for lab, nm in ((1, "present (class [-9,-4])"), (0, "absent (class [-4,-2])")):
        dd = d[df.label == lab]; L.append(f"  {nm:<24} n {len(dd):3d}  median {dd.median():.3g}  > 0.1 {(dd > 0.1).mean()*100:5.1f} %  > 1 {(dd > 1).mean()*100:5.1f} %")
    L.append("")
    dist = (df.full_logX - ID.THRESH).abs()
    L += ["== 3. where the full-class optimum lies (|log10 X - (-4)|)",
          f"  at the boundary (< 0.01 dex) {(dist < 0.01).mean()*100:.1f} %   within 0.1 dex {(dist < 0.1).mean()*100:.1f} %   interior > 0.5 dex {(dist > 0.5).mean()*100:.1f} %   > 1 dex {(dist > 1).mean()*100:.1f} %",
          f"  among labels with boundary - full > 0.1: at boundary (<0.01) {(dist[d > 0.1] < 0.01).mean()*100:.1f} %, median distance {dist[d > 0.1].median():.3g} dex",
          f"  among labels with boundary - full > 0.1 whose optimum is at the boundary (gain from the extra nuisance search, not from moving X): "
          f"{((d > 0.1) & (dist < 0.01)).sum()}",
          "  winning start: " + ", ".join(f"{k} {v}" for k, v in df.full_best_start.value_counts().items())]
    for mol in ID.MOLS:
        g = df.molecule == mol
        L.append(f"  {mol:<4} at boundary {(dist[g] < 0.01).mean()*100:5.1f} %   median distance {dist[g].median():.3g} dex   max {dist[g].max():.3g} dex")
    L.append("")
    thr = df.molecule.map(THR); vb = df.dchi2_boundary > thr; vf = df.dchi2_full > thr
    ch = vb != vf
    L += ["== 4. verdict changes at the calibrated thresholds (decidable if dchi2 > thr: CH4 0.45, H2O 0.98, CO2 0.90, NH3 0.56)",
          f"  boundary decidable {vb.mean()*100:.1f} %, full-class decidable {vf.mean()*100:.1f} %; changed {ch.sum()} / {len(df)} = {ch.mean()*100:.1f} %"
          f"  (AD8 expects at most 5 %); all changes are decidable -> undecidable: {bool((vb[ch] & ~vf[ch]).all())}"]
    vs = m.dchi2_stored > m.molecule.map(THR); vr = m.dchi2_boundary > m.molecule.map(THR)
    L.append(f"  stored vs rerun boundary verdict changes (optimiser noise): {(vs != vr).sum()} / {len(m)}")
    for mol in ID.MOLS:
        g = df.molecule == mol; L.append(f"  {mol:<4} changed {ch[g].sum()} / {g.sum()}")
    if ch.any():
        L += ["  changed labels:"] + ["    " + s for s in df[ch][["name", "molecule", "label", "log_true", "dchi2_boundary", "dchi2_full", "full_logX"]].to_string(index=False).split("\n")]
    L += ["", f"== cost: median calls boundary {df.calls_boundary.median():.0f}, full {df.calls_full.median():.0f}; "
          f"median s per label boundary {df.sec_boundary.median():.0f}, full {df.sec_full.median():.0f}"
          + (f"; wall {wall/3600:.2f} h this session" if wall else "")]
    if write:
        open(OUT_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", action="store_true"); a = ap.parse_args()
    stored = pd.read_csv(os.path.join(RESULTS, "identifiability_none_tier1.csv"))
    if a.summarize:
        summarize(pd.read_csv(OUT_CSV), stored); return
    Pte, Y, sig, wl, edges = load()
    assert len(Pte) == stored["index"].nunique() and (Pte["_index"].to_numpy() == stored["index"].drop_duplicates().to_numpy()).all()
    pos = np.sort(np.random.default_rng(SEL_SEED).choice(len(Pte), N_PLANETS, replace=False))
    if a.limit:
        pos = pos[:a.limit]
    done = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) and not a.limit else pd.DataFrame()
    todo = [p for p in pos if done.empty or p not in set(done.pos)]
    print(f"{len(pos)} planets, {len(todo)} to do, {a.jobs} jobs", flush=True); t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_planet, int(p), Pte.iloc[p].to_dict(), Y[p], sig[p], wl, edges) for p in todo]
        for k, fu in enumerate(as_completed(futs)):
            new = pd.DataFrame(fu.result()); done = pd.concat([done, new], ignore_index=True)
            if not a.limit:
                done.to_csv(OUT_CSV, index=False)
            print(f"  {k+1}/{len(todo)} planets, {time.time()-t0:.0f} s; {new.name.iloc[0]}: boundary-full "
                  + " ".join(f"{r.molecule} {r.dchi2_boundary - r.dchi2_full:.3g}" for r in new.itertuples()), flush=True)
    if a.limit:
        print(done.drop(columns=["name"]).to_string());
    summarize(done, stored, time.time() - t0)


if __name__ == "__main__":
    main()
