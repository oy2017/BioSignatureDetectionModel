"""AD4 (REVISION_EXPECTATIONS.md): sensitivity of the held-out grids' truth labels to how "present" is defined.
Baseline (crossfield_test.load, goyal_test.chem): arithmetic mean of the mole fraction over the profile layers with 1e-4 <= p <= 0.1 bar (10-10^4 Pa), log10, present when > -4.
Alternatives: windows 1-10^3 Pa (1e-5..1e-2 bar) and 10^2-10^5 Pa (1e-3..1 bar), and the mean of log10 mole fraction over 10-10^4 Pa.
Both grids store their profiles (Crossfield vertical-profile csv, ATMO local-condensation chem files), so both are recomputed.
Rescored: matched ensemble (results/round3/votes_C+B-mlp_<grid>), routing method (votes_frozen, design "the method", split = 0.5),
and on the photochemical grid the four-gas retrieval on its 90 spectra (first run and temperature-prior rerun).
Writes results/round3/grid_truth_windows.txt."""
import gzip
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOLS = ["CH4", "H2O", "CO2", "NH3"]; G = "results/goyal_test"; O = "results/round3"; SEED = 4104; NB = 2000
WIN = {"base 10-1e4 Pa": (1e-4, 0.1, "lin"), "1-1e3 Pa": (1e-5, 1e-2, "lin"), "1e2-1e5 Pa": (1e-3, 1.0, "lin"), "log-mean 10-1e4 Pa": (1e-4, 0.1, "log")}
CF_COL = {"H2O": "H2O_HITEMP", "CH4": "CH4", "CO2": "CO2", "NH3": "NH3"}; GY_COL = {"H2O": 10, "CH4": 13, "CO2": 16, "NH3": 69}
BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")


def cf_file(f):
    V = pd.read_csv(os.path.join("data", "crossfield2025", "x", f, "mtgrid_helios_vertical_profiles.csv")); out = {}
    for w, (a, b, kind) in WIN.items():
        k = (V.p >= a) & (V.p <= b)
        for m, c in CF_COL.items():
            x = V.loc[k, c].to_numpy(float)
            out[(w, m)] = float(np.log10(np.clip(x.mean(), 1e-30, None))) if kind == "lin" else float(np.mean(np.log10(np.clip(x, 1e-30, None))))
    return f, out


def gy_file(f):
    with gzip.open(os.path.join("data", "goyal2019", "x", "chemfiles_local", f.replace("trans-", "chem-")), "rt") as h: A = np.loadtxt(h)
    out = {}
    for w, (a, b, kind) in WIN.items():
        k = (A[:, 0] >= a) & (A[:, 0] <= b)
        for m, c in GY_COL.items():
            x = np.clip(A[k, c], 1e-30, None)
            out[(w, m)] = float(np.log10(x.mean())) if kind == "lin" else float(np.mean(np.log10(x)))
    return f, out


def truths(grid):
    Y = pd.read_csv(f"{G}/truth_{grid}.csv"); fn = cf_file if grid == "crossfield" else gy_file
    R = dict(Parallel(n_jobs=4)(delayed(fn)(f) for f in sorted(Y.file.unique())))
    rows = []
    for _, q in Y.iterrows():
        for m in MOLS:
            d = dict(planet=q.planet, row=q.row, molecule=m, stored=q[f"v_{m}"]); d.update({w: R[q.file][(w, m)] for w in WIN}); rows.append(d)
    return pd.DataFrame(rows)


def bal(y, c):
    return 0.5 * (c[y].mean() + c[~y].mean())


def boot_diff(A, ca, cb, y, rng):
    """Planet-bootstrap interval (units = base planet) of bal(ca) - bal(cb)."""
    u = A.planet.map(BASE).to_numpy(); us = np.unique(u); ix = [np.flatnonzero(u == v) for v in us]; out = []
    for _ in range(NB):
        s = np.concatenate([ix[i] for i in rng.integers(0, len(us), len(us))]); out.append(bal(y[s], ca[s]) - bal(y[s], cb[s]))
    return np.percentile(out, [2.5, 97.5])


def main():
    L = []; rng = np.random.default_rng(SEED)
    Ve = {g: pd.read_csv(f"{O}/votes_C+B-mlp_{g}.csv.gz") for g in ("local", "crossfield")}
    Vf = {g: pd.read_csv(f"{G}/votes_frozen_{g}.csv.gz").query('design=="the method"') for g in ("local", "crossfield")}
    TR = {}
    for grid, name in (("local", "ATMO local condensation"), ("crossfield", "photochemical (Crossfield)")):
        T = truths(grid); TR[grid] = T; b = T["base 10-1e4 Pa"]
        d = np.abs(b - T.stored); same_lab = ((b > -4) == (T.stored > -4)).mean()
        L.append(f"=== {name}: {len(T)} labels ({T[['planet','row']].drop_duplicates().shape[0]} spectra), {(T.stored > -4).mean() * 100:.1f} % present under the stored truth")
        L.append(f"  reproduction of stored v_<gas> from the profiles (all rows): max |diff| {d.max():.2e} dex, exact (==) {np.mean(b.to_numpy() == T.stored.to_numpy()) * 100:.1f} %, labels identical {same_lab * 100:.2f} %")
        y0 = T.stored.to_numpy() > -4; near = np.abs(T.stored.to_numpy() + 4) < 0.5
        L.append(f"  labels within 0.5 dex of 1e-4 under the baseline: {near.mean() * 100:.1f} %")
        for w in list(WIN)[1:]:
            y1 = T[w].to_numpy() > -4; fl = y0 != y1
            byg = ", ".join(f"{m} {fl[T.molecule == m].mean() * 100:.1f} % ({(fl & y0 & (T.molecule == m)).sum()} P->A, {(fl & ~y0 & (T.molecule == m)).sum()} A->P)" for m in MOLS)
            L.append(f"  {w:<20} flips {fl.mean() * 100:.2f} % ({fl.sum()}), of which within 0.5 dex of 1e-4 in the baseline {(fl & near).sum() / max(fl.sum(), 1) * 100:.0f} %; present {y1.mean() * 100:.1f} %")
            L.append(f"      by gas: {byg}")
        L.append("")
    # full-grid rescoring
    L.append("=== Full-grid balanced credit (pooled labels; split/no-verdict = 0.5; interval = 2000 planet-bootstrap draws)")
    for grid in ("local", "crossfield"):
        T = TR[grid]
        for pp in ("JWST", "Tier-1"):
            E = Ve[grid][Ve[grid].precision == pp][["planet", "row", "molecule", "present", "absent"]]; F = Vf[grid][Vf[grid].precision == pp][["planet", "row", "molecule", "present", "absent"]].rename(columns={"present": "fp", "absent": "fa"})
            A = T.merge(E, on=["planet", "row", "molecule"]).merge(F, on=["planet", "row", "molecule"]); assert len(A) == len(E) == len(F)
            base_g = None
            for w in ["stored"] + list(WIN):
                y = (A[w] > -4).to_numpy(); ce = np.where(y, A.present, A.absent) + 0.5 * (1 - A.present - A.absent); cr = np.where(y, A.fp, A.fa) + 0.5 * (1 - A.fp - A.fa)
                se, sr = bal(y, ce), bal(y, cr); g = se - sr; base_g = g if w == "stored" else base_g; lo, hi = boot_diff(A, ce, cr, y, rng)
                L.append(f"  {grid:<10} {pp:<6} {w:<20} present {y.mean() * 100:4.1f} %  ensemble {se:.3f}  routing {sr:.3f}  gain {g:+.3f} ({lo:+.3f},{hi:+.3f})  shift {g - base_g:+.3f}  ens false-present {A.present[~y].mean():.3f}  ens recall {A.present[y].mean():.3f}")
    L.append("")
    # retrieval on 90 photochemical spectra
    L.append("=== Photochemical grid, four-gas retrieval on its 90 spectra (present when <gas>_p_present >= 0.5); finished spectra / all 90 with timeouts = 0.5")
    T = TR["crossfield"]
    for run, tag in (("first run", ""), ("T-prior rerun", "_tp")):
        for prec, pp in (("jwst", "JWST"), ("tier1", "Tier-1")):
            R = pd.read_csv(f"{O}/grid_retrieval{tag}_{prec}.csv"); R[["planet", "row"]] = R.key.str.split(":", expand=True); R["row"] = R.row.astype(int); R["done"] = R[[f"{m}_p_present" for m in MOLS]].notna().all(1)
            Rl = R.melt(id_vars=["planet", "row", "done"], value_vars=[f"{m}_p_present" for m in MOLS], var_name="molecule", value_name="pp"); Rl["molecule"] = Rl.molecule.str.replace("_p_present", "")
            E = Ve["crossfield"][Ve["crossfield"].precision == pp][["planet", "row", "molecule", "present", "absent"]]; F = Vf["crossfield"][Vf["crossfield"].precision == pp][["planet", "row", "molecule", "present", "absent"]].rename(columns={"present": "fp", "absent": "fa"})
            A0 = Rl.merge(T, on=["planet", "row", "molecule"]).merge(E, on=["planet", "row", "molecule"]).merge(F, on=["planet", "row", "molecule"]); assert len(A0) == 4 * len(R)
            for sub, A in (("finished", A0[A0.done].reset_index(drop=True)), ("all 90", A0)):
                if sub == "all 90" and A0.done.all(): continue
                base_gap = None; nsp = A[["planet", "row"]].drop_duplicates().shape[0]
                for w in ["stored"] + list(WIN):
                    y = (A[w] > -4).to_numpy(); ret = (A.pp >= 0.5).astype(float).to_numpy(); ct = np.where(A.done, np.where(y, ret, 1 - ret), 0.5)
                    ce = np.where(y, A.present, A.absent) + 0.5 * (1 - A.present - A.absent); cr = np.where(y, A.fp, A.fa) + 0.5 * (1 - A.fp - A.fa)
                    st, se, sr = bal(y, ct), bal(y, ce), bal(y, cr); gap = se - st; base_gap = gap if w == "stored" else base_gap; lo, hi = boot_diff(A, ce, ct, y, rng)
                    L.append(f"  {run:<13} {pp:<6} {sub:<8} {nsp:>2} sp {w:<20} present {y.mean() * 100:4.1f} %  retrieval {st:.3f}  ensemble {se:.3f}  routing {sr:.3f}  ens-ret {gap:+.3f} ({lo:+.3f},{hi:+.3f})  shift {gap - base_gap:+.3f}  ens-routing {se - sr:+.3f}")
    t = "\n".join(L); open(f"{O}/grid_truth_windows.txt", "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
