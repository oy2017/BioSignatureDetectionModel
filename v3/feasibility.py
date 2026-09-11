"""Feasibility gate for the C/O-label study (RESEARCH_PLAN.md section 7, item 2).

Answers three questions before anything is regenerated:
  1. Class balance at both candidate cuts (solar 0.55, carbon-rich 1.0).
  2. Is the C/O label learnable from an Ariel-like spectrum at all?
  3. THE CONFOUND: is the classifier reading C/O, or reading temperature?
     Under equilibrium the CH4/CO transition is driven by T as much as by C/O, so a
     screen could learn T from the continuum and report it as C/O. Tested three ways:
       (a) accuracy from bulk parameters alone, spectrum withheld  (the floor)
       (b) accuracy from the spectrum                               (the claim)
       (c) accuracy from the spectrum, per temperature band         (where it fails)

Composition comes from FastChem equilibrium (pyfastchem 4.0.3, Asplund 2009 solar
abundances, logK from the FastChem repo) at the planet's isothermal T and 10 mbar.
C/O and [M/H] are sampled independently. Bulk parameters are the recorded draws from
v2/data/test1_params.parquet so the grid design carries over unchanged.

Environment change this depends on: ~/exotransmit_src/Opac/opacCO.dat was copied into
MultiREx's data directory on 2026-09-11 so CO carries opacity (verified: +1.1e-4 mean
depth change in the 4.4-5.0 um band, ~0 outside). Without it a C/O label is not one.

Usage: python feasibility.py [--n 600] [--jobs 6]
Writes results/feasibility.txt and results/feasibility_params.parquet
"""
import argparse, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")   # XGBoost after a joblib pool otherwise thrashes
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), "v2")
sys.path.insert(0, V2)
from common import centres, configs, noise_spec, SNR            # noqa: E402
from bin_spectra import bin_native                              # noqa: E402
from noise import add_noise                                     # noqa: E402
from generate_grid import native_wavelengths, worker, chunked, GASES  # noqa: E402
from joblib import Parallel, delayed                            # noqa: E402
import pyfastchem                                               # noqa: E402

FC_DIR = os.path.expanduser("~/fastchem_input")
HILL = {"H2O": "H2O1", "CH4": "C1H4", "CO": "C1O1", "CO2": "C1O2", "NH3": "H3N1", "O3": "O3"}
P_CHEM_BAR = 1e-2
CO_RANGE, MH_RANGE = (0.2, 1.5), (-1.0, 1.5)
CUTS = {"solar": 0.55, "carbon-rich": 1.0}
T_BANDS = [(500, 1000), (1000, 1500), (1500, 2500)]


def equilibrium(T, co, mh, fc, base, idx, iC, iO, metals):
    ab = base.copy(); ab[metals] *= 10 ** mh; ab[iC] = ab[iO] * co
    fc.setElementAbundances(ab)
    inp, out = pyfastchem.FastChemInput(), pyfastchem.FastChemOutput()
    inp.temperature, inp.pressure = [float(T)], [P_CHEM_BAR]
    fc.calcDensities(inp, out)
    nd = np.array(out.number_densities)[0]; tot = nd.sum()
    return {g: float(np.log10(max(nd[idx[g]] / tot, 1e-30))) for g in HILL}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=6); a = ap.parse_args()
    rng = np.random.default_rng(3)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    # bulk draws reused from the v2 grid; chemistry parameters drawn independently
    P = pd.read_parquet(os.path.join(V2, "data", "test1_params.parquet")).sample(a.n, random_state=3).reset_index(drop=True)
    P["co_ratio"] = rng.uniform(*CO_RANGE, len(P)); P["mh"] = rng.uniform(*MH_RANGE, len(P))

    fc = pyfastchem.FastChem(os.path.join(FC_DIR, "asplund_2009.dat"), os.path.join(FC_DIR, "logK.dat"), 0)
    idx = {g: fc.getGasSpeciesIndex(h) for g, h in HILL.items()}
    base = np.array(fc.getElementAbundances()); iC, iO = fc.getElementIndex("C"), fc.getElementIndex("O")
    metals = [i for i in range(fc.getElementNumber()) if fc.getElementSymbol(i) not in ("H", "He")]
    t0 = time.time()
    chem = [equilibrium(r["atm temperature"], r.co_ratio, r.mh, fc, base, idx, iC, iO, metals) for _, r in P.iterrows()]
    for g in GASES: P[f"atm {g}"] = [c[g] for c in chem]
    print(f"chemistry for {len(P)} planets: {time.time()-t0:.1f} s", flush=True)

    wl = native_wavelengths(); rows = [P.iloc[i].to_dict() for i in range(len(P))]
    t0 = time.time()
    out = Parallel(n_jobs=a.jobs)(delayed(worker)(c, wl) for c in chunked(rows, 25))
    specs = [s for c in out for s in c]; ok = np.array([s is not None for s in specs])
    print(f"rendered {ok.sum()}/{len(P)} in {time.time()-t0:.0f} s", flush=True)
    Xn = np.vstack([s for s in specs if s is not None]); P = P[ok].reset_index(drop=True)

    edges = np.array(configs()["ariel"]["edges"]); cen = centres("ariel"); nshape, nlevel = noise_spec("ariel")
    Xb = bin_native(Xn, wl, edges)
    good = np.all(np.isfinite(Xb), axis=1); Xb, P = Xb[good], P[good].reset_index(drop=True)
    X, _ = add_noise(Xb, P, cen, snr=SNR, shape=nshape, seed=11, level_ppm=nlevel)
    Xs = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)   # per-spectrum normalization

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    T = P["atm temperature"].to_numpy()
    bulk = P[["p_radius", "p_mass", "atm temperature", "s radius", "s temperature", "sma"]].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    xgb = lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, verbosity=0, n_jobs=1)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    L = [f"C/O feasibility on {len(P)} planets, Ariel layout, peak-to-peak SNR {SNR}", ""]
    for name, cut in CUTS.items():
        y = (P.co_ratio > cut).astype(int).to_numpy()
        L += [f"=== cut: C/O > {cut} ({name}) ===", f"positive rate {y.mean():.3f}"]
        a_bulk = cross_val_score(xgb(), bulk, y, cv=cv).mean()
        a_T = cross_val_score(lr(), T[:, None], y, cv=cv).mean()
        a_spec = cross_val_score(xgb(), Xs, y, cv=cv).mean()
        a_spec_lr = cross_val_score(lr(), Xs, y, cv=cv).mean()
        L += [f"  (a) bulk params only, no spectrum : {100*a_bulk:5.1f}%   [T alone: {100*a_T:5.1f}%]   <- confound floor",
              f"  (b) spectrum, normalized, XGBoost  : {100*a_spec:5.1f}%   [logistic: {100*a_spec_lr:5.1f}%]",
              f"      spectrum beats bulk by {100*(a_spec-a_bulk):+.1f} points"]
        for lo, hi in T_BANDS:
            m = (T >= lo) & (T < hi)
            if m.sum() < 40 or y[m].min() == y[m].max():
                L.append(f"  (c) T {lo:>4}-{hi:<4} K: n={m.sum():3d}  (too few / one class)"); continue
            ab = cross_val_score(xgb(), Xs[m], y[m], cv=StratifiedKFold(4, shuffle=True, random_state=0)).mean()
            L.append(f"  (c) T {lo:>4}-{hi:<4} K: n={m.sum():3d}  pos {y[m].mean():.2f}  spectrum acc {100*ab:5.1f}%  majority {100*max(y[m].mean(),1-y[m].mean()):5.1f}%")
        L.append("")
    L += ["Reading: the label is usable where (b) clearly exceeds (a) AND (c) holds inside each",
          "temperature band. If (c) collapses to majority in the cool band, the C/O signature is",
          "not observable there and the label must be restricted or stratified by temperature."]
    txt = "\n".join(L); print(txt)
    open(os.path.join(HERE, "results", "feasibility.txt"), "w").write(txt + "\n")
    P.to_parquet(os.path.join(HERE, "results", "feasibility_params.parquet"))


if __name__ == "__main__":
    main()
