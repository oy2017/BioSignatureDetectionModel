"""Regenerate the grid with equilibrium-chemistry composition and a C/O label (plan section 7, item 4).

What is kept from v2, deliberately: the bulk-parameter draws and the five test splits are
read back from v2/data/{split}_params.parquet, so every planet is the same planet with a
different atmosphere, and the independence properties and split design carry over.

What changes:
  * composition: FastChem equilibrium at the planet's isothermal T and P_CHEM_BAR, from
    C/O ~ U(0.2, 1.8) and [M/H] ~ U(-1, 1.5) drawn independently per planet.
    U(0.2, 1.8) puts the carbon-rich cut C/O > 1.0 at the median.
  * label = (C/O > 1.0).  Invariant under every shift axis because it is a chemistry input.
  * CO carries opacity (opacCO.dat placed in MultiREx's data dir on 2026-09-11).
  * O3 is floored at log10 = -15: equilibrium gives ~1e-50 in hydrogen, MultiREx wants a number.
  * mode="quenched" is a HOOK (Axis 8): not implemented yet; raises so nothing silently runs
    equilibrium under that name.

Usage:
  python generate_grid.py --smoke 40          # render 40 planets from test1, report balance
  python generate_grid.py --splits train test1 ... --jobs 12
Outputs (v3/data/): {split}_params.parquet, {split}_native.npy, native_wl.npy
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), "v2")
sys.path.insert(0, V2)
from generate_grid import build_system, native_wavelengths, one_native, chunked, GASES, BULK  # noqa: E402
import pyfastchem  # noqa: E402

DATA = os.path.join(HERE, "data")
FC_DIR = os.path.expanduser("~/fastchem_input")
HILL = {"H2O": "H2O1", "CH4": "C1H4", "CO": "C1O1", "CO2": "C1O2", "NH3": "H3N1", "O3": "O3"}
P_CHEM_BAR = 1e-2
CO_RANGE, MH_RANGE, CO_CUT = (0.2, 1.8), (-1.0, 1.5), 1.0
LOG_FLOOR = -15.0
SPLITS = ["train", "test1", "test2", "test3", "test4", "test5"]


class Chemistry:
    def __init__(self):
        self.fc = pyfastchem.FastChem(os.path.join(FC_DIR, "asplund_2009.dat"), os.path.join(FC_DIR, "logK.dat"), 0)
        fc = self.fc
        self.idx = {g: fc.getGasSpeciesIndex(h) for g, h in HILL.items()}
        self.base = np.array(fc.getElementAbundances())
        self.iC, self.iO = fc.getElementIndex("C"), fc.getElementIndex("O")
        self.metals = [i for i in range(fc.getElementNumber()) if fc.getElementSymbol(i) not in ("H", "He")]

    def composition(self, T, co, mh, mode="equilibrium"):
        if mode != "equilibrium":
            raise NotImplementedError("Axis 8 (quenched) is a hook; implement before use")
        ab = self.base.copy(); ab[self.metals] *= 10 ** mh; ab[self.iC] = ab[self.iO] * co
        self.fc.setElementAbundances(ab)
        inp, out = pyfastchem.FastChemInput(), pyfastchem.FastChemOutput()
        inp.temperature, inp.pressure = [float(T)], [P_CHEM_BAR]
        self.fc.calcDensities(inp, out)
        nd = np.array(out.number_densities)[0]; tot = nd.sum()
        return {g: float(max(np.log10(max(nd[i] / tot, 1e-300)), LOG_FLOOR)) for g, i in self.idx.items()}


def worker(rows, wl):
    import warnings; warnings.filterwarnings("ignore")
    return [one_native(r, wl) for r in rows]


def make_split(name, seed, jobs, wl, limit=None):
    P = pd.read_parquet(os.path.join(V2, "data", f"{name}_params.parquet"))
    P = P[list(BULK) + ["atm fill_gas"]].copy() if "atm fill_gas" in P else P[list(BULK)].copy()
    if limit: P = P.iloc[:limit].copy()
    rng = np.random.default_rng(seed)
    P["co_ratio"] = rng.uniform(*CO_RANGE, len(P)); P["mh"] = rng.uniform(*MH_RANGE, len(P))
    chem = Chemistry(); t0 = time.time()
    comp = [chem.composition(r["atm temperature"], r.co_ratio, r.mh) for _, r in P.iterrows()]
    for g in GASES: P[f"atm {g}"] = [c[g] for c in comp]
    P["atm fill_gas"] = "H2"; P["label_co"] = (P.co_ratio > CO_CUT).astype(int)
    print(f"{name}: chemistry for {len(P)} in {time.time()-t0:.1f} s", flush=True)
    rows = [P.iloc[i].to_dict() for i in range(len(P))]; t0 = time.time()
    out = Parallel(n_jobs=jobs)(delayed(worker)(c, wl) for c in chunked(rows, 50))
    specs = [s for c in out for s in c]; ok = np.array([s is not None for s in specs])
    X = np.vstack([s for s in specs if s is not None]); kept = P[ok].reset_index(drop=True)
    print(f"{name}: {ok.sum()}/{len(P)} rendered ({100*ok.mean():.1f}%), label balance {kept.label_co.mean():.3f}, {time.time()-t0:.0f} s", flush=True)
    return kept, X


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--jobs", type=int, default=12); ap.add_argument("--smoke", type=int, default=0); a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    wl = native_wavelengths(); np.save(os.path.join(DATA, "native_wl.npy"), wl)
    if a.smoke:
        kept, X = make_split("test1", 3, min(a.jobs, 6), wl, limit=a.smoke)
        print("smoke: log10 abundance medians", {g: round(float(kept[f"atm {g}"].median()), 2) for g in GASES})
        print("smoke: native spectrum shape", X.shape, "depth range", f"{X.min():.3e}-{X.max():.3e}")
        return
    for k, s in enumerate(a.splits):
        kept, X = make_split(s, 100 + k, a.jobs, wl)
        kept.to_parquet(os.path.join(DATA, f"{s}_params.parquet")); np.save(os.path.join(DATA, f"{s}_native.npy"), X)


if __name__ == "__main__":
    main()
