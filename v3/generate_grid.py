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
# load v2's generator by path under its own name: importing it as `generate_grid` would
# collide with this module whenever this file is imported rather than run
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("v2_generate_grid", os.path.join(V2, "generate_grid.py"))
_v2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_v2)
build_system, native_wavelengths, one_native, chunked, GASES, BULK = (
    _v2.build_system, _v2.native_wavelengths, _v2.one_native, _v2.chunked, _v2.GASES, _v2.BULK)
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

    def _equilibrium_at(self, T, P_bar, co, mh):
        ab = self.base.copy(); ab[self.metals] *= 10 ** mh; ab[self.iC] = ab[self.iO] * co
        self.fc.setElementAbundances(ab)
        inp, out = pyfastchem.FastChemInput(), pyfastchem.FastChemOutput()
        inp.temperature, inp.pressure = [float(T)], [float(P_bar)]
        self.fc.calcDensities(inp, out)
        nd = np.array(out.number_densities)[0]; tot = nd.sum()
        return {g: float(max(np.log10(max(nd[i] / tot, 1e-300)), LOG_FLOOR)) for g, i in self.idx.items()}

    def composition(self, T, co, mh, mode="equilibrium", g_cgs=None):
        """mode='equilibrium': FastChem at the isothermal T and P_CHEM_BAR (the primary grid).

        mode='quenched' (Axis 8). Quenching is a property of the T-P profile, which an isothermal
        model does not have, so the chemistry step builds one: a Guillot (2010) radiative profile
        (kappa_th = 1e-2 cm^2/g, gamma = 0.4, T_int = 100 K, f = 1/4) RESCALED so that its temperature
        at P_CHEM_BAR equals the planet's T -- which makes the equilibrium composition at that level
        identical to the primary grid's by construction, so the axis isolates quenching alone -- with
        a convective adiabat (T ~ P^(2/7), H2-dominated) below the radiative-convective boundary, since
        that hot interior is where the CO carried up into cool atmospheres comes from. The CO->CH4
        conversion time of Zahnle & Marley (2014), t_chem = 1.5e-6 P^-1 exp(42000/T) s (P in bar), is
        compared with t_mix = H^2 / K_zz, K_zz = 1e9 cm^2/s, and the carbon/oxygen partitioning is
        frozen at the shallowest level where chemistry still keeps up (the quench point). If that level
        lies above P_CHEM_BAR the composition is unchanged. The spectrum is still rendered isothermal
        with the quenched composition. The profile parameters and K_zz are standard placeholders and
        are disclosed as such.
        """
        if mode == "equilibrium":
            return self._equilibrium_at(T, P_CHEM_BAR, co, mh)
        if mode != "quenched":
            raise ValueError(mode)
        if g_cgs is None:
            raise ValueError("quenched mode needs the surface gravity g_cgs")
        P = np.logspace(-4, 3, 300)                                     # bar, top to bottom
        tau = 1e-2 * (P * 1e6) / g_cgs
        gam, Tint, Teq, f = 0.4, 100.0, float(T), 0.25
        Tp = (0.75 * Tint**4 * (2/3 + tau) + 0.75 * Teq**4 * f * (2/3 + 1/(gam*np.sqrt(3))
              + (gam/np.sqrt(3) - 1/(gam*np.sqrt(3))) * np.exp(-gam*tau*np.sqrt(3)))) ** 0.25
        Tp = Tp * (float(T) / np.interp(P_CHEM_BAR, P, Tp))          # anchor: T(P_CHEM_BAR) == T
        # convective interior: follow the adiabat once the radiative gradient falls below it
        dlnT = np.gradient(np.log(Tp), np.log(P)); ad = 2.0 / 7.0
        deep = np.where((P > P_CHEM_BAR) & (dlnT < ad))[0]
        if deep.size:
            i0 = deep[0]; Tp[i0:] = Tp[i0] * (P[i0:] / P[i0]) ** ad
        t_chem = 1.5e-6 / P * np.exp(42000.0 / Tp)
        mu_mH = 2.3 * 1.6726e-24; H = 1.380649e-16 * Tp / (mu_mH * g_cgs)
        t_mix = H**2 / 1e9
        fast = (t_chem < t_mix) & (P >= P_CHEM_BAR) & (P <= 100.0)     # levels at/below the photosphere that equilibrate
        if not fast.any():
            return self._equilibrium_at(T, P_CHEM_BAR, co, mh)         # nothing equilibrates: no quench signal
        # the quench point is the SHALLOWEST level where chemistry still keeps up: below it the gas
        # is in equilibrium, above it the composition is frozen at this level's value and mixed up
        iq = np.min(np.where(fast)[0])
        return self._equilibrium_at(Tp[iq], P[iq], co, mh)

def worker(rows, wl):
    import warnings; warnings.filterwarnings("ignore")
    return [one_native(r, wl) for r in rows]


def make_split(name, seed, jobs, wl, limit=None, mode="equilibrium"):
    P = pd.read_parquet(os.path.join(V2, "data", f"{name}_params.parquet"))
    P = P[list(BULK) + ["atm fill_gas"]].copy() if "atm fill_gas" in P else P[list(BULK)].copy()
    if limit: P = P.iloc[:limit].copy()
    if mode == "equilibrium":
        rng = np.random.default_rng(seed)
        P["co_ratio"] = rng.uniform(*CO_RANGE, len(P)); P["mh"] = rng.uniform(*MH_RANGE, len(P))
    else:
        # a shifted variant must be the SAME planets with the SAME chemistry parameters: reuse the
        # equilibrium run's stored draws rather than redrawing, so the pairing cannot drift
        E = pd.read_parquet(os.path.join(DATA, f"{name}_params.parquet"))
        assert len(E) <= len(P), f"{name}: equilibrium params longer than the bulk table"
        key = list(BULK)
        M = P[key].round(9).merge(E[key + ["co_ratio", "mh"]].round(9), on=key, how="left")
        if M["co_ratio"].isna().any():
            raise RuntimeError(f"{name}: could not pair {int(M['co_ratio'].isna().sum())} planets with the equilibrium run")
        P["co_ratio"] = M["co_ratio"].to_numpy(); P["mh"] = M["mh"].to_numpy()
    chem = Chemistry(); t0 = time.time()
    G = 6.674e-8 * (P["p_mass"].to_numpy() * 5.972e27) / (P["p_radius"].to_numpy() * 6.371e8) ** 2   # cgs
    comp = [chem.composition(r["atm temperature"], r.co_ratio, r.mh, mode=mode, g_cgs=g)
            for (_, r), g in zip(P.iterrows(), G)]
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
    ap.add_argument("--jobs", type=int, default=12); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--mode", default="equilibrium", choices=["equilibrium", "quenched"]); a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    wl = native_wavelengths(); np.save(os.path.join(DATA, "native_wl.npy"), wl)
    if a.smoke:
        kept, X = make_split("test1", 3, min(a.jobs, 6), wl, limit=a.smoke, mode=a.mode)
        print("smoke: log10 abundance medians", {g: round(float(kept[f"atm {g}"].median()), 2) for g in GASES})
        print("smoke: native spectrum shape", X.shape, "depth range", f"{X.min():.3e}-{X.max():.3e}")
        return
    for s in a.splits:
        kept, X = make_split(s, 100 + SPLITS.index(s), a.jobs, wl, mode=a.mode)
        tag = "" if a.mode == "equilibrium" else f"_{a.mode}"
        kept.to_parquet(os.path.join(DATA, f"{s}_params{tag}.parquet")); np.save(os.path.join(DATA, f"{s}_native{tag}.npy"), X)


if __name__ == "__main__":
    main()
