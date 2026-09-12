"""Missing-absorber axis: HCN and C2H2, which the training forward model omits.

Carbon-rich and quenched atmospheres carry HCN and C2H2 (Moses et al. 2013), whose bands at
~3.0 and 3.3 um overlap the CH4 features the C/O screen reads. The test planets are re-rendered
with both added at their FastChem abundances (equilibrium at the photospheric level, or the
quenched composition), using the Exo-Transmit tables opacHCN.dat / opacC2H2.dat dropped into
MultiREx's data directory exactly as opacCO.dat was. The six training gases are recomputed and
checked against the stored values so the ONLY change is the two new absorbers.

Usage: python shift_absorbers.py --jobs 4 [--mode equilibrium|quenched]
Writes data/{split}_native_absorbers[_quenched].npy (NaN rows for failed renders)
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, TESTS  # noqa: E402
import generate_grid as G  # noqa: E402

EXTRA = {"HCN": "C1H1N1_hcn", "C2H2": "C2H2"}


def build_system(row, gases):
    from multirex import Atmosphere, Planet, Star, System
    atm = Atmosphere(temperature=float(row["atm temperature"]), base_pressure=float(row["atm base_pressure"]),
                     top_pressure=float(row["atm top_pressure"]),
                     composition={g: float(row[f"atm {g}"]) for g in gases}, fill_gas=G.FILL_GAS)
    planet = Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]), atmosphere=atm)
    star = Star(temperature=float(row["s temperature"]), radius=float(row["s radius"]), mass=float(row["s mass"]))
    system = System(planet=planet, star=star, sma=float(row["sma"])); system.make_tm()
    return system


def one_native(row, wl_native, gases):
    try:
        system = build_system(row, gases)
        wn, rprs = system.transmission.model()[:2]
        wl = 1e4 / np.asarray(wn); o = np.argsort(wl); wl, rprs = wl[o], np.asarray(rprs)[o]
        m = (wl >= G.WL_MIN) & (wl <= G.WL_MAX); y = rprs[m]
        if y.shape[0] != wl_native.shape[0] or not np.allclose(wl[m], wl_native):
            y = np.interp(wl_native, wl[m], y)
        if not np.all(np.isfinite(y)) or np.any(y > 1.0) or np.any(y <= 0):
            return None
        return y.astype(np.float32)
    except Exception:
        return None


def worker(rows, wl, gases):
    import warnings; warnings.filterwarnings("ignore")
    return [one_native(r, wl, gases) for r in rows]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--mode", default="equilibrium", choices=["equilibrium", "quenched"])
    ap.add_argument("--splits", nargs="*", default=TESTS); a = ap.parse_args()
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    chem = G.Chemistry()
    for g, h in EXTRA.items():
        chem.idx[g] = chem.fc.getGasSpeciesIndex(h)
        assert chem.idx[g] >= 0, f"FastChem has no species {h}"
    gases = list(G.GASES) + list(EXTRA)
    tag = "absorbers" if a.mode == "equilibrium" else "absorbers_quenched"
    for s in a.splits:
        out = os.path.join(DATA, f"{s}_native_{tag}.npy")
        if os.path.exists(out):
            print(f"{s}: exists"); continue
        src = f"{s}_params.parquet" if a.mode == "equilibrium" else f"{s}_params_quenched.parquet"
        P = pd.read_parquet(os.path.join(DATA, src)).reset_index(drop=True)
        Gc = 6.674e-8 * (P["p_mass"].to_numpy() * 5.972e27) / (P["p_radius"].to_numpy() * 6.371e8) ** 2
        t0 = time.time()
        comp = [chem.composition(r["atm temperature"], r.co_ratio, r.mh, mode=a.mode, g_cgs=g) for (_, r), g in zip(P.iterrows(), Gc)]
        for g in G.GASES:   # the recomputed training gases must reproduce the stored ones
            new = np.array([c[g] for c in comp]); old = P[f"atm {g}"].to_numpy()
            assert np.allclose(new, old, atol=1e-6), f"{s}: {g} recomputed differently (max |d| {np.abs(new-old).max():.2e})"
        for g in EXTRA:
            P[f"atm {g}"] = [c[g] for c in comp]
        print(f"{s}: chemistry {time.time()-t0:.0f} s; median log10 HCN {P['atm HCN'].median():.2f}, C2H2 {P['atm C2H2'].median():.2f};"
              f" C/O>1 planets: HCN {P.loc[P.label_co==1,'atm HCN'].median():.2f}, C2H2 {P.loc[P.label_co==1,'atm C2H2'].median():.2f}", flush=True)
        rows = [P.iloc[i].to_dict() for i in range(len(P))]; t0 = time.time()
        res = Parallel(n_jobs=a.jobs)(delayed(worker)(c, wl, gases) for c in G.chunked(rows, 50))
        specs = [x for c in res for x in c]
        X = np.full((len(P), len(wl)), np.nan, dtype=np.float32)
        for i, sp in enumerate(specs):
            if sp is not None: X[i] = sp
        np.save(out, X); P.to_parquet(os.path.join(DATA, f"{s}_params_{tag}.parquet"))
        print(f"{s}: {sum(sp is not None for sp in specs)}/{len(P)} rendered, {time.time()-t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
