"""Re-render the v2 grid planets with an aerosol, on the native grid.

This is the v2 counterpart of ../generate_aerosol_paired.py. That script took
the committed 550-bin test planets and re-rendered each one with a grey cloud
deck at five cloud-top pressures and with a Lee et al. (2013) Mie haze at five
particle densities, keeping every other parameter fixed, so that the aerosol
effect is a within-planet difference. This module does the same for the v2
planets in v2/data/, but at TauREx's native resolution (2,753 points on
0.5-7.8 um, v2/data/native_wl.npy), so the aerosol spectra can be binned and
noised by bin_spectra.py / noise.py exactly like the clear ones.

Aerosol prescriptions (copied verbatim from generate_aerosol_paired.py, which
copied them from generate_cloudy_testset.py and generate_hazy_testset.py):

  cloud  TauREx SimpleCloudsContribution: an optically thick grey deck with
         cloud top at pressure P (Pa), via Atmosphere(cloud_pressure=P).
         Levels CLOUD_LEVELS = [1e5, 1e4, 1e3, 1e2, 1e1] Pa. Decks deeper than
         the region the spectrum probes have almost no effect (1e5 Pa is
         effectively clear); 1e1 Pa leaves the spectrum nearly featureless.

  haze   TauREx LeeMieContribution (Lee et al. 2013), via
         Atmosphere(cloud_model={"type": "lee_mie", "radius": 0.1, "q": 40,
                                 "mix_ratio": D}).
         radius = HAZE_RADIUS = 0.1 um particle radius; q = HAZE_Q = 40 is the
         Q0 extinction parameter (TauREx default); mix_ratio = D is the
         particle number density in particles/m^3 (an absolute density, not a
         fractional mixing ratio). The haze fills the whole column: bottomP and
         topP are left at the fork's defaults (-1, i.e. unbounded), as in the
         old script. Levels HAZE_LEVELS = [2e5, 2e6, 3e7, 2.4e8, 1e10] m^-3.
         The Mie opacity falls with wavelength, so the haze mostly lifts the
         short-wavelength continuum instead of muting features uniformly.

Everything else about each planet (planet radius and mass, atmospheric
temperature and pressures, the six abundances, fill gas, stellar properties,
semi-major axis) is taken from {split}_params.parquet and passed through the
same Atmosphere/Planet/Star/System path as generate_grid.build_system, so the
only difference from {split}_native.npy is the aerosol. --validate checks that
by rendering with no aerosol and comparing against the stored spectra.

File names mirror the old paired files, so results are comparable level by
level:
  old  multirex_spectra_H2_paired_cloudy_{tag}Pa.parquet   tag = 1e5 ... 1e1
  new  v2/data/{split}_native_cloud_{tag}Pa.npy
  old  multirex_spectra_H2_paired_hazy_{tag}.parquet       tag = 2p0e5, 2p0e6,
  new  v2/data/{split}_native_haze_{tag}.npy                     3p0e7, 2p4e8, 1p0e10
Each output is float32 (n, 2753) in the same row order as {split}_params;
rows whose forward model fails (exception, non-finite, depth > 1 or <= 0) are
NaN and are counted in the printout.

Usage:
    python shift_aerosol.py --validate
    python shift_aerosol.py --kind both --limit 20 --jobs 2      # timing
    python shift_aerosol.py --kind cloud --jobs 4               # all levels, five test splits
    python shift_aerosol.py --kind haze --levels 3p0e7 1p0e10 --splits test1 test2
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate_grid import DATA, FILL_GAS, GASES, WL_MAX, WL_MIN  # noqa: E402

TESTS = [f"test{k}" for k in range(1, 6)]

# generate_cloudy_testset.py / generate_aerosol_paired.py
CLOUD_LEVELS = [1e5, 1e4, 1e3, 1e2, 1e1]          # cloud-top pressure, Pa
# generate_hazy_testset.py / generate_aerosol_paired.py
HAZE_LEVELS = [2e5, 2e6, 3e7, 2.4e8, 1e10]        # particle density, m^-3
HAZE_RADIUS, HAZE_Q = 0.1, 40                     # um, Q0


def cloud_tag(cp):
    """1e5 -> '1e5Pa', as in multirex_spectra_H2_paired_cloudy_{tag}Pa.parquet."""
    return f"{cp:.0e}".replace("+", "").replace("e0", "e") + "Pa"


def haze_tag(hz):
    """2.4e8 -> '2p4e8', as in multirex_spectra_H2_paired_hazy_{tag}.parquet."""
    return f"{hz:.1e}".replace("+", "").replace("e0", "e").replace(".", "p")


def levels(kind):
    """(kind, tag, aerosol kwargs) for every level of the requested kind(s)."""
    out = []
    if kind in ("cloud", "both"):
        out += [("cloud", cloud_tag(cp), dict(cloud_pressure=cp)) for cp in CLOUD_LEVELS]
    if kind in ("haze", "both"):
        out += [("haze", haze_tag(hz), dict(haze_density=hz)) for hz in HAZE_LEVELS]
    return out


def out_path(split, kind, tag):
    return os.path.join(DATA, f"{split}_native_{kind}_{tag}.npy")


def build_system(row, cloud_pressure=None, haze_density=None):
    """generate_grid.build_system with the aerosol keywords added. With both
    aerosol arguments None this is the clear path and must reproduce
    {split}_native.npy."""
    from multirex import Atmosphere, Planet, Star, System
    kw = dict(temperature=float(row["atm temperature"]),
              base_pressure=float(row["atm base_pressure"]),
              top_pressure=float(row["atm top_pressure"]),
              composition={g: float(row[f"atm {g}"]) for g in GASES},
              fill_gas=FILL_GAS)
    if cloud_pressure is not None:
        kw["cloud_pressure"] = float(cloud_pressure)
    if haze_density is not None:
        kw["cloud_model"] = {"type": "lee_mie", "radius": HAZE_RADIUS,
                             "q": HAZE_Q, "mix_ratio": float(haze_density)}
    atm = Atmosphere(**kw)
    planet = Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]), atmosphere=atm)
    star = Star(temperature=float(row["s temperature"]), radius=float(row["s radius"]),
                mass=float(row["s mass"]))
    system = System(planet=planet, star=star, sma=float(row["sma"]))
    system.make_tm()
    return system


def one_native(row, wl_native, cloud_pressure=None, haze_density=None):
    """Noise-free native transit depth on wl_native, or None on failure.
    Extraction and cleaning are identical to generate_grid.one_native."""
    try:
        system = build_system(row, cloud_pressure, haze_density)
        wn, rprs = system.transmission.model()[:2]
        wl = 1e4 / np.asarray(wn)
        order = np.argsort(wl)
        wl, rprs = wl[order], np.asarray(rprs)[order]
        m = (wl >= WL_MIN) & (wl <= WL_MAX)
        y = rprs[m]
        if y.shape[0] != wl_native.shape[0] or not np.allclose(wl[m], wl_native):
            y = np.interp(wl_native, wl[m], y)
        if not np.all(np.isfinite(y)) or np.any(y > 1.0) or np.any(y <= 0):
            return None
        return y.astype(np.float32)
    except Exception:
        return None


def _worker(rows, wl_native, aerosol):
    import warnings
    warnings.filterwarnings("ignore")
    return [one_native(r, wl_native, **aerosol) for r in rows]


def render(params, wl_native, jobs=4, chunk=25, **aerosol):
    """float32 (n, len(wl_native)) in params' row order; failed rows are NaN."""
    rows = [params.iloc[i].to_dict() for i in range(len(params))]
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    out = Parallel(n_jobs=jobs)(delayed(_worker)(c, wl_native, aerosol) for c in chunks)
    X = np.full((len(rows), len(wl_native)), np.nan, dtype=np.float32)
    for i, s in enumerate(s for c in out for s in c):
        if s is not None:
            X[i] = s
    return X


def load_params(split):
    return pd.read_parquet(os.path.join(DATA, f"{split}_params.parquet"))


def wait_for(split, max_minutes=30):
    """Poll for {split}_params.parquet and {split}_native.npy."""
    need = [os.path.join(DATA, f"{split}_params.parquet"),
            os.path.join(DATA, f"{split}_native.npy")]
    t0 = time.time()
    while not all(os.path.exists(p) for p in need):
        if time.time() - t0 > 60 * max_minutes:
            raise SystemExit(f"{split}: data files not present after {max_minutes} min")
        print(f"waiting for {split} data files...", flush=True)
        time.sleep(60)
    time.sleep(5)  # let a writer finish
    p = load_params(split)
    X = np.load(need[1], mmap_mode="r")
    if X.shape[0] != len(p):
        raise SystemExit(f"{split}: {X.shape[0]} spectra vs {len(p)} params rows")
    return p


def amplitude_ratio(X_aer, X_clear):
    """Median over planets of peak-to-peak(aerosol) / peak-to-peak(clear),
    over rows valid in both."""
    ok = np.isfinite(X_aer).all(axis=1) & np.isfinite(X_clear).all(axis=1)
    pa = np.ptp(X_aer[ok].astype(np.float64), axis=1)
    pc = np.ptp(X_clear[ok].astype(np.float64), axis=1)
    return float(np.median(pa / pc)), int(ok.sum())


def validate(jobs, n=20):
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    params = wait_for("test1").iloc[:n]
    stored = np.load(os.path.join(DATA, "test1_native.npy"))[:n].astype(np.float64)

    print(f"validate: {n} planets of test1, {len(wl)} native points")
    t0 = time.time()
    clear = render(params, wl, jobs=jobs)
    dt = (time.time() - t0) / n
    nfail = int(np.isnan(clear).any(axis=1).sum())
    resid = np.abs(clear.astype(np.float64) - stored)
    print(f"  no aerosol: {nfail} failures, {dt:.2f} s/planet")
    print(f"  max |residual| vs test1_native.npy = {np.nanmax(resid):.3e} "
          f"(relative to mean depth {np.nanmax(resid) / stored.mean():.3e})")
    ok = np.nanmax(resid) < 1e-6
    print("  " + ("reproduces the stored spectra" if ok else "DOES NOT reproduce the stored spectra"))

    for kind, tag, aer in levels("both"):
        if tag not in ("1e4Pa", haze_tag(HAZE_LEVELS[2])):
            continue
        t0 = time.time()
        X = render(params, wl, jobs=jobs, **aer)
        r, nok = amplitude_ratio(X, clear)
        print(f"  {kind} {tag}: median ptp ratio aerosol/clear = {r:.3f} "
              f"over {nok} planets, {(time.time() - t0) / n:.2f} s/planet")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", nargs="+", default=TESTS)
    ap.add_argument("--kind", choices=["cloud", "haze", "both"], default="both")
    ap.add_argument("--levels", nargs="+", default=None,
                    help="subset of level tags, e.g. 1e4Pa 3p0e7")
    ap.add_argument("--limit", type=int, default=None, help="first N planets per split")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    if a.validate:
        sys.exit(0 if validate(a.jobs) else 1)

    todo = levels(a.kind)
    if a.levels:
        known = {t for _, t, _ in todo}
        bad = set(a.levels) - known
        if bad:
            raise SystemExit(f"unknown level tags {sorted(bad)}; known: {sorted(known)}")
        todo = [x for x in todo if x[1] in a.levels]
    wl = np.load(os.path.join(DATA, "native_wl.npy"))

    for split in a.splits:
        params = wait_for(split)
        if a.limit:
            params = params.iloc[:a.limit]
        clear = np.load(os.path.join(DATA, f"{split}_native.npy"))[:len(params)]
        print(f"=== {split}: {len(params)} planets ===", flush=True)
        for kind, tag, aer in todo:
            t0 = time.time()
            X = render(params, wl, jobs=a.jobs, **aer)
            dt = time.time() - t0
            nfail = int(np.isnan(X).any(axis=1).sum())
            r, _ = amplitude_ratio(X, clear)
            path = out_path(split, kind, tag)
            if a.limit:
                path = path.replace(".npy", f"_limit{a.limit}.npy")
            np.save(path, X)
            print(f"  {kind} {tag}: {nfail} failures, ptp ratio {r:.3f}, "
                  f"{dt / len(params):.2f} s/planet, {dt:.0f} s -> {os.path.basename(path)}",
                  flush=True)


if __name__ == "__main__":
    main()
