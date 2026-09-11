"""Regenerate the aerosol test sets on the COMMITTED planets.

Why this supersedes generate_clear_control.py
---------------------------------------------

The aerosol degradations were quoted against the 88.92% clear baseline from the
five committed test sets, but the aerosol sets are a different draw of planets,
and the generator's sampling has moved since the committed data was made. Part
of every recorded degradation was therefore the draw rather than the aerosol.

generate_clear_control.py addressed that by drawing a fresh CLEAR set the same
way the aerosol sets were drawn, so both sides shared the new sampling. That
removes the distributional mismatch, but it still compares two different draws
of ~540 planets against each other, so it carries the sampling noise between
them and it discards the committed baseline.

This script does the better thing: it applies the aerosol physics to the exact
planets already in the five committed clear test sets. Nothing is resampled. The
cloudy and hazy spectra are then paired planet-by-planet with the committed
clear spectra, the 88.92% baseline is a legitimate comparator again, and the
degradation is a within-planet difference with no draw confound of any kind.

Everything except the aerosol is taken verbatim from each committed row -
abundances, temperature, pressures, planet radius and mass, stellar properties,
semi-major axis - so the only difference from the committed spectrum is the
aerosol. Stage 0 verifies that by regenerating with NO aerosol and checking the
result reproduces the committed spectra.

Aerosol prescriptions are copied from generate_cloudy_testset.py and
generate_hazy_testset.py so the physics matches what was published.

Usage:
    python generate_aerosol_paired.py            # all 10 levels
    python generate_aerosol_paired.py --quick    # 60 planets per level
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

WL_MIN, WL_MAX, RESOLUTION = 0.5, 7.8, 550
FILL_GAS = "H2"
BIO_CH4, BIO_O3 = -6.0, -7.0
GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]

CLOUD_LEVELS = [1e5, 1e4, 1e3, 1e2, 1e1]          # generate_cloudy_testset.py
HAZE_LEVELS = [2e5, 2e6, 3e7, 2.4e8, 1e10]        # generate_hazy_testset.py
HAZE_RADIUS, HAZE_Q = 0.1, 40

SOURCE_FMT = f"multirex_spectra_{FILL_GAS}_test_set_{{}}.parquet"


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def one_spectrum(row, wn_grid, cloud_pressure=None, haze_density=None):
    """One committed planet, re-rendered with an aerosol added.

    Built through MultiREx's Atmosphere/Planet/Star/System path rather than by
    constructing a TauREx model directly, so the physics is identical to the
    published generators.
    """
    from multirex import Atmosphere, Planet, Star, System

    kw = dict(temperature=float(row["atm temperature"]),
              base_pressure=float(row["atm base_pressure"]),
              top_pressure=float(row["atm top_pressure"]),
              composition={g: float(row[f"atm {g}"]) for g in GASES},
              fill_gas=FILL_GAS)
    if cloud_pressure is not None:
        kw["cloud_pressure"] = cloud_pressure
    if haze_density is not None:
        kw["cloud_model"] = {"type": "lee_mie", "radius": HAZE_RADIUS,
                             "q": HAZE_Q, "mix_ratio": haze_density}

    atm = Atmosphere(**kw)
    planet = Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]),
                    atmosphere=atm)
    star = Star(temperature=float(row["s temperature"]),
                radius=float(row["s radius"]), mass=float(row["s mass"]))
    system = System(planet=planet, star=star, sma=float(row["sma"]))
    system.make_tm()
    return np.asarray(system.generate_spectrum(wn_grid)[1])


def render(df, wn_grid, order, n_jobs, **aerosol):
    rows = [df.iloc[i] for i in range(len(df))]
    out = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(one_spectrum)(r, wn_grid, **aerosol) for r in rows)
    return np.vstack([o[order] for o in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="60 planets per level")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    from multirex import Physics
    wn_grid = Physics.wavenumber_grid(WL_MIN, WL_MAX, RESOLUTION)
    order = np.argsort(10000.0 / wn_grid)

    src = pd.concat([pd.read_parquet(SOURCE_FMT.format(i)) for i in range(1, 6)],
                    ignore_index=True)
    if args.quick:
        src = src.iloc[:60].copy()
    cols = spectral_cols(src)
    stored = src[cols].values
    print(f"{len(src)} committed planets, {len(cols)} bins\n")

    # Stage 0 - no aerosol must reproduce the committed spectra exactly.
    print("verifying the no-aerosol path reproduces the committed spectra...")
    chk = render(src.iloc[:12], wn_grid, order, args.jobs)
    dev = np.abs(chk - stored[:12]).max() / stored[:12].mean()
    print(f"  max|dev| = {dev:.3e} of mean depth")
    if dev > 1e-12:
        raise SystemExit("no-aerosol path does NOT reproduce the committed "
                         "spectra; the parameter mapping is wrong and every "
                         "paired degradation would be unattributable.")
    print("  reproduces exactly - only the aerosol will differ\n")

    labels = np.where((src["atm CH4"] >= BIO_CH4) & (src["atm O3"] >= BIO_O3),
                      "yes", "no")
    assert (labels == src["biosignature"].values).all(), "label mismatch"

    jobs = []
    for cp in CLOUD_LEVELS:
        tag = f"{cp:.0e}".replace("+", "").replace("e0", "e")
        jobs.append((f"deck {cp:.0e} Pa",
                     f"multirex_spectra_{FILL_GAS}_paired_cloudy_{tag}Pa.parquet",
                     dict(cloud_pressure=cp)))
    for hz in HAZE_LEVELS:
        tag = f"{hz:.1e}".replace("+", "").replace("e0", "e").replace(".", "p")
        jobs.append((f"haze {hz:.1e} m-3",
                     f"multirex_spectra_{FILL_GAS}_paired_hazy_{tag}.parquet",
                     dict(haze_density=hz)))

    amp_src = np.median(stored.std(axis=1) / stored.mean(axis=1))
    for name, out, aerosol in jobs:
        print(f"--- {name} ---", flush=True)
        spec = render(src, wn_grid, order, args.jobs, **aerosol)

        d = src.copy()
        d[cols] = spec
        d["biosignature"] = labels
        # Cleaning matches the published generators: restrict to spectral
        # columns, since the fork reports the unused aerosol field as None.
        s = d[cols]
        keep = s.notna().all(axis=1) & (s <= 1.0).all(axis=1)
        d["aerosol_valid"] = keep.values
        d.to_parquet(out)

        v = keep.values
        amp = np.median(spec[v].std(axis=1) / spec[v].mean(axis=1)) / amp_src
        print(f"    valid {int(v.sum())}/{len(d)}   feature amplitude "
              f"{amp:.3f}x the committed clear spectra")
        print(f"    wrote {out}\n", flush=True)

    print("Evaluate with: python evaluate_aerosol_paired.py")


if __name__ == "__main__":
    main()
