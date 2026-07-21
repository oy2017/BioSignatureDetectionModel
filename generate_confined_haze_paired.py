"""
Confined-haze control on the committed planets (R1-3).

The deck-versus-haze comparison at matched feature suppression attributes the
divergence to the wavelength dependence of the opacity. The two prescriptions
also differ in vertical structure - the deck truncates the atmosphere at a
pressure level while the haze fills the column - so a control is needed to show
that vertical structure is not what drives the difference.

This renders the committed test planets with the haze density that matches the
10^3 Pa deck's suppression, confined to pressures below 10^3 Pa (the deck's own
altitude). Comparing it with the whole-column haze at the same density isolates
vertical structure with everything else held fixed.

An earlier version of this control drew fresh planets, which put it on a
different population from the paired aerosol table and made its absolute
accuracy incomparable. This uses the same committed planets as
generate_aerosol_paired.py, so all three arms - deck, whole-column haze and
confined haze - sit on one footing.

Usage:
    python generate_confined_haze_paired.py
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from generate_aerosol_paired import (GASES, FILL_GAS, HAZE_Q, HAZE_RADIUS,
                                     SOURCE_FMT, spectral_cols)

HAZE_DENSITY = 2.4e8      # matches the whole-column arm already generated
BOTTOM_P = 1e3            # confine to P < 1e3 Pa, the matched deck's altitude
OUT_FMT = "multirex_spectra_H2_paired_hazeconfined_set_{}.parquet"
N_JOBS = 6


def one_spectrum(row, wn_grid):
    from multirex import Atmosphere, Planet, Star, System
    atm = Atmosphere(
        temperature=float(row["atm temperature"]),
        base_pressure=float(row["atm base_pressure"]),
        top_pressure=float(row["atm top_pressure"]),
        composition={g: float(row[f"atm {g}"]) for g in GASES},
        fill_gas=FILL_GAS,
        cloud_model={"type": "lee_mie", "radius": HAZE_RADIUS, "q": HAZE_Q,
                     "mix_ratio": HAZE_DENSITY, "bottomP": BOTTOM_P})
    planet = Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]),
                    atmosphere=atm)
    star = Star(temperature=float(row["s temperature"]),
                radius=float(row["s radius"]), mass=float(row["s mass"]))
    system = System(planet=planet, star=star, sma=float(row["sma"]))
    system.make_tm()
    return np.asarray(system.generate_spectrum(wn_grid)[1])


def main():
    from multirex import Physics
    wn_grid = Physics.wavenumber_grid(0.5, 7.8, 550)
    wl = 10000.0 / np.asarray(wn_grid)
    order = np.argsort(wl)

    print(f"confined haze: {HAZE_DENSITY:.1e} m^-3, P < {BOTTOM_P:.0e} Pa, "
          f"rendered on the committed test planets\n")
    for i in range(1, 6):
        df = pd.read_parquet(SOURCE_FMT.format(i))
        cols = spectral_cols(df)
        print(f"--- set {i}: {len(df)} planets ---", flush=True)
        rows = [df.iloc[j] for j in range(len(df))]
        spec = np.vstack([o[order] for o in Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(one_spectrum)(r, wn_grid) for r in rows)])

        stored = df[cols].to_numpy(dtype=float)
        rel = np.abs((spec - stored) / stored)
        if np.median(rel) < 1e-6:
            raise RuntimeError(
                f"set {i}: output matches the clear baseline (median relative "
                f"difference {np.median(rel):.2e}); the haze was not applied")
        print(f"    differs from clear: median rel {np.median(rel):.3e}")

        out = df.copy()
        out[cols] = spec
        out.to_parquet(OUT_FMT.format(i))
        print(f"    wrote {OUT_FMT.format(i)} ({len(out)} rows)\n")


if __name__ == "__main__":
    main()
