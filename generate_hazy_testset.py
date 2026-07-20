"""
Generate hazy test sets at controlled haze densities.

Completes the haze half of axis 3 of Reviewer 1's R1-3 list ("different cloud
and haze prescriptions"). The haze is TauREx's LeeMieContribution (Lee et al.
2013): Mie scattering by small particles, whose opacity falls with wavelength
so it acts principally as a short-wavelength continuum lift rather than the
uniform muting of the grey deck. The training data contains no aerosols of any
kind, so this is again a distribution shift produced by different forward-model
physics.

Requires the Mie-capable MultiREx fork (cloud_model parameter on Atmosphere):

    pip install "git+https://github.com/oy2017/MultiREx-public.git@2e20551"

Design notes:

  * Everything except the haze replicates generate_cloudy_testset.py exactly -
    same wavelength grid, SNR, parameter ranges, stratified composition plan
    and cleaning - so the aerosol prescription is the only difference between
    the cloudy and hazy sets.

  * The particle density (lee_mie_mix_ratio, particles/m3 - an absolute number
    density, NOT a fractional mixing ratio; values below ~1e4 are optically
    invisible) is held FIXED within each batch. The five levels were
    calibrated in a paired clear-vs-hazy experiment to span the grey deck's
    informative suppression range; the generated sets measured
    0.86 / 0.74 / 0.58 / 0.50 / 0.30 of clear amplitude, against the deck's
    0.92 / 0.73 / 0.50 / 0.27. Comparing at matched muting is the point: at
    equal suppression, performance differences between deck and haze isolate
    the wavelength dependence of the opacity (a control with the haze
    confined to P < 1e3 Pa reproduces the whole-column result, so vertical
    structure does not drive the comparison - see
    generate_confined_haze_control.py).

  * Particle radius 0.1 um and Q0 = 40 are fixed (the TauREx defaults for
    this contribution, radius aside); the haze fills the whole atmospheric
    column, a simplification - real photochemical hazes are altitude-confined.
    Only the density varies between levels.

  * Suppression saturates near 0.3x at high density because the haze's own
    wavelength-dependent opacity imposes a spectral slope of its own; unlike
    the grey deck it cannot produce a featureless flat spectrum until the
    density is so extreme that the whole column is opaque at every wavelength.

Usage:
    python generate_hazy_testset.py            # all levels, ~600 planets each
    python generate_hazy_testset.py --quick    # 120 planets each, for a smoke test
"""

import argparse
import re

import numpy as np
import pandas as pd
from multirex import Atmosphere, Planet, Star, System, Physics

SNR = 15
WL_MIN, WL_MAX, RESOLUTION = 0.5, 7.8, 550
FILL_GAS = "H2"
BIO_CH4, BIO_O3 = -6.0, -7.0
HAZE_RADIUS, HAZE_Q = 0.1, 40

# Particle number densities in particles/m3, paired-calibrated so population
# feature suppression spans the grey deck's informative range.
HAZE_LEVELS = [2e5, 2e6, 3e7, 2.4e8, 1e10]

# Identical to generate_cloudy_testset.py.
PROFILES = {
    "biosignature": ({"CH4": (BIO_CH4, -3), "O3": (BIO_O3, -1)}, 0.5),
    "nonbio_ch4":   ({"CH4": (BIO_CH4, -3), "O3": (-10, BIO_O3)}, 0.1666),
    "nonbio_o3":    ({"CH4": (-9, BIO_CH4), "O3": (BIO_O3, -1)}, 0.1666),
    "nonbio_none":  ({"CH4": (-9, BIO_CH4), "O3": (-10, BIO_O3)}, 0.1668),
}
BACKGROUND = {"H2O": (-10, -1), "CO": (-9, -3), "CO2": (-9, -3), "NH3": (-9, -3)}


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return [c for c in df.columns
            if isinstance(c, float) or (isinstance(c, str) and fp.match(c))]


def generate_one_level(mix_ratio, n_universes, wn_grid):
    """Generate a stratified, class-balanced batch at one haze density."""
    frames = []
    star = Star(temperature=(2500, 7500), radius=(0.1, 1.7), mass=(0.1, 1.7))

    for name, (chem, frac) in PROFILES.items():
        count = int(n_universes * frac)
        if count == 0:
            continue
        print(f"    {name:<14} n={count:<4} haze={mix_ratio:.1e} m^-3")

        atmosphere = Atmosphere(
            temperature=(500, 2500),
            base_pressure=(1e5, 10e5),
            top_pressure=(1, 10),
            composition={**BACKGROUND, **chem},
            fill_gas=FILL_GAS,
            cloud_model={"type": "lee_mie", "radius": HAZE_RADIUS,
                         "q": HAZE_Q, "mix_ratio": mix_ratio},
        )
        planet = Planet(radius=(1.0, 26.0), mass=(1.0, 300.0), atmosphere=atmosphere)
        system = System(planet=planet, star=star, sma=(0.01, 0.5))
        system.make_tm()
        res = system.explore_multiverse(
            wn_grid=wn_grid, n_universes=count, n_observations=1,
            snr=SNR, header=True, path=None, n_jobs=-1)
        frames.append(res["spectra"])

    return pd.concat(frames, ignore_index=True)


def clean_and_label(df):
    """Cleaning and labelling as in generate_cloudy_testset.py, with one fix:
    dropna is restricted to the spectral columns, because the Mie-capable fork
    reports 'atm cloud_pressure' as None (-> NaN) whenever cloud_model is used,
    and a blanket dropna() would silently delete every row."""
    spectral = spectral_cols(df)
    before = len(df)
    df = df.dropna(subset=spectral)
    df = df[(df[spectral] <= 1.0).all(axis=1)]
    df = df.copy()
    df["biosignature"] = np.where(
        (df["atm CH4"] >= BIO_CH4) & (df["atm O3"] >= BIO_O3), "yes", "no")
    print(f"    cleaned {before} -> {len(df)} rows "
          f"({(df['biosignature'] == 'yes').mean():.1%} positive)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="120 planets per level instead of 600")
    args = ap.parse_args()
    n = 120 if args.quick else 600

    wn_grid = Physics.wavenumber_grid(WL_MIN, WL_MAX, RESOLUTION)
    print(f"Generating hazy {FILL_GAS} test sets: {len(HAZE_LEVELS)} levels "
          f"x {n} planets\n")

    for mix in HAZE_LEVELS:
        tag = f"{mix:.1e}".replace("+", "").replace(".0e", "e").replace(".", "p")
        print(f"--- haze density {mix:.1e} m^-3 ---")
        df = clean_and_label(generate_one_level(mix, n, wn_grid))
        out = f"multirex_spectra_{FILL_GAS}_hazy_{tag}.parquet"
        df.to_parquet(out)
        print(f"    wrote {out}\n")

    print("Done. Evaluate with: python evaluate_hazy.py")


if __name__ == "__main__":
    main()
