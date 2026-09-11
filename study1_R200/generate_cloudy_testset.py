"""
Generate cloudy test sets at controlled cloud-top pressures.

Addresses axis 3 of Reviewer 1's R1-3 list ("different cloud and haze
prescriptions"). Clouds are the dominant confounder in real transmission
spectroscopy: an optically thick deck mutes absorption features, so a
molecule-rich atmosphere behind a high cloud resembles a molecule-poor one.
The training data contains no clouds at all, so this is a genuine distribution
shift produced by different physics rather than by perturbing existing spectra.

Requires the cloud-capable MultiREx:

    pip install "git+https://github.com/oy2017/MultiREx-public.git@2281923"

Design notes:

  * Everything except clouds replicates generate_multiverse_data.py exactly -
    same wavelength grid, SNR, parameter ranges, stratified composition plan
    and cleaning - so cloud presence is the only difference from the clear
    test sets the models were evaluated on.

  * Cloud-top pressure is held FIXED within each batch rather than drawn from
    a range. Two reasons. First, MultiREx draws range parameters uniformly in
    linear space, so a (10, 1e5) range would place roughly 90% of planets
    above 1e4 Pa where suppression is under 11%, producing a dataset that is
    nearly cloud-free. Second, and more usefully, fixing the deck per batch
    turns the result into a degradation curve against cloud-top pressure
    rather than a single cloudy-versus-clear number.

  * Levels span the full physical range of the effect, measured on a fixed
    planet with all else held constant:

        1e5 Pa   0.6% feature suppression   (effectively clear)
        1e4 Pa  11.3%
        1e3 Pa  35.5%
        1e2 Pa  65.0%
        1e1 Pa  93.1%                       (nearly featureless)

    Clouds deeper than the region the spectrum probes have almost no effect,
    which is why the levels are log-spaced toward low pressure rather than
    spread uniformly through the atmosphere.

Usage:
    python generate_cloudy_testset.py            # all levels, ~600 planets each
    python generate_cloudy_testset.py --quick    # 120 planets each, for a smoke test
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

# Cloud-top pressures in Pa, log-spaced across the range where the effect
# transitions from negligible to near-total.
CLOUD_LEVELS = [1e5, 1e4, 1e3, 1e2, 1e1]

# Identical to generate_multiverse_data.py.
PROFILES = {
    "biosignature": ({"CH4": (BIO_CH4, -3), "O3": (BIO_O3, -1)}, 0.5),
    "nonbio_ch4":   ({"CH4": (BIO_CH4, -3), "O3": (-10, BIO_O3)}, 0.1666),
    "nonbio_o3":    ({"CH4": (-9, BIO_CH4), "O3": (BIO_O3, -1)}, 0.1666),
    "nonbio_none":  ({"CH4": (-9, BIO_CH4), "O3": (-10, BIO_O3)}, 0.1668),
}
BACKGROUND = {"H2O": (-10, -1), "CO": (-9, -3), "CO2": (-9, -3), "NH3": (-9, -3)}


def generate_one_level(cloud_pressure, n_universes, wn_grid):
    """Generate a stratified, class-balanced batch at one cloud-top pressure."""
    frames = []
    star = Star(temperature=(2500, 7500), radius=(0.1, 1.7), mass=(0.1, 1.7))

    for name, (chem, frac) in PROFILES.items():
        count = int(n_universes * frac)
        if count == 0:
            continue
        print(f"    {name:<14} n={count:<4} cloud={cloud_pressure:.0e} Pa")

        atmosphere = Atmosphere(
            temperature=(500, 2500),
            base_pressure=(1e5, 10e5),
            top_pressure=(1, 10),
            composition={**BACKGROUND, **chem},
            fill_gas=FILL_GAS,
            cloud_pressure=cloud_pressure,
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
    """Cleaning and labelling as in generate_multiverse_data.py, with dropna
    restricted to the spectral columns: the Mie-capable fork reports
    'atm cloud_model' as None (-> NaN) when the grey deck is used via
    cloud_pressure, and a blanket dropna() would silently delete every row."""
    fp = re.compile(r"^-?\d+\.\d+$")
    spectral = [c for c in df.columns
                if isinstance(c, float) or (isinstance(c, str) and fp.match(c))]
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
    print(f"Generating cloudy {FILL_GAS} test sets: {len(CLOUD_LEVELS)} levels "
          f"x {n} planets\n")

    for cp in CLOUD_LEVELS:
        tag = f"{cp:.0e}".replace("+", "").replace("e0", "e")
        print(f"--- cloud top {cp:.0e} Pa ---")
        df = clean_and_label(generate_one_level(cp, n, wn_grid))
        out = f"multirex_spectra_{FILL_GAS}_cloudy_{tag}Pa.parquet"
        df.to_parquet(out)
        print(f"    wrote {out}\n")

    print("Done. Evaluate with: python evaluate_cloudy.py")


if __name__ == "__main__":
    main()
