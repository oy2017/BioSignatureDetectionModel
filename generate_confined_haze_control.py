"""
Control set: the 0.50x haze confined to high altitude (P < 1e3 Pa).

The deck and the whole-column haze differ in two ways at once - wavelength
dependence of the opacity and vertical distribution - so their divergence at
matched feature amplitude could not, by itself, be attributed to chromaticity.
This set holds the haze density at the whole-column 0.50x value (2.4e8 m-3)
and restricts the haze to pressures below 1e3 Pa, the altitude of the
matched-amplitude deck. Everything else replicates generate_hazy_testset.py.

Result (recorded in revision_plan.md): the confined set reproduces the
whole-column set to within sampling noise (63.23% vs 63.25% XGBoost at
0.502x vs 0.503x amplitude), while the deck at the same amplitude scores
71.43% - so the deck-haze divergence is due to the wavelength dependence,
not the vertical distribution.

Usage:
    python generate_confined_haze_control.py
"""

import pandas as pd
from multirex import Atmosphere, Planet, Star, System, Physics

from generate_hazy_testset import (BACKGROUND, FILL_GAS, HAZE_Q, HAZE_RADIUS,
                                   PROFILES, SNR, clean_and_label)

MIX, BOTTOMP, N = 2.4e8, 1e3, 600
OUT = "multirex_spectra_H2_haze_confined_2p4e08.parquet"


def main():
    wn = Physics.wavenumber_grid(0.5, 7.8, 550)
    frames = []
    star = Star(temperature=(2500, 7500), radius=(0.1, 1.7), mass=(0.1, 1.7))
    for name, (chem, frac) in PROFILES.items():
        count = int(N * frac)
        print(f"    {name:<14} n={count:<4} confined haze {MIX:.1e} m^-3, "
              f"P < {BOTTOMP:.0e} Pa")
        atm = Atmosphere(
            temperature=(500, 2500), base_pressure=(1e5, 10e5),
            top_pressure=(1, 10), composition={**BACKGROUND, **chem},
            fill_gas=FILL_GAS,
            cloud_model={"type": "lee_mie", "radius": HAZE_RADIUS,
                         "q": HAZE_Q, "mix_ratio": MIX, "bottomP": BOTTOMP})
        planet = Planet(radius=(1.0, 26.0), mass=(1.0, 300.0), atmosphere=atm)
        system = System(planet=planet, star=star, sma=(0.01, 0.5))
        system.make_tm()
        res = system.explore_multiverse(wn_grid=wn, n_universes=count,
                                        n_observations=1, snr=SNR,
                                        header=True, path=None, n_jobs=-1)
        frames.append(res["spectra"])
    df = clean_and_label(pd.concat(frames, ignore_index=True))
    df.to_parquet(OUT)
    print(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
