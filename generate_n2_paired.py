"""
Composition transfer H2 -> N2 (R1-1, axis 7): render the committed test planets
with an N2 fill gas instead of H2, paired planet-by-planet.

Same abundances, radii, pressures, star, and planet as the committed H2 test
sets -- only the background gas changes (H2, mu~2, -> N2, mu~28), which shrinks
the atmospheric scale height and mutes every feature. This isolates the
composition domain shift: the frozen H2-trained pipeline is later evaluated on
these N2 spectra against the committed H2 baseline (evaluate_n2.py).

Re-render, not resample -- the H2 path reproduces the committed spectra to
floating-point precision, so no generator drift.
"""
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]
SRC = "multirex_spectra_H2_test_set_{}.parquet"
OUT = "multirex_spectra_N2_paired_set_{}.parquet"
fp = re.compile(r"^-?\d+\.\d+$")


def spectral_cols(df):
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))], key=float)


def one(row, cols, wn, fill):
    from multirex import Atmosphere, Planet, Star, System
    atm = Atmosphere(temperature=float(row["atm temperature"]),
                     base_pressure=float(row["atm base_pressure"]),
                     top_pressure=float(row["atm top_pressure"]),
                     composition={g: float(row[f"atm {g}"]) for g in GASES}, fill_gas=fill)
    s = System(planet=Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]),
               atmosphere=atm),
               star=Star(temperature=float(row["s temperature"]), radius=float(row["s radius"]),
               mass=float(row["s mass"])), sma=float(row["sma"]))
    s.make_tm()
    return np.asarray(s.generate_spectrum(wn)[1])[::-1]     # ascending wavelength


def main():
    for i in range(1, 6):
        df = pd.read_parquet(SRC.format(i))
        cols = spectral_cols(df)
        wl = np.array([float(c) for c in cols]); wn = 1e4 / wl[::-1]
        rows = [df.iloc[j] for j in range(len(df))]

        # validate on the first planet: H2 render must reproduce the committed spectrum
        h2 = one(rows[0], cols, wn, "H2")
        rel = np.max(np.abs(h2 - df[cols].values[0]) / np.abs(df[cols].values[0]))
        assert rel < 1e-10, f"set {i}: H2 re-render does not reproduce committed ({rel:.1e})"

        spec = Parallel(n_jobs=-1, verbose=0)(
            delayed(one)(r, cols, wn, "N2") for r in rows)
        out = df.copy()
        out[cols] = np.vstack(spec)
        out.to_parquet(OUT.format(i))
        amp_h2 = np.median(df[cols].values.std(1) / df[cols].values.mean(1))
        amp_n2 = np.median(np.vstack(spec).std(1) / np.vstack(spec).mean(1))
        print(f"set {i}: {len(df)} planets rendered with N2  "
              f"(H2 repro {rel:.1e}; feature amp H2 {amp_h2:.5f} -> N2 {amp_n2:.5f}, "
              f"{amp_n2/amp_h2:.2f}x)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
