"""
Band-resolved feature suppression for the paired aerosol sets (R1-3).

The deck-versus-haze comparison at matched overall suppression is the
scientific point of the aerosol work, and the mechanism argument rests on
*where* in the band each aerosol removes structure. This measures that
directly on the paired sets - the committed planets re-rendered with each
aerosol - so the numbers are consistent with H2_aerosol_paired.txt rather
than with the superseded resampled sets.

Amplitude is per-spectrum scatter within the band divided by that spectrum's
own mean depth, normalised by the same quantity on the clear training data.

Usage:
    python analyze_band_suppression.py
"""

import os
import re

import numpy as np
import pandas as pd

TRAIN = "multirex_spectra_H2_train.parquet"
BANDS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 7.8)]
SETS = [
    ("deck 1e5 Pa", "multirex_spectra_H2_paired_cloudy_1e5Pa.parquet"),
    ("deck 1e4 Pa", "multirex_spectra_H2_paired_cloudy_1e4Pa.parquet"),
    ("deck 1e3 Pa", "multirex_spectra_H2_paired_cloudy_1e3Pa.parquet"),
    ("deck 1e2 Pa", "multirex_spectra_H2_paired_cloudy_1e2Pa.parquet"),
    ("haze 2e5 m-3", "multirex_spectra_H2_paired_hazy_2p0e5.parquet"),
    ("haze 2e6 m-3", "multirex_spectra_H2_paired_hazy_2p0e6.parquet"),
    ("haze 3e7 m-3", "multirex_spectra_H2_paired_hazy_3p0e7.parquet"),
    ("haze 2.4e8 m-3", "multirex_spectra_H2_paired_hazy_2p4e8.parquet"),
    ("haze 1e10 m-3", "multirex_spectra_H2_paired_hazy_1p0e10.parquet"),
]


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(str(c)))],
                  key=float)


def main():
    tr = pd.read_parquet(TRAIN)
    cols = spectral_cols(tr)
    wl = np.array([float(c) for c in cols])
    Xt = tr[cols].to_numpy(float)

    def band_amp(X, lo, hi):
        m = (wl >= lo) & (wl < hi)
        return np.median(X[:, m].std(axis=1) / X.mean(axis=1))

    base = [band_amp(Xt, lo, hi) for lo, hi in BANDS]
    ref = np.median(Xt.std(axis=1) / Xt.mean(axis=1))

    hdr = (f"{'set':<16}{'overall':>9}" +
           "".join(f"{lo}-{hi}um".rjust(11) for lo, hi in BANDS))
    lines = ["Band-resolved feature suppression, paired aerosol sets",
             "",
             "Values are band feature amplitude relative to the clear training",
             "data. Measured on the paired sets (committed planets re-rendered",
             "with the aerosol), consistent with H2_aerosol_paired.txt.",
             "", hdr, "-" * len(hdr)]
    rows = []
    for name, f in SETS:
        d = pd.read_parquet(f)
        X = d[spectral_cols(d)].to_numpy(float)
        ov = np.median(X.std(axis=1) / X.mean(axis=1)) / ref
        r = [band_amp(X, lo, hi) / b for (lo, hi), b in zip(BANDS, base)]
        lines.append(f"{name:<16}{ov:>9.2f}" + "".join(f"{v:11.3f}" for v in r))
        rows.append((name, ov, *r))

    lines += [
        "",
        "=" * 72,
        "The matched-amplitude pair is deck 1e3 Pa (0.58x) and haze 3e7 m-3",
        "(0.59x). At that match the deck concentrates its muting at short",
        "wavelengths (0.085x at 0.5-1um) while the haze preserves every band",
        "(0.77-0.92x) - yet the haze scores 65.27% against the deck's 71.64%.",
        "The haze retains more diagnostic structure in every band and still",
        "classifies worse, so its excess damage cannot be information loss.",
        "",
        "Note the haze's chromatic signature is not monotonic in density: at",
        "2e6 m-3 it raises blue-band amplitude above clear (1.486x) by adding",
        "a steep short-wavelength slope, and only at 2.4e8 and above does it",
        "extinguish the blue.",
    ]

    out = "\n".join(lines)
    print(out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_band_suppression.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["set", "overall"] +
                 [f"{lo}-{hi}um" for lo, hi in BANDS]).to_csv(
        "final_results/H2_band_suppression.csv", index=False)
    print("\nWrote final_results/H2_band_suppression.{txt,csv}")


if __name__ == "__main__":
    main()
