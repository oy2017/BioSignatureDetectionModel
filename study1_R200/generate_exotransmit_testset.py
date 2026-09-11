"""Generate Exo-Transmit spectra for the committed test planets.

Answers axis 1 of R1-3, "independent radiative transfer codes", with the
opacity data held exactly fixed.

Exo-Transmit is the code MultiREx's opacity tables came from: the .dat files in
multirex/data are byte-identical to Exo_Transmit/Opac (md5 verified). Running it
therefore changes the radiative transfer implementation - different language,
different authors, different solver - while every cross section stays the same
file. That isolates the code from the opacity data, which the axis-2 experiment
varies separately.

Nothing is resampled. Spectra are computed for the exact planets in the five
committed clear test sets, so each Exo-Transmit spectrum is paired with a TauREx
spectrum of the same planet and the 88.92% baseline stays valid.

Configuration matched to MultiREx's make_tm(), and the residual difference that
could NOT be matched, are both documented in exotransmit_harness.py. In short:
same absorbers, no CIA, Rayleigh on, 100 isothermal layers, radius at base
pressure - but Exo-Transmit assumes constant gravity where TauREx integrates
with gravity falling as altitude rises, which is a genuine difference between
the codes rather than a setting.

Usage:
    python generate_exotransmit_testset.py           # all five sets
    python generate_exotransmit_testset.py --quick   # 40 planets from set 1
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import exotransmit_harness as H

SOURCE_FMT = "multirex_spectra_H2_test_set_{}.parquet"
OUT_FMT = "multirex_spectra_H2_exotransmit_set_{}.parquet"
WORK_ROOT = "/tmp/exotransmit_workers"


def worker_spectra(rows, wl, widx):
    """One worker process, one private Exo-Transmit tree."""
    wd = H.make_workdir(os.path.join(WORK_ROOT, f"w{widx}"))
    H.write_selectchem(wd)
    out = np.empty((len(rows), len(wl)))
    for i, row in enumerate(rows):
        w, d = H.run_planet(wd, row)
        out[i] = H.bin_to_grid(w, d, wl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()

    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    os.makedirs(WORK_ROOT, exist_ok=True)

    sets = [1] if args.quick else [1, 2, 3, 4, 5]
    for si in sets:
        df = pd.read_parquet(SOURCE_FMT.format(si))
        if args.quick:
            df = df.iloc[:40].copy()
        cols = H.spectral_cols(df)
        wl = np.array([float(c) for c in cols])
        rows = [df.iloc[i] for i in range(len(df))]
        chunks = np.array_split(np.arange(len(rows)), args.jobs)
        print(f"--- set {si}: {len(df)} planets across {args.jobs} workers ---",
              flush=True)

        res = Parallel(n_jobs=args.jobs)(
            delayed(worker_spectra)([rows[i] for i in ch], wl, k)
            for k, ch in enumerate(chunks) if len(ch))
        spec = np.vstack(res)

        out = df.copy()
        out[cols] = spec
        s = out[cols]
        valid = s.notna().all(axis=1) & (s <= 1.0).all(axis=1)
        out["exotransmit_valid"] = valid.values

        # Labels come from the recorded abundances and must match the source.
        assert (out["biosignature"].values == df["biosignature"].values).all()
        out.to_parquet(OUT_FMT.format(si))

        v = valid.values
        a = df[cols].values
        b = spec
        amp_a = np.median(a[v].std(axis=1) / a[v].mean(axis=1))
        amp_b = np.median(b[v].std(axis=1) / b[v].mean(axis=1))
        cor = np.median([np.corrcoef(a[i], b[i])[0, 1] for i in np.where(v)[0]])
        print(f"    valid {int(v.sum())}/{len(df)}   "
              f"depth ratio {np.median(b[v].mean(axis=1) / a[v].mean(axis=1)):.4f}   "
              f"amp ratio {amp_b / amp_a:.3f}   correlation {cor:.4f}")
        print(f"    wrote {OUT_FMT.format(si)}\n", flush=True)

    print("Evaluate with: python evaluate_exotransmit.py")


if __name__ == "__main__":
    main()
