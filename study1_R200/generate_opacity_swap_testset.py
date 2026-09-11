"""
Opacity-database swap on the committed test planets (R1-3 axis 2).

Reviewer 1 asked for validation against "alternative molecular opacity
databases". This regenerates the forward model for the **exact planets in the
five committed clear test sets**, changing only the molecular opacity data.

Why paired rather than resampled
--------------------------------
Newly drawn MultiREx planets do not follow the same parameter distribution as
the committed sets (see final_results/H2_generator_drift.txt), so a freshly
sampled comparison set would confound the opacity change with a sampling shift
of several accuracy points. Regenerating the committed planets from their own
recorded parameters removes that entirely: every planet in the swapped set has
a one-to-one partner in the baseline, with identical radius, mass, temperature,
pressures, composition, star and orbit. The only difference is the opacity
tables.

This also means the comparison can be made per planet, not just per
distribution, and the committed 88.92% baseline stays valid.

What is and is not swapped
--------------------------
MultiREx ships Exo-Transmit opacity tables for five molecules (CH4, CO2, H2O,
O2, O3). Of the six gases in the composition, CO and NH3 have no opacity data
in either database and act only through mean molecular weight; O2 is unused.

  swapped   H2O  Exo-Transmit -> ExoMol POKAZATEL
            CH4  Exo-Transmit -> ExoMol YT34to10
            CO2  Exo-Transmit -> ExoMol UCL-4000
  retained  O3   Exo-Transmit (ExoMolOP provides no ozone opacity)
            O2   Exo-Transmit (unused by the composition)

O3 is one of the two label-determining molecules, so the measured effect is a
**lower bound** on a full database change. State that wherever the result is
quoted.

Usage:
    python generate_opacity_swap_testset.py --validate   # run this FIRST
    python generate_opacity_swap_testset.py --generate
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

TEST_FMT = "multirex_spectra_H2_test_set_{}.parquet"
OUT_FMT = "multirex_spectra_H2_opacityswap_set_{}.parquet"
EXOMOLOP_DIR = os.path.expanduser("~/exomolop")
O3_DIR = os.path.expanduser("~/exomolop_o3")   # converted HITRAN ozone
SWAP_DIR = os.path.expanduser("~/opacity_swap")
GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]
WL_MIN, WL_MAX, RESOLUTION = 0.5, 7.8, 550
N_JOBS = 6          # each worker holds ~1 GB of opacity tables
FLOAT_RE = re.compile(r"^-?\d+\.\d+$")

# Keyed by (pid, path), NOT by path alone. When this module is run as a script
# it is __main__, so joblib pickles the worker function by value with
# cloudpickle - which carries these globals into the worker. A path-only key
# then makes every worker believe it is already configured, skip
# set_opacity_path, and silently use whatever opacities MultiREx installs at
# import. That produced a full run of baseline-identical spectra that looked
# like a clean "no effect" result. Including the pid forces each process to
# configure itself.
_configured = {"key": None}


def original_opacity_dir():
    import multirex
    return os.path.join(os.path.dirname(multirex.__file__), "data")


def build_swap_dir(only=None, with_o3=False):
    """Symlink directory holding exactly one opacity file per molecule.

    Explicit symlinks rather than a two-entry search path, so there is no
    ambiguity about which file wins for a molecule present in both databases.

    `only` restricts the swap to the named molecules, leaving the rest on
    their original tables - used for the per-molecule ablation that
    attributes the total effect to individual absorbers.
    """
    os.makedirs(SWAP_DIR, exist_ok=True)
    for f in os.listdir(SWAP_DIR):
        os.unlink(os.path.join(SWAP_DIR, f))

    orig_dir = original_opacity_dir()
    swapped = {}
    for fn in os.listdir(EXOMOLOP_DIR):
        if not fn.endswith(".h5"):
            continue
        mol = {"1H2-16O": "H2O", "12C-1H4": "CH4", "12C-16O2": "CO2"}.get(
            fn.split("__")[0])
        if mol is None:
            raise ValueError(f"unrecognised ExoMolOP file: {fn}")
        if only is not None and mol not in only:
            continue          # leave this molecule on its original table
        os.symlink(os.path.join(EXOMOLOP_DIR, fn), os.path.join(SWAP_DIR, fn))
        swapped[mol] = fn

    # Converted HITRAN ozone, if present and requested.
    if with_o3 and os.path.isdir(O3_DIR):
        for fn in os.listdir(O3_DIR):
            if fn.endswith(".h5") and (only is None or "O3" in only):
                os.symlink(os.path.join(O3_DIR, fn),
                           os.path.join(SWAP_DIR, fn))
                swapped["O3"] = fn

    orig = orig_dir
    retained = {}
    for mol in ("O3", "O2", "H2O", "CH4", "CO2"):
        if mol in swapped:
            continue
        src = os.path.join(orig, f"opac{mol}.dat")
        if os.path.exists(src):
            os.symlink(src, os.path.join(SWAP_DIR, f"opac{mol}.dat"))
            retained[mol] = f"opac{mol}.dat"

    print(f"swap dir {SWAP_DIR}")
    for m, f in sorted(swapped.items()):
        print(f"  swapped  {m:<4} -> {f}")
    for m, f in sorted(retained.items()):
        print(f"  retained {m:<4} -> {f}")
    return SWAP_DIR


def configure_opacities(path):
    """Point TauREx at `path`. Idempotent per process; safe in every worker."""
    key = (os.getpid(), path)
    if _configured["key"] == key:
        return
    import multirex  # noqa: F401  (sets its own path at import time)
    from taurex.cache import OpacityCache
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(path)
    _configured["key"] = key


def spectral_cols(df):
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and FLOAT_RE.match(c))],
                  key=float)


def regenerate(row, opacity_path):
    """Forward-model one committed planet from its recorded parameters.

    Returns the spectrum on ascending wavelength, matching the parquet column
    order. Scalar (not range) parameters are passed, so generation is
    deterministic and reproduces the committed spectra exactly when the
    original opacities are used - which is what --validate checks.
    """
    configure_opacities(opacity_path)
    from multirex import Atmosphere, Planet, Star, System, Physics

    wn_grid = Physics.wavenumber_grid(WL_MIN, WL_MAX, RESOLUTION)
    atm = Atmosphere(
        temperature=float(row["atm temperature"]),
        base_pressure=float(row["atm base_pressure"]),
        top_pressure=float(row["atm top_pressure"]),
        composition={g: float(row[f"atm {g}"]) for g in GASES},
        fill_gas=str(row["atm fill_gas"]))
    planet = Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]),
                    atmosphere=atm)
    star = Star(temperature=float(row["s temperature"]),
                radius=float(row["s radius"]), mass=float(row["s mass"]))
    system = System(planet=planet, star=star, sma=float(row["sma"]))
    system.make_tm()
    bin_wn, bin_rprs = system.generate_spectrum(wn_grid)
    wl = 10000.0 / np.asarray(bin_wn)
    return np.asarray(bin_rprs)[np.argsort(wl)]


def run(df, opacity_path, n_jobs=N_JOBS):
    rows = [df.iloc[i] for i in range(len(df))]
    out = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(regenerate)(r, opacity_path) for r in rows)
    return np.vstack(out)


def validate(n_planets):
    """Regenerate committed planets with the ORIGINAL opacities.

    Must reproduce the stored spectra to floating-point precision. If this
    fails, the harness is wrong and no swapped result means anything.
    """
    orig = original_opacity_dir()
    print(f"validating against original opacities: {orig}\n")
    worst = 0.0
    for i in range(1, 6):
        df = pd.read_parquet(TEST_FMT.format(i))
        cols = spectral_cols(df)
        sub = df.head(n_planets)
        regen = run(sub, orig)
        stored = sub[cols].to_numpy(dtype=float)
        resid = np.abs(stored - regen).max()
        rel = np.abs((stored - regen) / stored).max()
        worst = max(worst, rel)
        print(f"  set {i}: n={len(sub)}  max|abs diff|={resid:.3e}  "
              f"max|rel diff|={rel:.3e}")
    print(f"\nworst relative difference: {worst:.3e}")
    ok = worst < 1e-9
    print("PASS - harness reproduces the committed spectra" if ok else
          "FAIL - do not proceed; the harness does not reproduce the data")
    return ok


def generate(only=None, tag="", with_o3=False):
    path = build_swap_dir(only, with_o3)
    configure_opacities(path)
    from taurex.cache import OpacityCache
    print("molecules visible to TauREx:",
          sorted(OpacityCache().find_list_of_molecules()), "\n")

    for i in range(1, 6):
        df = pd.read_parquet(TEST_FMT.format(i))
        cols = spectral_cols(df)
        print(f"--- set {i}: {len(df)} planets ---")
        spectra = run(df, path)

        # Guard against silently generating with the wrong opacity tables.
        # A first attempt at this experiment produced output identical to the
        # baseline to ~1e-15 - i.e. the original tables were used - despite the
        # cache reporting the swapped path in every worker. The cause was never
        # identified, so the check is enforced here rather than trusted:
        # swapped spectra must differ from the stored ones by far more than
        # regeneration noise (~1e-14 relative, see --validate).
        stored = df[cols].to_numpy(dtype=float)
        rel = np.abs((spectra - stored) / stored)
        if np.median(rel) < 1e-6:
            raise RuntimeError(
                f"set {i}: output matches the baseline (median relative "
                f"difference {np.median(rel):.2e}). The original opacities "
                f"were used. Do not trust any result from this run.")
        print(f"    differs from baseline: median rel {np.median(rel):.3e}, "
              f"max rel {rel.max():.3e}")

        out = df.copy()
        out[cols] = spectra
        bad = ~np.isfinite(spectra).all(axis=1) | (spectra > 1.0).any(axis=1)
        if bad.any():
            print(f"    dropping {bad.sum()} invalid rows")
            out = out[~bad]
        name = OUT_FMT.format(i).replace(".parquet", f"{tag}.parquet")
        out.to_parquet(name)
        print(f"    wrote {name} ({len(out)} rows)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="check the harness reproduces committed spectra")
    ap.add_argument("--generate", action="store_true",
                    help="generate the opacity-swapped test sets")
    ap.add_argument("--n", type=int, default=20,
                    help="planets per set for --validate")
    ap.add_argument("--with-o3", action="store_true",
                    help="also swap O3 to the converted HITRAN table")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated molecules to swap (ablation), "
                         "e.g. H2O; others keep their original tables")
    args = ap.parse_args()
    if args.validate:
        validate(args.n)
    elif args.generate:
        only = args.only.split(",") if args.only else None
        tag = f"_{args.only}" if args.only else ""
        generate(only, tag + ('_o3' if args.with_o3 else ''), args.with_o3)
    else:
        ap.error("choose --validate or --generate")


if __name__ == "__main__":
    main()
