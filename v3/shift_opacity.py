"""Opacity-database swap on the v2 grid (R1-3 axis 2, native resolution).

Re-renders the committed planets of the given splits with MultiREx while
TauREx's OpacityCache points at a directory of symlinks that replaces the
Exo-Transmit tables shipped with MultiREx by ExoMolOP tables. Every output row
is paired one-to-one with the stored noise-free native spectrum in
{split}_native.npy, so the effect of the opacity data alone can be measured
per planet.

Variants
    exomol      H2O -> ExoMol POKAZATEL, CH4 -> ExoMol YT34to10,
                CO2 -> ExoMol UCL-4000; O3 and O2 keep the Exo-Transmit tables
    exomol_o3   the above plus O3 -> converted HITRAN table (~/exomolop_o3)
CO and NH3 have no opacity data in either database and act only through the
mean molecular weight, exactly as in the baseline.

Outputs (v2/data/), float32 (n, 2753), same row order as {split}_params.parquet,
NaN rows where the forward model failed:
    {split}_native_exomol.npy
    {split}_native_exomol_o3.npy
With --limit N the file is named {split}_native_{variant}.partialN.npy so a
partial run is never mistaken for a full one.

Mapping onto the old generate_opacity_swap_testset.py
    build_swap_dir        same idea; one directory per variant so both can be
                          rendered in one process without rebuilding
    configure_opacities   same (pid, path) key - see the note below
    regenerate            -> render(): generate_grid.build_system() instead of a
                          local copy of the System construction, and the native
                          TauREx model (system.transmission.model()) instead of
                          generate_spectrum() on the R = 550 grid
    validate              same check (original tables must reproduce the stored
                          spectra to floating-point precision), then renders the
                          same planets with each swap and reports the median
                          per-planet Pearson r and mean-depth ratio
    generate              -> run_split(), with the same guard that the swapped
                          output must differ from the baseline

Native grid under the swap
    With the Exo-Transmit tables TauREx's native grid is the 2,753-point R ~ 1000
    grid stored in native_wl.npy, and the model is returned on it unchanged.
    With the ExoMolOP tables (R = 15,000) the model comes back on a 41,211-point
    grid, so it must be reduced to native_wl. generate_grid.one_native() would
    np.interp, i.e. point-sample an R = 15,000 line spectrum at R = 1,000 nodes;
    that injects line-sampling noise that inflates the measured opacity effect
    and would be carried into every binned configuration. The default here
    (--reduce bin) flux-averages the fine model over bins whose edges are the
    midpoints between native nodes, using bin_spectra.bin_native, the same
    operation the old script's generate_spectrum() performed on its R = 550
    grid. --reduce interp keeps the literal one_native() behaviour so the two
    can be compared.

Per-process opacity configuration
    The configured-path record is keyed by (pid, path), not by path. When this
    file runs as a script it is __main__, and joblib pickles the worker by
    value with cloudpickle, carrying module globals into every worker; a
    path-only key would make each worker believe it is configured, skip
    set_opacity_path, and silently render with the tables MultiREx installs at
    import. run_split() additionally refuses to write output that matches the
    baseline.

Usage
    python shift_opacity.py --validate
    python shift_opacity.py --splits test1 --variant exomol --limit 20 --jobs 2
    python shift_opacity.py --variant both --jobs 4
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
EXOMOLOP_DIR = os.path.expanduser("~/exomolop")
O3_DIR = os.path.expanduser("~/exomolop_o3")            # converted HITRAN ozone
SWAP_DIRS = {"exomol": os.path.expanduser("~/opacity_swap_v3_exomol"),
             "exomol_o3": os.path.expanduser("~/opacity_swap_v3_exomol_o3")}
EXOMOL_MOL = {"1H2-16O": "H2O", "12C-1H4": "CH4", "12C-16O2": "CO2", "12C-16O": "CO"}   # v3: CO added
TEST_SPLITS = [f"test{k}" for k in range(1, 6)]
CHUNK = 25
POLL_SECONDS, POLL_MAX_SECONDS = 60, 30 * 60

_configured = {"key": None}


# ----------------------------------------------------------------- opacities
def original_opacity_dir():
    import multirex
    return os.path.join(os.path.dirname(multirex.__file__), "data")


def build_swap_dir(variant):
    """Symlink directory with exactly one opacity file per molecule."""
    if variant not in SWAP_DIRS:
        raise ValueError(f"unknown variant {variant!r}")
    swap = SWAP_DIRS[variant]
    os.makedirs(swap, exist_ok=True)
    for f in os.listdir(swap):
        os.unlink(os.path.join(swap, f))

    swapped = {}
    for fn in sorted(os.listdir(EXOMOLOP_DIR)):
        if not fn.endswith(".h5"):
            continue
        mol = EXOMOL_MOL.get(fn.split("__")[0])
        if mol is None:
            raise ValueError(f"unrecognised ExoMolOP file: {fn}")
        os.symlink(os.path.join(EXOMOLOP_DIR, fn), os.path.join(swap, fn))
        swapped[mol] = fn
    if variant == "exomol_o3":
        o3 = [f for f in sorted(os.listdir(O3_DIR)) if f.endswith(".h5")]
        if len(o3) != 1:
            raise FileNotFoundError(f"expected one .h5 in {O3_DIR}, found {o3}")
        os.symlink(os.path.join(O3_DIR, o3[0]), os.path.join(swap, o3[0]))
        swapped["O3"] = o3[0]

    retained = {}
    orig = original_opacity_dir()
    for mol in ("O3", "O2", "H2O", "CH4", "CO2"):
        if mol in swapped:
            continue
        src = os.path.join(orig, f"opac{mol}.dat")
        if os.path.exists(src):
            os.symlink(src, os.path.join(swap, f"opac{mol}.dat"))
            retained[mol] = f"opac{mol}.dat"

    expected = {"exomol": {"H2O", "CH4", "CO2"}, "exomol_o3": {"H2O", "CH4", "CO2", "O3"}}
    if set(swapped) != expected[variant]:
        raise RuntimeError(f"{variant}: swapped {sorted(swapped)}, expected {sorted(expected[variant])}")
    print(f"swap dir [{variant}] {swap}")
    for m, f in sorted(swapped.items()):
        print(f"  swapped  {m:<4} -> {f}")
    for m, f in sorted(retained.items()):
        print(f"  retained {m:<4} -> {f}")
    return swap


def configure_opacities(path):
    """Point TauREx at `path`. Idempotent per (process, path)."""
    key = (os.getpid(), path)
    if _configured["key"] == key:
        return
    import multirex  # noqa: F401  (sets its own path at import)
    from taurex.cache import OpacityCache
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(path)
    _configured["key"] = key


# ----------------------------------------------------------------- rendering
def native_edges(wl_native):
    """Bin edges at the midpoints between native nodes, extended half a step at each end."""
    wl = np.asarray(wl_native, dtype=np.float64)
    mid = 0.5 * (wl[1:] + wl[:-1])
    return np.concatenate([[wl[0] - (mid[0] - wl[0])], mid, [wl[-1] + (wl[-1] - mid[-1])]])


def render(row, wl_native, opacity_path, reduce="bin"):
    """Noise-free native-resolution transit depth on wl_native under the given
    opacity path, or None on failure. Mirrors generate_grid.one_native, except
    that a model returned on a finer grid is flux-averaged onto the native
    bins (reduce='bin') rather than point-sampled (reduce='interp')."""
    configure_opacities(opacity_path)
    from generate_grid import WL_MAX, WL_MIN, build_system, one_native
    if reduce == "interp":
        return one_native(row, wl_native)
    try:
        system = build_system(row)
        wn, rprs = system.transmission.model()[:2]
        wl = 1e4 / np.asarray(wn, dtype=np.float64)
        order = np.argsort(wl)
        wl, rprs = wl[order], np.asarray(rprs, dtype=np.float64)[order]
        m = (wl >= WL_MIN) & (wl <= WL_MAX)
        if m.sum() == wl_native.shape[0] and np.allclose(wl[m], wl_native):
            y = rprs[m]
        else:
            from bin_spectra import bin_native
            y = bin_native(rprs[None, :], wl, native_edges(wl_native))[0].astype(np.float64)
        if not np.all(np.isfinite(y)) or np.any(y > 1.0) or np.any(y <= 0):
            return None
        return y.astype(np.float32)
    except Exception:
        return None


def worker(rows, wl_native, opacity_path, reduce):
    import warnings
    warnings.filterwarnings("ignore")
    return [render(r, wl_native, opacity_path, reduce) for r in rows]


def render_rows(rows, wl_native, opacity_path, jobs, reduce="bin"):
    """Render a list of parameter dicts; returns float32 (n, len(wl_native))
    with NaN rows for failures, and the number of failures."""
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    out = Parallel(n_jobs=jobs, verbose=0)(
        delayed(worker)(c, wl_native, opacity_path, reduce) for c in chunks)
    specs = [s for c in out for s in c]
    X = np.full((len(rows), wl_native.shape[0]), np.nan, dtype=np.float32)
    n_fail = 0
    for i, s in enumerate(specs):
        if s is None:
            n_fail += 1
        else:
            X[i] = s
    return X, n_fail


# ----------------------------------------------------------------- data
def wait_for(paths):
    """Poll until every path exists (the grid generator may still be writing)."""
    t0 = time.time()
    while True:
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            return
        if time.time() - t0 > POLL_MAX_SECONDS:
            raise FileNotFoundError(f"still missing after {POLL_MAX_SECONDS // 60} min: {missing}")
        print(f"waiting for {[os.path.basename(p) for p in missing]} ...", flush=True)
        time.sleep(POLL_SECONDS)


def load_split(split, limit=None):
    pp = os.path.join(DATA, f"{split}_params.parquet")
    npth = os.path.join(DATA, f"{split}_native.npy")
    wait_for([pp, npth, os.path.join(DATA, "native_wl.npy")])
    # both files are written by generate_grid after the full split renders,
    # params first; a short re-check guards against reading a half-written .npy
    params = pd.read_parquet(pp)
    for _ in range(5):
        try:
            stored = np.load(npth)
            if stored.shape[0] == len(params):
                break
        except (ValueError, EOFError):
            pass
        time.sleep(10)
    else:
        raise RuntimeError(f"{split}: {npth} does not match {pp}")
    wl_native = np.load(os.path.join(DATA, "native_wl.npy"))
    if limit is not None:
        params, stored = params.head(limit), stored[:limit]
    rows = [params.iloc[i].to_dict() for i in range(len(params))]
    return rows, stored, wl_native


# ----------------------------------------------------------------- drivers
def compare(swapped, stored):
    """Per-planet Pearson r and mean-depth ratio (swapped / stored), finite rows only."""
    ok = np.isfinite(swapped).all(axis=1)
    r = np.array([np.corrcoef(swapped[i], stored[i])[0, 1] for i in np.flatnonzero(ok)])
    ratio = swapped[ok].mean(axis=1) / stored[ok].mean(axis=1)
    rel = np.abs(swapped[ok] - stored[ok]) / stored[ok]
    return r, ratio, rel


def run_split(split, variant, jobs, limit=None, reduce="bin"):
    path = build_swap_dir(variant)
    rows, stored, wl_native = load_split(split, limit)
    print(f"--- {split} [{variant}]: {len(rows)} planets, jobs={jobs}, reduce={reduce} ---", flush=True)
    t0 = time.time()
    X, n_fail = render_rows(rows, wl_native, path, jobs, reduce)
    dt = time.time() - t0
    print(f"    {dt:.0f} s total, {dt / len(rows):.2f} s/planet (wall, jobs={jobs}); "
          f"{n_fail} failures (NaN rows)")

    r, ratio, rel = compare(X, stored)
    if np.median(rel) < 1e-6:
        raise RuntimeError(
            f"{split} [{variant}]: output matches the baseline (median relative "
            f"difference {np.median(rel):.2e}). The original opacities were used. "
            f"Do not trust any result from this run.")
    print(f"    vs baseline: median rel diff {np.median(rel):.3e}, max {rel.max():.3e}; "
          f"median Pearson r {np.median(r):.6f}, median mean-depth ratio {np.median(ratio):.6f}")

    name = f"{split}_native_{variant}" + (f".partial{limit}" if limit is not None else "") + ".npy"
    out = os.path.join(DATA, name)
    np.save(out, X)
    print(f"    wrote {out} {X.shape}\n", flush=True)
    return X


def validate(jobs, n=20, split="test1", reduce="bin"):
    rows, stored, wl_native = load_split(split, n)
    orig = original_opacity_dir()
    print(f"validating on the first {len(rows)} planets of {split}\n"
          f"original opacities: {orig}")
    t0 = time.time()
    X, n_fail = render_rows(rows, wl_native, orig, jobs, reduce)
    dt = time.time() - t0
    resid = np.abs(X.astype(np.float64) - stored.astype(np.float64))
    rel = resid / stored
    ok = n_fail == 0 and np.nanmax(rel) < 1e-9
    print(f"  {dt / len(rows):.2f} s/planet (wall, jobs={jobs}); {n_fail} failures\n"
          f"  max |abs residual| = {np.nanmax(resid):.3e}   max |rel residual| = {np.nanmax(rel):.3e}")
    print("  PASS - harness reproduces the stored native spectra" if ok else
          "  FAIL - do not proceed; the opacity path handling is wrong")
    if not ok:
        return False

    for variant in ("exomol", "exomol_o3"):
        path = build_swap_dir(variant)
        for red in ([reduce] if reduce == "interp" else ["bin", "interp"]):
            t0 = time.time()
            Y, n_fail = render_rows(rows, wl_native, path, jobs, red)
            dt = time.time() - t0
            r, ratio, relv = compare(Y, stored)
            print(f"  [{variant:<9} reduce={red:<6}] {dt / len(rows):.2f} s/planet, {n_fail} failures; "
                  f"median Pearson r {np.median(r):.6f} (min {r.min():.6f}), "
                  f"median mean-depth ratio {np.median(ratio):.6f} "
                  f"(range {ratio.min():.6f}-{ratio.max():.6f}), "
                  f"median rel diff {np.median(relv):.3e}")
            if np.median(relv) < 1e-6:
                print("  FAIL - swapped output matches the baseline; the swap was not applied")
                return False
        print()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", nargs="+", default=TEST_SPLITS)
    ap.add_argument("--variant", choices=["exomol", "exomol_o3", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="first N planets of each split")
    ap.add_argument("--jobs", type=int, default=4, help="workers; each holds ~1 GB of tables")
    ap.add_argument("--reduce", choices=["bin", "interp"], default="bin",
                    help="how a finer TauREx grid is brought onto native_wl (see docstring)")
    ap.add_argument("--validate", action="store_true",
                    help="check the original tables reproduce test1, then size the swap effect")
    a = ap.parse_args()
    if a.validate:
        sys.exit(0 if validate(a.jobs, reduce=a.reduce) else 1)
    variants = ["exomol", "exomol_o3"] if a.variant == "both" else [a.variant]
    for split in a.splits:
        for variant in variants:
            run_split(split, variant, a.jobs, a.limit, a.reduce)


if __name__ == "__main__":
    main()
