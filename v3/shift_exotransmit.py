"""Re-render the v2 test planets with Exo-Transmit (the "different code" shift).

For each split, every planet in data/{split}_params.parquet is run through the
Exo-Transmit binary at ~/exotransmit_src and the resulting transit depth is
written to data/{split}_native_exotransmit.npy: float32, shape
(n_planets, 2753), the same row order as the params file and exactly the
native wavelength grid data/native_wl.npy that the MultiREx/TauREx spectra in
data/{split}_native.npy sit on. Planets for which Exo-Transmit fails are NaN
rows (counted and reported, never fatal).

How this maps onto the old driver (../exotransmit_harness.py and
../generate_exotransmit_testset.py):

  * The Exo-Transmit configuration is unchanged and is reproduced verbatim:
    absorbers CH4, CO2, H2O, O3 and, in v3, CO (opacCO.dat added to MultiREx and
    present in Exo-Transmit's Opac/); NH3 now absorbs in both codes (opacNH3.dat added to MultiREx); Collision Induced
    Absorption off; Rayleigh on with augmentation 1.0; 100 isothermal layers
    between top and base pressure (otherInput.in's optical-depth count is set
    to 100 to match TauREx); planet radius at the base of the atmosphere;
    cloud-top pressure 0 (clear). The per-planet T_P, EOS and userInput.in
    writers are the old harness's, and Exo-Transmit's percent output is
    divided by 100 to give a fraction, as before.
  * The old input was a parquet with the spectrum AND the parameters as
    columns, and the output grid was the parquet's 550 spectral columns. Here
    the parameters come from {split}_params.parquet (same column names, the
    generate_grid.py sampling) and the target grid is native_wl.npy (2,753
    points, R ~ 1000), so the binned product is a native-resolution array
    that bin_spectra.py / noise.py can treat exactly like {split}_native.npy.
  * The binning is the old bin_to_grid: Exo-Transmit points are averaged
    within bins whose edges are the geometric midpoints between consecutive
    native wavelengths; bins that receive no point (Exo-Transmit's own grid is
    also R ~ 1000, so this happens) are filled by linear interpolation from
    their neighbours.
  * Parallelism is joblib over planets. Each loky worker process owns one
    private Exo-Transmit tree under /tmp/exotransmit_workers_v2/w<pid>
    (binary copied, otherInput.in patched, Opac symlinked because it is
    250 MB), created lazily on the worker's first planet and reused.
  * The old validation record (bottom of exotransmit_harness.py) stands: mean
    depth agrees with TauREx to ~0.4%, while spectral contrast is 4-12% lower
    because Exo-Transmit assumes constant gravity through the atmosphere. So
    --validate should show depth ratios near 1.00 and correlations > 0.99.

Usage:
    python shift_exotransmit.py --validate                 # 20 planets of test1 vs MultiREx
    python shift_exotransmit.py --limit 20 --jobs 4        # timing run
    python shift_exotransmit.py --jobs 8                   # all five test splits
    python shift_exotransmit.py --splits test1 --jobs 8    # one split
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SRC = os.path.expanduser("~/exotransmit_src")
WORK_ROOT = "/tmp/exotransmit_workers_v2"
DEFAULT_SPLITS = ["test1", "test2", "test3", "test4", "test5"]

R_EARTH_M, R_SUN_M, M_EARTH_KG = 6.3781e6, 6.957e8, 5.972167867791379e24
G_SI = 6.67430e-11
N_LAYERS = 100
EXO_TIMEOUT_S = 1800

GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]
ABSORBERS = {"CH4", "CO2", "H2O", "O3", "CO", "NH3"}   # v3: CO and NH3 carry opacity in both codes (opacCO.dat, opacNH3.dat)

EOS_SPECIES = ("C CH4 CO COS CO2 C2H2 C2H4 C2H6 H HCN HCl HF H2 H2CO H2O H2S "
               "He K MgH N N2 NO2 NH3 NO Na O O2 O3 OH PH3 SH SO2 SiH SiO TiO "
               "VO").split()
EOS_T = [3000 - 100 * i for i in range(30)]          # 3000 down to 100 K
EOS_P = [10.0 ** e for e in range(8, -5, -1)]        # 1e8 down to 1e-4 Pa

SELECT_ORDER = ("CH4 CO2 CO H2O NH3 O2 O3 C2H2 C2H4 C2H6 H2CO H2S HCl HCN HF "
                "MgH N2 NO NO2 OCS OH PH3 SH SiH SiO SO2 TiO VO Na K").split()


# --- Exo-Transmit work tree --------------------------------------------------

def make_workdir(path):
    """A private Exo-Transmit tree; Opac is symlinked since it is 250 MB.

    Idempotent, so a persistent worker can call it on every planet cheaply.
    """
    os.makedirs(path, exist_ok=True)
    for d in ("T_P", "EOS", "Spectra"):
        os.makedirs(os.path.join(path, d), exist_ok=True)
    opac = os.path.join(path, "Opac")
    if not os.path.exists(opac):
        os.symlink(os.path.join(SRC, "Opac"), opac)
    binary = os.path.join(path, "Exo_Transmit")
    if not os.path.exists(binary):
        shutil.copy(os.path.join(SRC, "Exo_Transmit"), binary)
    other_path = os.path.join(path, "otherInput.in")
    if not os.path.exists(other_path):
        # otherInput.in carries the grid constants. The optical-depth count
        # must equal the number of layers in the T_P file; Exo-Transmit ships
        # 334, TauREx uses 100.
        other = open(os.path.join(SRC, "otherInput.in")).read().split("\n")
        other[47] = str(N_LAYERS)
        with open(other_path, "w") as f:
            f.write("\n".join(other))
    if not os.path.exists(os.path.join(path, "selectChem.in")):
        write_selectchem(path)
    return path


def write_selectchem(path):
    lines = ["For each gas, place a 1 after the equals sign if the gas opacity "
             "will be present in your transmission calculations, and 0 if not. "
             "Do not change the order of the gases in this file!"]
    for g in SELECT_ORDER:
        lines.append(f"{g} = {1 if g in ABSORBERS else 0}")
    lines.append("Scattering = 1")
    lines.append("Collision Induced Absorption = 0")
    lines.append("Do not delete this line!")
    with open(os.path.join(path, "selectChem.in"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_tp(path, row, name="t_p_planet.dat"):
    p_top = float(row["atm top_pressure"])
    p_base = float(row["atm base_pressure"])
    T = float(row["atm temperature"])
    ps = np.logspace(np.log10(p_top), np.log10(p_base), N_LAYERS)
    with open(os.path.join(path, "T_P", name), "w") as f:
        f.write("    i\tP\tT\n")
        for i, p in enumerate(ps):
            f.write(f"    {i}\t{p:.7E}\t{T:.7e}\n")
    return "/T_P/" + name


def write_eos(path, row, name="eos_planet.dat"):
    """Constant mixing ratios replicated over the whole T,P grid; H2 fills."""
    vmr = {g: 10.0 ** float(row[f"atm {g}"]) for g in GASES}
    vmr["H2"] = max(1.0 - sum(vmr.values()), 0.0)

    abund = {s: 0.0 for s in EOS_SPECIES}
    for g, v in vmr.items():
        abund[g] = v

    row_vals = "\t".join(f"{abund[s]:.6e}" for s in EOS_SPECIES)
    out = ["T\t\tP\t\t" + "\t\t".join(EOS_SPECIES), ""]
    for p in EOS_P:
        out.append(f"{p:.6e}")
        out.append("")
        for T in EOS_T:
            out.append(f"{float(T):.6e}\t{p:.6e}\t{row_vals}")
        out.append("")
    with open(os.path.join(path, "EOS", name), "w") as f:
        f.write("\n".join(out) + "\n")
    return "/EOS/" + name


def write_userinput(path, row, tp_rel, eos_rel, out_rel="/Spectra/out.dat"):
    r_p = float(row["p_radius"]) * R_EARTH_M
    g = G_SI * float(row["p_mass"]) * M_EARTH_KG / r_p ** 2
    r_s = float(row["s radius"]) * R_SUN_M
    body = [
        "userInput.in - ", "Formatting here is very important.",
        "Exo_Transmit home directory:", path,
        "Temperature-Pressure data file:", tp_rel,
        "Equation of State file:", eos_rel,
        "Output file:", out_rel,
        "Planet surface gravity (in m/s^-2):", f"{g:.6e}",
        "Planet radius (in m):", f"{r_p:.6e}",
        "Star radius (in m):", f"{r_s:.6e}",
        "Pressure of cloud top (in Pa):", "0.0",
        "Rayleigh scattering augmentation factor:", "1.0",
        "End of userInput.in (Do not change this line)",
    ]
    with open(os.path.join(path, "userInput.in"), "w") as f:
        f.write("\n".join(body) + "\n")
    return os.path.join(path, out_rel.lstrip("/"))


def run_planet(path, row):
    """Run Exo-Transmit once; returns (wavelength in microns, depth fraction)."""
    tp = write_tp(path, row)
    eos = write_eos(path, row)
    outfile = write_userinput(path, row, tp, eos)
    if os.path.exists(outfile):
        os.remove(outfile)
    r = subprocess.run(["./Exo_Transmit"], cwd=path, capture_output=True,
                       text=True, timeout=EXO_TIMEOUT_S)
    if not os.path.exists(outfile):
        raise RuntimeError(f"no output (rc={r.returncode}).\n"
                           f"stdout:{r.stdout[-800:]}\nstderr:{r.stderr[-800:]}")
    d = np.loadtxt(outfile, skiprows=2)
    wl_um = d[:, 0] * 1e6          # metres -> microns
    depth = d[:, 1] / 100.0        # percent -> fraction
    return wl_um, depth


# --- Binning -----------------------------------------------------------------

def bin_to_grid(wl, y, grid):
    """Average y within bins whose edges are geometric midpoints of grid;
    empty bins are linearly interpolated from their filled neighbours."""
    order = np.argsort(wl)
    wl, y = np.asarray(wl)[order], np.asarray(y)[order]
    lg = np.log(grid)
    edges = np.empty(len(grid) + 1)
    edges[1:-1] = np.exp(0.5 * (lg[:-1] + lg[1:]))
    edges[0] = np.exp(lg[0] - 0.5 * (lg[1] - lg[0]))
    edges[-1] = np.exp(lg[-1] + 0.5 * (lg[-1] - lg[-2]))
    idx = np.digitize(wl, edges) - 1
    inside = (idx >= 0) & (idx < len(grid))
    sums = np.bincount(idx[inside], weights=y[inside], minlength=len(grid))
    counts = np.bincount(idx[inside], minlength=len(grid))
    out = np.full(len(grid), np.nan)
    filled = counts > 0
    out[filled] = sums[filled] / counts[filled]
    if not filled.all():
        out = np.interp(grid, grid[filled], out[filled])
    return out


# --- Worker ------------------------------------------------------------------

def _worker_dir():
    return make_workdir(os.path.join(WORK_ROOT, f"w{os.getpid()}"))


def render_one(index, row, grid):
    """One planet in this process's private tree.

    Returns (index, depth on grid or None, seconds, error string or None).
    """
    t0 = time.time()
    try:
        wd = _worker_dir()
        wl, depth = run_planet(wd, row)
        binned = bin_to_grid(wl, depth, grid).astype(np.float32)
        if not np.all(np.isfinite(binned)):
            raise RuntimeError("non-finite depth after binning")
        return index, binned, time.time() - t0, None
    except Exception as e:  # never let one planet kill the run
        return index, None, time.time() - t0, f"{type(e).__name__}: {str(e)[:300]}"


def load_rows(split, limit=None):
    df = pd.read_parquet(os.path.join(DATA, f"{split}_params.parquet"))
    if limit is not None:
        df = df.iloc[:limit]
    fill = df["atm fill_gas"].unique() if "atm fill_gas" in df else ["H2"]
    if set(fill) != {"H2"}:
        raise ValueError(f"{split}: expected fill gas H2 only, got {fill}")
    return df, [df.iloc[i].to_dict() for i in range(len(df))]


def render_split(split, grid, jobs=8, limit=None, verbose=True):
    """Exo-Transmit depth for every planet of split; NaN rows where it failed."""
    from joblib import Parallel, delayed
    df, rows = load_rows(split, limit)
    n = len(rows)
    out = np.full((n, len(grid)), np.nan, dtype=np.float32)
    times = np.full(n, np.nan)
    errors = {}
    t0 = time.time()
    done = 0
    report_every = max(1, n // 20)
    gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(render_one)(i, r, grid) for i, r in enumerate(rows))
    for i, y, dt, err in gen:
        times[i] = dt
        if y is None:
            errors[i] = err
        else:
            out[i] = y
        done += 1
        if verbose and (done % report_every == 0 or done == n):
            el = time.time() - t0
            print(f"  {split}: {done}/{n} done, {el:.0f} s elapsed, "
                  f"{el / done:.2f} s/planet effective, "
                  f"eta {el / done * (n - done):.0f} s", flush=True)
    wall = time.time() - t0
    return df, out, times, errors, wall


def timing_summary(split, n, times, errors, wall, jobs):
    ok = np.isfinite(times)
    print(f"{split}: {n} planets, {len(errors)} Exo-Transmit failures (NaN rows), "
          f"jobs={jobs}")
    print(f"  per-planet wall clock inside a worker: "
          f"median {np.nanmedian(times):.1f} s, mean {np.nanmean(times):.1f} s, "
          f"min {np.nanmin(times):.1f} s, max {np.nanmax(times):.1f} s")
    print(f"  end-to-end: {wall:.0f} s total = {wall / n:.2f} s/planet effective "
          f"({ok.sum()} timed)")
    full = 2000 * wall / n
    print(f"  projected for a 2,000-planet split at jobs={jobs}: "
          f"{full / 60:.0f} min; five splits {5 * full / 3600:.1f} h")
    for i, e in list(errors.items())[:5]:
        print(f"  failure row {i}: {e.splitlines()[0]}")


def wait_for_splits(splits, max_wait_s=1800, poll_s=60):
    """Block until {split}_params.parquet and {split}_native.npy exist."""
    t0 = time.time()
    while True:
        missing = [s for s in splits
                   if not (os.path.exists(os.path.join(DATA, f"{s}_params.parquet"))
                           and os.path.exists(os.path.join(DATA, f"{s}_native.npy")))]
        if not missing:
            return
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"gave up waiting for {missing} after {max_wait_s} s")
        print(f"waiting for {missing} ({(time.time() - t0) / 60:.0f} min so far)",
              flush=True)
        time.sleep(poll_s)


# --- Modes -------------------------------------------------------------------

K_B, M_H = 1.380649e-23, 1.66053907e-27
MOLAR_MASS = {"H2": 2.01588, "H2O": 18.01528, "CO": 28.0101, "CO2": 44.0095,
              "NH3": 17.03052, "CH4": 16.04246, "O3": 47.99820}


def thickness_over_radius(row):
    """Isothermal atmospheric extent H ln(P_base/P_top) in planet radii.

    Exo-Transmit holds gravity constant through the atmosphere, TauREx lets it
    fall with altitude, so the depth deficit grows with this number (see the
    validation record in ../exotransmit_harness.py, which reached 0.147).
    """
    vmr = {g: 10.0 ** float(row[f"atm {g}"]) for g in GASES}
    vmr["H2"] = max(1.0 - sum(vmr.values()), 0.0)
    mu = sum(MOLAR_MASS[g] * v for g, v in vmr.items())
    r_p = float(row["p_radius"]) * R_EARTH_M
    g = G_SI * float(row["p_mass"]) * M_EARTH_KG / r_p ** 2
    H = K_B * float(row["atm temperature"]) / (mu * M_H * g)
    return H * np.log(float(row["atm base_pressure"]) / float(row["atm top_pressure"])) / r_p


def coarse_corr(a, b, factor=8):
    """Pearson correlation after averaging `factor` adjacent native points.

    At R ~ 1000 the two codes' grids do not line up point for point, so the
    native-resolution correlation is depressed on planets whose spectra are
    nearly flat; the coarsened figure is the fairer agreement measure."""
    n = len(a) // factor * factor
    return np.corrcoef(a[:n].reshape(-1, factor).mean(1),
                       b[:n].reshape(-1, factor).mean(1))[0, 1]


def validate(grid, jobs, n=20, split="test1"):
    """Compare Exo-Transmit with the MultiREx native depth for the first n planets."""
    df, exo, times, errors, wall = render_split(split, grid, jobs=jobs, limit=n)
    native = np.load(os.path.join(DATA, f"{split}_native.npy"), mmap_mode="r")[:n]
    assert native.shape[1] == len(grid) == exo.shape[1]
    print(f"\nvalidation: {split} first {n} planets, Exo-Transmit vs MultiREx native")
    print(f"{'row':>4} {'bio':>3} {'R_p':>6} {'M_p':>6} {'T':>5} {'thick/R':>8} "
          f"{'depth MultiREx':>15} {'depth ExoT':>13} {'ratio':>8} {'corr':>8} "
          f"{'corr x8':>8} {'contrast':>9} {'s':>5}")
    ratios, cors, cors8, amps = [], [], [], []
    for i in range(n):
        a, b = np.asarray(native[i], dtype=float), exo[i].astype(float)
        row = df.iloc[i]
        head = (f"{i:>4} {int(row['biosignature']):>3} {row['p_radius']:>6.2f} "
                f"{row['p_mass']:>6.1f} {row['atm temperature']:>5.0f} "
                f"{thickness_over_radius(row):>8.3f} {a.mean():>15.4e}")
        if i in errors:
            print(f"{head} {'FAILED':>13}")
            continue
        ratio = b.mean() / a.mean()
        cor, cor8 = np.corrcoef(a, b)[0, 1], coarse_corr(a, b)
        amp = (b.std() / b.mean()) / (a.std() / a.mean())
        ratios.append(ratio); cors.append(cor); cors8.append(cor8); amps.append(amp)
        print(f"{head} {b.mean():>13.4e} {ratio:>8.4f} {cor:>8.4f} {cor8:>8.4f} "
              f"{amp:>9.3f} {times[i]:>5.1f}")
    print(f"\nmedian depth ratio {np.median(ratios):.4f}   "
          f"median correlation {np.median(cors):.4f} (native), "
          f"{np.median(cors8):.4f} (8 native points averaged)   "
          f"median contrast ratio {np.median(amps):.3f}   "
          f"({len(ratios)} of {n} compared)")
    print("depth ratios below ~0.98 are expected for thick/R above ~0.2: that is "
          "the constant-gravity difference, not a unit error")
    if not (0.97 < np.median(ratios) < 1.03 and np.median(cors8) > 0.99):
        print("WARNING: outside the expected range (median ratio ~1.00, coarse "
              "corr > 0.99); check units before running the full job")
    timing_summary(split, n, times, errors, wall, jobs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N planets of each split (testing)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--validate", action="store_true",
                    help="compare the first 20 planets of test1 with MultiREx; "
                         "writes nothing")
    ap.add_argument("--no-wait", action="store_true",
                    help="fail immediately if a split's files are missing")
    a = ap.parse_args(argv)

    if not os.path.exists(os.path.join(SRC, "Exo_Transmit")):
        raise SystemExit(f"Exo-Transmit binary not found at {SRC}")
    os.makedirs(WORK_ROOT, exist_ok=True)
    grid = np.load(os.path.join(DATA, "native_wl.npy"))

    if a.validate:
        wait_for_splits(["test1"], max_wait_s=0 if a.no_wait else 1800)
        validate(grid, a.jobs)
        return

    wait_for_splits(a.splits, max_wait_s=0 if a.no_wait else 1800)
    for split in a.splits:
        print(f"--- {split} ---", flush=True)
        df, out, times, errors, wall = render_split(split, grid, jobs=a.jobs,
                                                     limit=a.limit)
        timing_summary(split, len(df), times, errors, wall, a.jobs)
        if a.limit is None:
            path = os.path.join(DATA, f"{split}_native_exotransmit.npy")
            np.save(path, out)
            print(f"  wrote {path}  shape {out.shape}  "
                  f"NaN rows {int(np.isnan(out).any(axis=1).sum())}", flush=True)
        else:
            print(f"  --limit given: nothing written", flush=True)


if __name__ == "__main__":
    main()
