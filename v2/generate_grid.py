"""Regenerate the synthetic grid with independently sampled parameters.

Differences from the old generate_multiverse_data.py, all deliberate:
  * Every parameter is drawn independently with a NumPy generator and passed to
    MultiREx as a scalar. The old grid drew ranges through MultiREx's
    clone_shuffled(), which delivered planet radius, stellar temperature and
    semi-major axis with |r| >= 0.97 (manuscript Section 3.1). Independence of
    the delivered sample is checked and written to results/independence.txt.
  * Spectra are stored NOISE-FREE at TauREx's native resolution (R ~ 1000,
    2,753 points on 0.5-7.8 um). Every observing configuration is produced by
    binning (bin_spectra.py) and every noise realisation by noise.py, so the
    same planet can be observed at any resolution and SNR.
  * Sizes: 20,000 training + 5 x 2,000 test planets before cleaning.

Sampling plan (identical bounds and class profiles to the old Table 1):
  planet radius U(1, 26) R_earth; mass U(1, 300) M_earth; T_atm U(500, 2500) K;
  base pressure U(1e5, 1e6) Pa; top pressure U(1, 10) Pa; T_star U(2500, 7500) K;
  R_star U(0.1, 1.7) R_sun; M_star U(0.1, 1.7) M_sun; sma U(0.01, 0.5) AU;
  H2O logU(-10, -1); CO, CO2, NH3 logU(-9, -3); CH4 and O3 by profile:
    biosignature 50%   CH4 logU(-6, -3), O3 logU(-7, -1)
    nonbio_ch4   1/6   CH4 logU(-6, -3), O3 logU(-10, -7)
    nonbio_o3    1/6   CH4 logU(-9, -6), O3 logU(-7, -1)
    nonbio_none  1/6   CH4 logU(-9, -6), O3 logU(-10, -7)
  label = (log CH4 > -6) and (log O3 > -7)

Cleaning: rows whose forward model fails, returns NaN, or gives any transit
depth > 1 are dropped and counted (the old grid lost ~10%, mostly low-mass
puffy planets; the acceptance rate by mass decile is reported).

Usage:
    python generate_grid.py --n-train 20000 --n-test 2000 --n-sets 5 --jobs 12
    python generate_grid.py --check      # bin native -> 550 grid vs MultiREx's own binning
Outputs (v2/data/): native_wl.npy, {split}_params.parquet, {split}_native.npy
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
WL_MIN, WL_MAX = 0.5, 7.8
FILL_GAS = "H2"
GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]
BIO_CH4, BIO_O3 = -6.0, -7.0

PROFILES = {
    "biosignature": {"CH4": (-6, -3), "O3": (-7, -1)},
    "nonbio_ch4": {"CH4": (-6, -3), "O3": (-10, -7)},
    "nonbio_o3": {"CH4": (-9, -6), "O3": (-7, -1)},
    "nonbio_none": {"CH4": (-9, -6), "O3": (-10, -7)},
}
PROFILE_SHARE = {"biosignature": 0.5, "nonbio_ch4": 1 / 6, "nonbio_o3": 1 / 6,
                 "nonbio_none": 1 / 6}
OTHER_GASES = {"H2O": (-10, -1), "CO": (-9, -3), "CO2": (-9, -3), "NH3": (-9, -3)}
BULK = {
    "p_radius": (1.0, 26.0), "p_mass": (1.0, 300.0),
    "atm temperature": (500.0, 2500.0), "atm base_pressure": (1e5, 1e6),
    "atm top_pressure": (1.0, 10.0),
    "s temperature": (2500.0, 7500.0), "s radius": (0.1, 1.7), "s mass": (0.1, 1.7),
    "sma": (0.01, 0.5),
}


def sample_params(n, rng):
    """n independently drawn planets with the profile shares enforced exactly."""
    counts = {k: int(round(v * n)) for k, v in PROFILE_SHARE.items()}
    counts["nonbio_none"] += n - sum(counts.values())
    rows = []
    for prof, cnt in counts.items():
        for _ in range(cnt):
            r = {k: rng.uniform(*b) for k, b in BULK.items()}
            for g, b in OTHER_GASES.items():
                r[f"atm {g}"] = rng.uniform(*b)
            for g, b in PROFILES[prof].items():
                r[f"atm {g}"] = rng.uniform(*b)
            r["profile"] = prof
            rows.append(r)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    df["atm fill_gas"] = FILL_GAS
    df["biosignature"] = ((df["atm CH4"] > BIO_CH4) & (df["atm O3"] > BIO_O3)).astype(int)
    return df


def native_wavelengths():
    """TauREx's native wavelength grid restricted to [WL_MIN, WL_MAX], ascending."""
    from multirex import Atmosphere, Planet, Star, System
    sysm = build_system(dict(p_radius=8.0, p_mass=50.0, **{"atm temperature": 1000.0,
                        "atm base_pressure": 1e5, "atm top_pressure": 1.0,
                        "s temperature": 5000.0, "s radius": 1.0, "s mass": 1.0, "sma": 0.1},
                        **{f"atm {g}": -5.0 for g in GASES}))
    wn = sysm.transmission.model()[0]
    wl = 1e4 / np.asarray(wn)
    m = (wl >= WL_MIN) & (wl <= WL_MAX)
    return np.sort(wl[m])


def build_system(row):
    from multirex import Atmosphere, Planet, Star, System
    atm = Atmosphere(temperature=float(row["atm temperature"]),
                     base_pressure=float(row["atm base_pressure"]),
                     top_pressure=float(row["atm top_pressure"]),
                     composition={g: float(row[f"atm {g}"]) for g in GASES},
                     fill_gas=FILL_GAS)
    planet = Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]), atmosphere=atm)
    star = Star(temperature=float(row["s temperature"]), radius=float(row["s radius"]),
                mass=float(row["s mass"]))
    system = System(planet=planet, star=star, sma=float(row["sma"]))
    system.make_tm()
    return system


def one_native(row, wl_native):
    """Noise-free native-resolution transit depth on wl_native, or None on failure."""
    try:
        system = build_system(row)
        wn, rprs = system.transmission.model()[:2]
        wl = 1e4 / np.asarray(wn)
        order = np.argsort(wl)
        wl, rprs = wl[order], np.asarray(rprs)[order]
        m = (wl >= WL_MIN) & (wl <= WL_MAX)
        y = rprs[m]
        if y.shape[0] != wl_native.shape[0] or not np.allclose(wl[m], wl_native):
            y = np.interp(wl_native, wl[m], y)
        if not np.all(np.isfinite(y)) or np.any(y > 1.0) or np.any(y <= 0):
            return None
        return y.astype(np.float32)
    except Exception:
        return None


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def worker(rows, wl_native):
    import warnings
    warnings.filterwarnings("ignore")
    return [one_native(r, wl_native) for r in rows]


def generate_split(name, n, seed, jobs, wl_native):
    rng = np.random.default_rng(seed)
    params = sample_params(n, rng)
    rows = [params.iloc[i].to_dict() for i in range(len(params))]
    t0 = time.time()
    chunks = list(chunked(rows, 50))
    out = Parallel(n_jobs=jobs, verbose=0)(delayed(worker)(c, wl_native) for c in chunks)
    specs = [s for c in out for s in c]
    ok = np.array([s is not None for s in specs])
    X = np.vstack([s for s in specs if s is not None])
    kept = params[ok].reset_index(drop=True)
    kept["gen_seed"] = seed
    print(f"{name}: {ok.sum()}/{n} kept ({100*ok.mean():.1f}%), "
          f"positives {kept['biosignature'].mean():.3f}, {time.time()-t0:.0f} s", flush=True)
    # acceptance by mass decile, for the disclosure sentence
    dec = pd.qcut(params["p_mass"], 10, labels=False)
    acc = pd.Series(ok).groupby(dec).mean()
    kept.to_parquet(os.path.join(DATA, f"{name}_params.parquet"))
    np.save(os.path.join(DATA, f"{name}_native.npy"), X)
    return kept, acc


def independence_report(df, path):
    """Two groups, by design, and the report must not blur them.

    Thirteen parameters are drawn independently. CH4 and O3 are NOT: they are drawn
    from ranges chosen by a latent profile (biosignature 50%, the three non-bio
    profiles 1/6 each) so the classes come out balanced. That stratification is the
    whole reason r(CH4, O3) is nonzero, and quoting a single "max |r|" over all
    fifteen hides which group it came from, so the two are reported separately
    along with what the stratification buys and costs.
    """
    INDEP = ["p_radius", "p_mass", "atm temperature", "atm base_pressure",
             "atm top_pressure", "s temperature", "s radius", "s mass", "sma"] + \
            [f"atm {g}" for g in GASES if g not in ("CH4", "O3")]
    LABEL = ["atm CH4", "atm O3"]
    c = df[INDEP + LABEL].corr()

    def worst_within(cols):
        sub = c.loc[cols, cols]
        off = sub.where(~np.eye(len(cols), dtype=bool))
        return off.abs().stack().sort_values(ascending=False).drop_duplicates(), off

    w_ind, off_ind = worst_within(INDEP)
    r_label = float(c.loc["atm CH4", "atm O3"])
    cross = c.loc[INDEP, LABEL].abs().stack().sort_values(ascending=False)

    ch4, o3 = df["atm CH4"].to_numpy(), df["atm O3"].to_numpy()
    pos = float(((ch4 > BIO_CH4) & (o3 > BIO_O3)).mean())
    rng = np.random.default_rng(0)
    indep_pos = float(np.mean([((ch4 > BIO_CH4) & (rng.permutation(o3) > BIO_O3)).mean()
                               for _ in range(400)]))

    with open(path, "w") as f:
        f.write("Pearson correlations between delivered (post-cleaning) parameters\n")
        f.write(f"n = {len(df)}\n\n")
        f.write(f"GROUP 1: the {len(INDEP)} independently drawn parameters.\n")
        f.write("  largest |r| pairs:\n")
        for (a, b), _ in w_ind.head(6).items():
            f.write(f"    {a:<20s} {b:<20s} r = {off_ind.loc[a, b]:+.3f}\n")
        f.write(f"  max |r| = {w_ind.iloc[0]:.3f}\n\n")
        f.write("GROUP 2: the two label gases, which are NOT independently drawn.\n")
        f.write("  They are stratified by profile so the classes balance, which induces\n")
        f.write(f"    r(log CH4, log O3) = {r_label:+.3f}\n")
        f.write(f"  positive rate delivered                    {pos:.4f}\n")
        f.write(f"  positive rate if the pair were independent {indep_pos:.4f}\n")
        f.write("  The stratification also puts a third of all planets in the two\n")
        f.write("  one-gas configurations, which are the confusable negatives.\n\n")
        f.write("BETWEEN the groups (this is what would threaten attributing a\n")
        f.write("sensitivity to one parameter, and it is negligible):\n")
        f.write(f"  max |r| = {cross.iloc[0]:.3f}  ({cross.index[0][0]} / {cross.index[0][1]})\n\n")
        f.write("(old grid: radius / T_star / sma r >= 0.97; mass / R_star / P_base r >= 0.85)\n")
    return w_ind.iloc[0]


def check_binning(wl_native):
    """Native spectrum binned by flux-average onto MultiREx's 550 grid vs
    MultiREx's own generate_spectrum on that grid."""
    from multirex import Physics
    sys.path.insert(0, HERE)
    from bin_spectra import bin_native
    from ariel_bins import configurations
    cfg = configurations()
    rng = np.random.default_rng(0)
    df = sample_params(20, rng)
    worst = 0.0
    n = 0
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        y = one_native(row, wl_native)
        if y is None:
            continue
        system = build_system(row)
        wn550 = Physics.wavenumber_grid(WL_MIN, WL_MAX, 550)
        wb, db = system.generate_spectrum(wn550)
        wlb = 1e4 / np.asarray(wb)
        o = np.argsort(wlb)
        mine = bin_native(y[None, :], wl_native, np.array(cfg["r200"]["edges"]))[0]
        rel = np.abs(mine - np.asarray(db)[o]) / np.asarray(db)[o]
        worst = max(worst, rel.max())
        n += 1
    print(f"binning check on {n} planets: max relative difference vs MultiREx binning = {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-sets", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    wl_native = native_wavelengths()
    np.save(os.path.join(DATA, "native_wl.npy"), wl_native)
    print(f"native grid: {len(wl_native)} points, {wl_native[0]:.3f}-{wl_native[-1]:.3f} um")
    if a.check:
        check_binning(wl_native)
        return
    train, acc = generate_split("train", a.n_train, a.seed, a.jobs, wl_native)
    lines = ["acceptance by mass decile (train):"]
    lines += [f"  decile {i}: {v:.3f}" for i, v in acc.items()]
    for k in range(1, a.n_sets + 1):
        generate_split(f"test{k}", a.n_test, a.seed + k, a.jobs, wl_native)
    r = independence_report(train, os.path.join(RESULTS, "independence.txt"))
    with open(os.path.join(RESULTS, "generation.txt"), "w") as f:
        f.write("\n".join(lines) + f"\nmax |r| between delivered parameters: {r:.3f}\n")
    print(f"max |r| between delivered parameters: {r:.3f}")


if __name__ == "__main__":
    main()
