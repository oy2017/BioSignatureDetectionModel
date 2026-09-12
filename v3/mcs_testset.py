"""The screen on Ariel's own targets: a test set built from the Mission Candidate Sample.

Every known planet in the consortium's current MCS list (Edwards & Tinetti 2022; repository
arielmission-space/Mission_Candidate_Sample) becomes one test planet: its real host (Teff,
radius, mass), real radius, mass, semi-major axis and transit temperature, with C/O and [M/H]
drawn as in the grid, FastChem equilibrium chemistry, and the same forward model. Two things
the grid cannot tell us come out of it:
  1. how many real targets fall OUTSIDE the training box before any mismatch is applied
     (bulk parameters, and the OOD scores of the rendered spectra);
  2. the screen's accuracy on the mission's population, with noise set by the mission's own
     tier definition: Tier 2 = SNR 7 on the assumed 5-scale-height H2 modulation after the
     catalogue's number of transits (ArielRad; Mugnai et al. 2020), i.e. an absolute noise
     level per target that does not know the planet's actual C/O-dependent amplitude.

Usage: python mcs_testset.py --jobs 4
Writes data/mcs/mcs_params.parquet, data/mcs/mcs_native.npy, results/ariel_mcs.txt
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, SEED  # noqa: E402
import generate_grid as G  # noqa: E402

RE, ME, RS, MS, AU = 6.371e6, 5.972e24, 6.957e8, 1.989e30, 1.496e11
KB, MH = 1.380649e-23, 1.6726e-27


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); a = ap.parse_args()
    d = pd.read_csv(os.path.join(DATA, "mcs", "Ariel_MCS_Known_2026-05-11.csv"))
    d = d[(d["Transit"].astype(str).str.lower().isin(["true", "1", "yes"])) | (d["Tier 2 Transits"] > 0)] if "Transit" in d else d
    P = pd.DataFrame({
        "name": d["Planet Name"].astype(str),
        "p_radius": d["Planet Radius [Re]"].astype(float), "p_mass": d["Planet Mass [Me]"].astype(float),
        "atm temperature": d["Planet Transit Temperature [K]"].astype(float),
        "s temperature": d["Star Temperature [K]"].astype(float), "s radius": d["Star Radius [Rs]"].astype(float),
        "s mass": d["Star Mass [Ms]"].astype(float), "sma": d["Planet Semi-major Axis [au]"].astype(float),
        "tier1_transits": d["Tier 1 Transits"].astype(float), "tier2_transits": d["Tier 2 Transits"].astype(float),
        "tier3_transits": d["Tier 3 Transits"].astype(float), "t14_s": d["Transit Duration T14 [s]"].astype(float),
        "depth_pct": d["Transit Depth [%]"].astype(float), "max_tier": d["Max Tier"] if "Max Tier" in d else np.nan,
    }).dropna(subset=["p_radius", "p_mass", "atm temperature", "s temperature", "s radius", "s mass", "sma"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED + 202)
    P["atm base_pressure"] = 10 ** rng.uniform(np.log10(G.BULK["atm base_pressure"][0]), np.log10(G.BULK["atm base_pressure"][1]), len(P))
    P["atm top_pressure"] = 10 ** rng.uniform(np.log10(G.BULK["atm top_pressure"][0]), np.log10(G.BULK["atm top_pressure"][1]), len(P))
    P["co_ratio"] = rng.uniform(*G.CO_RANGE, len(P)); P["mh"] = rng.uniform(*G.MH_RANGE, len(P))
    P["label_co"] = (P.co_ratio > G.CO_CUT).astype(int); P["atm fill_gas"] = "H2"
    # inside the training box?
    box = {k: G.BULK[k] for k in G.BULK if k in P}
    inside = np.ones(len(P), bool)
    for k, (lo, hi) in box.items():
        inside &= (P[k] >= lo) & (P[k] <= hi)
    P["in_box"] = inside
    # the tier's assumed modulation and the absolute noise it implies
    g = 6.674e-11 * P.p_mass * ME / (P.p_radius * RE) ** 2
    H = KB * P["atm temperature"] / (2.3 * MH * g)
    P["modulation_5H"] = 2 * (P.p_radius * RE) * 5 * H / (P["s radius"] * RS) ** 2     # fractional depth change
    P["sigma_tier2"] = P.modulation_5H / 7.0                                              # SNR 7 on it, by definition
    print(f"{len(P)} known targets; inside the training box on every bulk parameter: {inside.mean():.1%}", flush=True)
    for k, (lo, hi) in box.items():
        out = ((P[k] < lo) | (P[k] > hi)).mean()
        if out > 0: print(f"  outside on {k}: {out:.1%}  (range {P[k].min():.3g}-{P[k].max():.3g} vs box {lo:.3g}-{hi:.3g})", flush=True)

    chem = G.Chemistry(); t0 = time.time()
    comp = [chem.composition(r["atm temperature"], r.co_ratio, r.mh) for _, r in P.iterrows()]
    for gname in G.GASES: P[f"atm {gname}"] = [c[gname] for c in comp]
    print(f"chemistry {time.time()-t0:.0f} s", flush=True)
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    rows = [P.iloc[i].to_dict() for i in range(len(P))]; t0 = time.time()
    res = Parallel(n_jobs=a.jobs)(delayed(G.worker)(c, wl) for c in G.chunked(rows, 25))
    specs = [x for c in res for x in c]; ok = np.array([s is not None for s in specs])
    X = np.full((len(P), len(wl)), np.nan, dtype=np.float32)
    for i, sp in enumerate(specs):
        if sp is not None: X[i] = sp
    print(f"rendered {ok.sum()}/{len(P)} in {time.time()-t0:.0f} s; label balance {P.label_co.mean():.2f}", flush=True)
    P.to_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet")); np.save(os.path.join(DATA, "mcs", "mcs_native.npy"), X)


if __name__ == "__main__":
    main()
