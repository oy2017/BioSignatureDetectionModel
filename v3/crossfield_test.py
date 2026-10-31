"""Confirmation on the photochemical Crossfield et al. (2025) grid (Zenodo 17101615; REVISION_EXPECTATIONS.md, section M): 936 VULCAN models, extracted under data/crossfield2025/x/. No clouds or hazes; all share one planet, rescaled to each target by scale height.

Usage: python crossfield_test.py --build, then --score (once, after the design is committed). Writes results/goyal_test/{grid_crossfield.npz, truth_crossfield.csv, votes_frozen_crossfield.csv.gz, votes_matched_crossfield.csv.gz}.
"""
import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import goyal_test as GT  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402

ROOT = os.path.join("data", "crossfield2025", "x"); OUT = GT.OUT; N = 200; KB, MH, RSUN, RE = 1.380658e-16, 1.6726e-24, 6.957e10, 6.371e8; RP0, G0, RS0 = 0.376 * 7.1492e9, 602.6, 0.5 * 6.957e10; COL = {"H2O": "H2O_HITEMP", "CH4": "CH4", "CO2": "CO2", "NH3": "NH3"}
PAT = re.compile(r"Kzz([\d.]+)_Tint(\d+)_TP_helios_(\d+)K_SNCHO_([\d.]+)x_z83_C([\d.]+)_N([\d.]+)_O([\d.]+)_S([\d.]+)")


def table():
    t = sorted(d for d in os.listdir(ROOT) if d.startswith("MTgrid") and os.path.exists(os.path.join(ROOT, d, "mtgrid_helios_transmission.csv"))); r = [PAT.search(d).groups() for d in t]
    return pd.DataFrame(dict(file=t, T=[float(x[2]) for x in r], kzz=[float(x[0]) for x in r], tint=[float(x[1]) for x in r], met=[float(x[3]) for x in r], co=[float(x[4]) / float(x[6]) * 0.55 for x in r], cmet=[float(x[4]) for x in r], omet=[float(x[6]) for x in r]))


def load(d):
    A = pd.read_csv(os.path.join(ROOT, d, "mtgrid_helios_transmission.csv"), usecols=["wave", "rprs2"]); V = pd.read_csv(os.path.join(ROOT, d, "mtgrid_helios_vertical_profiles.csv")); k = (V.p >= 1e-4) & (V.p <= 0.1)
    return A.wave.to_numpy(), A.rprs2.to_numpy(), {m: float(np.log10(np.clip(V.loc[k, c].mean(), 1e-30, None))) for m, c in COL.items()}, float(V.loc[k, "mmw"].mean())


def grid_for(j, i, T, M, edges):
    batch, pl, labs, x, err, filled, s, md = j; m = M.loc[ML.BASE(pl)]; rp = m.p_radius * RE; rs = float(m["s radius"]) * RSUN; tp = float(m["atm temperature"]); gp = 6.674e-8 * m.p_mass * 5.972e27 / rp ** 2; q = T[np.abs(T["T"] - tp) <= 200]
    if len(q) == 0: q = T[T["T"] == T["T"].iloc[np.argmin(np.abs(T["T"] - tp))]]
    q = q.sample(min(N, len(q)), random_state=SEED + 950 + i); X, Y = [], []
    for _, r in q.iterrows():
        w, d, c, mmw = load(r.file); h1 = KB * r["T"] / (mmw * MH * G0); h2 = KB * tp / (mmw * MH * gp); z = np.sqrt(d) * RS0 - RP0; dd = ((z * h2 / h1 + rp) / rs) ** 2   # the level in z is irrelevant after normalisation, so the small logarithmic term of gg_rescale is dropped
        X.append(ML.bin_native(dd[None, :], w, edges)[0]); Y.append(dict(file=r.file, T=r["T"], g=gp / 100, met=r.met, co=r.co, haze=0.0, cloud=0.0, mmm=mmw, **{f"v_{k}": v for k, v in c.items()}))
    return np.array(X), pd.DataFrame(Y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); ap.add_argument("--score", action="store_true"); ap.add_argument("--jobs", type=int, default=4); a = ap.parse_args(); L = ML.jobs_list(); cache = os.path.join(OUT, "grid_crossfield.npz")
    if a.build:
        edges, _ = J.tier3_layout(); M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name"); T = table(); print(len(T), flush=True); G = Parallel(n_jobs=a.jobs)(delayed(grid_for)(j, i, T, M, edges) for i, j in enumerate(L))
        Y = pd.concat([g[1].assign(planet=j[1], row=np.arange(len(g[1]))) for g, j in zip(G, L)], ignore_index=True); Y.to_csv(os.path.join(OUT, "truth_crossfield.csv"), index=False); np.savez_compressed(cache, **{j[1]: g[0] for g, j in zip(G, L)})
        print(Y.groupby("planet").size().describe().round(0).to_dict(), {m: round(float((Y[f"v_{m}"] > -4).mean()), 3) for m in ML.MOLS}, "nan", {j[1]: int(np.isnan(g[0]).sum()) for g, j in zip(G, L) if np.isnan(g[0]).any()})
    if a.score:
        import panel_tier1 as PT
        Z = np.load(cache); Y = pd.read_csv(os.path.join(OUT, "truth_crossfield.csv")); Xtr, Ptr = ML.training(); ctx = PT.Context(); F = pd.read_csv(os.path.join(RESULTS, "jwst_panel", "amplitude_flag.csv")).set_index("planet"); Fl = pd.read_csv(os.path.join(RESULTS, "jwst_panel_fourth", "fourth_batch_flags.csv")).set_index("planet")
        pd.concat([GT.run_frozen(ctx, j, i, Z[j[1]], F, Fl) for i, j in enumerate(L)]).to_csv(os.path.join(OUT, "votes_frozen_crossfield.csv.gz"), index=False); del ctx
        pd.concat(Parallel(n_jobs=a.jobs)(delayed(GT.run_matched)(j, i, Z[j[1]], Y[Y.planet == j[1]], Xtr, Ptr) for i, j in enumerate(L))).to_csv(os.path.join(OUT, "votes_matched_crossfield.csv.gz"), index=False)


if __name__ == "__main__":
    main()
