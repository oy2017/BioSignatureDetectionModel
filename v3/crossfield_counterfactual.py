"""Does the verdict read the molecule's own absorption (REVISION_EXPECTATIONS.md, N3)? The frozen C+B-mlp, refit with its seeds, reads each photochemical model with one molecule's opacity removed. Writes results/round3/counterfactual.csv.gz and counterfactual.txt."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import SEED  # noqa: E402
import crossfield_test as CT  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402
import round3_learner as R  # noqa: E402

COLS = {"H2O": "rprs2_no_H2O_HITEMP", "CH4": "rprs2_no_CH4", "CO2": "rprs2_no_CO2", "NH3": "rprs2_no_NH3"}


def spectra(j, Y, M, edges):
    batch, pl, *_ = j; m = M.loc[ML.BASE(pl)]; rp = m.p_radius * CT.RE; rs = float(m["s radius"]) * CT.RSUN; tp = float(m["atm temperature"]); gp = 6.674e-8 * m.p_mass * 5.972e27 / rp ** 2; q = Y[Y.planet == pl].sort_values("row"); out = {g: [] for g in COLS}
    for _, r in q.iterrows():
        A = pd.read_csv(os.path.join(CT.ROOT, r.file, "mtgrid_helios_transmission.csv"), usecols=["wave"] + list(COLS.values())); h = (tp / r["T"]) * (CT.G0 / gp)
        for g, c in COLS.items(): out[g].append(CT.ML.bin_native((((np.sqrt(A[c].to_numpy()) * CT.RS0 - CT.RP0) * h + rp) / rs)[None, :] ** 2, A.wave.to_numpy(), edges)[0])
    return {g: np.array(v) for g, v in out.items()}


def run(j, i, prec, X, P, Xg, C):
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; e = np.nan_to_num(err[meas]); e = np.sqrt(e ** 2 + s[meas] ** 2) if prec == "Tier-1" else e; noise = float(np.sqrt(np.mean(e ** 2))); seed = SEED + 5000 + 10 * i + (0 if prec == "JWST" else 3); decide, _ = R.fit_mlp(X, P, meas, e, md, noise, seed); out = []
    r = np.random.default_rng(SEED + 1700 + i + (0 if prec == "JWST" else 5000)); nz = r.normal(0, 1, (len(Xg), meas.sum())) * e; Te = Xg.copy(); Te[:, meas] += nz; full = decide(ML.features(Te, meas, noise, True))
    for g in COLS:
        Tc = C[g].copy(); Tc[:, meas] += nz; d = decide(ML.features(Tc, meas, noise, True))[g][0]; out.append(pd.DataFrame(dict(batch=batch, planet=pl, precision=prec, molecule=g, row=np.arange(len(d)), full=full[g][0], removed=d)))
    print(pl, prec, flush=True); return pd.concat(out)


def main():
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8; X, P = R.population("C+B-mlp"); L = ML.jobs_list(); e, _ = J.tier3_layout(); M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name"); Y = pd.read_csv(os.path.join(CT.OUT, "truth_crossfield.csv")); G = np.load(os.path.join(CT.OUT, "grid_crossfield.npz"))
    C = Parallel(n_jobs=jobs)(delayed(spectra)(j, Y, M, e) for j in L); print("spectra", flush=True); Q = pd.concat(Parallel(n_jobs=jobs)(delayed(run)(j, i, p, X, P, G[j[1]], C[i]) for i, j in enumerate(L) for p in ("JWST", "Tier-1"))); Q.to_csv(os.path.join(R.OUT, "counterfactual.csv.gz"), index=False)
    T = Y.melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in COLS], var_name="molecule", value_name="v"); T["molecule"] = T.molecule.str[2:]; A = Q.merge(T, on=["planet", "row", "molecule"]); A["y"] = A.v > -4; rng = np.random.default_rng(405); us = sorted(A.planet.map(ML.BASE).unique()); A["u"] = A.planet.map(ML.BASE); L2 = []
    for p in ("JWST", "Tier-1"):
        for m in ["all"] + list(COLS):
            g = A[(A.precision == p) & A.y & ((A.molecule == m) | (m == "all"))]; b = g.groupby("u")[["full", "removed"]].agg(["sum", "count"]); ix = rng.integers(0, len(b), (4000, len(b))); f, c, n = b[("full", "sum")].to_numpy(), b[("removed", "sum")].to_numpy(), b[("full", "count")].to_numpy(); d = (f[ix].sum(1) - c[ix].sum(1)) / n[ix].sum(1)
            L2.append(f"{p:<7}{m:<4} truly present: {len(g):>6}  called present, full {g.full.mean():.3f}, opacity removed {g.removed.mean():.3f}, own absorption {g.full.mean() - g.removed.mean():+.3f} ({np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}); flips to absent {(g.full.gt(0) & g.removed.eq(0)).sum() / max(g.full.sum(), 1):.3f} of those called present")
    t = "\n".join(L2); open(os.path.join(R.OUT, "counterfactual.txt"), "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
