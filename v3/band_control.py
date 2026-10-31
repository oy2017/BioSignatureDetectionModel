"""What the verdict reads (REVISION_EXPECTATIONS.md, N2): C+B-mlp refit with one molecule's bands removed from the input, at JWST precision. Writes results/round3/band_control_real.csv and band_control_votes_<cond>.csv.gz."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import SEED  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402
import round3_learner as R  # noqa: E402

BANDS = {"CH4": [(2.15, 2.45), (3.1, 3.7)], "H2O": [(1.3, 1.5), (1.75, 2.05), (2.5, 3.0), (5.5, 7.0)], "CO2": [(4.2, 4.45)], "NH3": [(2.9, 3.1), (5.8, 6.5)]}; CONDS = ["rainout", "local", "crossfield"]


def run(j, i, mol, X, P, grids, cen):
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; keep = meas & ~np.any([(cen >= a) & (cen <= b) for a, b in BANDS[mol]], axis=0); e = np.nan_to_num(err[keep]); noise = float(np.sqrt(np.mean(e ** 2))); decide, t = R.fit_mlp(X, P, keep, e, md, noise, SEED + 8000 + 10 * i); real, votes = [], []
    if labs.get(mol) is not None:
        pr, ab, p = decide(ML.features(x[None, :], keep, noise, True))[mol]; real.append(dict(batch=batch, planet=pl, molecule=mol, label=labs[mol], present=pr[0], absent=ab[0], prob=p[0], credit=float((pr if labs[mol] == "present" else ab)[0]), bins_removed=int(meas.sum() - keep.sum())))
    for c, G in grids.items():
        if pl not in G: continue
        r2 = np.random.default_rng(SEED + 1700 + i); Te = G[pl].copy(); Te[:, meas] += r2.normal(0, 1, (len(Te), meas.sum())) * np.nan_to_num(err[meas]); pr, ab, p = decide(ML.features(Te, keep, noise, True))[mol]; votes.append(pd.DataFrame(dict(cond=c, batch=batch, planet=pl, precision="JWST", molecule=mol, design="bands removed", row=np.arange(len(pr)), present=pr, absent=ab)))
    print(pl, mol, flush=True); return real, votes


def main():
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8; X, P = R.population("C+B-mlp"); grids = {c: dict(np.load(os.path.join(R.GRIDS, f"grid_{c}.npz"))) for c in CONDS}; L = ML.jobs_list(); n = len(L); L = L + R.independent(); e, _ = J.tier3_layout(); cen = 0.5 * (e[1:] + e[:-1])
    Q = Parallel(n_jobs=jobs, verbose=5)(delayed(run)(j, i, m, X, P, grids if i < n else {}, cen) for i, j in enumerate(L) for m in BANDS if i < n or j[2].get(m) is not None); pd.DataFrame([x for q in Q for x in q[0]]).to_csv(os.path.join(R.OUT, "band_control_real.csv"), index=False); V = pd.concat([v for q in Q for v in q[1]])
    for c, g in V.groupby("cond"): g.drop(columns="cond").to_csv(os.path.join(R.OUT, f"band_control_votes_{c}.csv.gz"), index=False)


if __name__ == "__main__":
    main()
