"""U1 of REVISION_EXPECTATIONS.md: the five existing rules on the matched ensemble itself, against its own confidence, at 70 % coverage on the held-out grids. Writes results/round3/round3_rules.txt."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import SEED  # noqa: E402
import matched_learner as ML  # noqa: E402
import round3_learner as R  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

MOLS = R.MOLS; KEEP = 0.7; CONDS = ("local", "crossfield")


def run(j, i, prec, X, P, grids):
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; e = np.nan_to_num(err[meas]); e = np.sqrt(e ** 2 + s[meas] ** 2) if prec == "Tier-1" else e; noise = float(np.sqrt(np.mean(e ** 2))); seed = SEED + 5000 + 10 * i + (0 if prec == "JWST" else 3); Y = (P[[f"atm {m}" for m in MOLS]].to_numpy() > -4).astype(int); mod = P["modulation_5H"].to_numpy(); hold = np.random.default_rng(SEED + 77).uniform(size=len(X)) < 0.15; nets = []; Zs = []
    for k in range(R.MEMBERS):
        rng = np.random.default_rng(seed + k); Z = np.vstack([R.draw(X[~hold], mod[~hold], meas, e, md, noise, rng) for _ in range(R.COPIES)]); Zs.append(Z[: (~hold).sum()]); nets.append(MLPClassifier((256, 128), batch_size=256, early_stopping=True, n_iter_no_change=6, max_iter=60, random_state=seed + k).fit(Z, np.tile(Y[~hold], (R.COPIES, 1))))
    T = R.thresholds(np.mean([n.predict_proba(R.draw(X[hold], mod[hold], meas, e, md, noise, np.random.default_rng(seed + 99))) for n in nets], axis=0), Y[hold]); Zt = Zs[0][:: 3]; mu = Zt.mean(0); C = np.cov(Zt.T) + 1e-6 * np.eye(Zt.shape[1]); Ci = np.linalg.inv(C); U, S_, Vt = np.linalg.svd(Zt - mu, full_matrices=False); keep = np.searchsorted(np.cumsum(S_ ** 2) / np.sum(S_ ** 2), 0.95) + 1; V = Vt[:keep]; nn = NearestNeighbors(n_neighbors=10).fit(Zt); out = []
    for cond, G in grids.items():
        if pl not in G: continue
        r2 = np.random.default_rng(SEED + 1700 + i + (0 if prec == "JWST" else 5000)); Te = G[pl].copy(); Te[:, meas] += r2.normal(0, 1, (len(Te), meas.sum())) * e; Q = ML.features(Te, meas, noise, True); Pr = np.array([n.predict_proba(Q) for n in nets]); p = Pr.mean(0); d = Q - mu; kn = nn.kneighbors(Q)[0].mean(1); ma = np.einsum("ij,jk,ik->i", d, Ci, d); pc = ((d - d @ V.T @ V) ** 2).sum(1)
        for a, m in enumerate(MOLS): out.append(pd.DataFrame(dict(cond=cond, planet=pl, precision=prec, molecule=m, row=np.arange(len(Q)), present=(p[:, a] >= T[a]).astype(float), own=-np.abs(p[:, a] - T[a]), margin=-np.abs(p[:, a] - 0.5), ensemble=Pr[:, :, a].std(0), knn=kn, mahalanobis=ma, pca_recon=pc)))
    print(pl, prec, flush=True); return pd.concat(out)


def main():
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8; X, P = R.population("C+B-mlp"); grids = {c: dict(np.load(os.path.join(R.GRIDS, f"grid_{c}.npz"))) for c in CONDS}; L = ML.jobs_list(); A = pd.concat(Parallel(n_jobs=jobs)(delayed(run)(j, i, p, X, P, grids) for i, j in enumerate(L) for p in ("JWST", "Tier-1"))); Out = []; rng = np.random.default_rng(405)
    def bal(g): return 0.5 * (g[g.y].present.mean() + 1 - g[~g.y].present.mean())
    for c in CONDS:
        Y = pd.read_csv(os.path.join(R.GRIDS, f"truth_{c}.csv")).melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in MOLS], var_name="molecule", value_name="v"); Y["molecule"] = Y.molecule.str[2:]; a = A[A.cond == c].merge(Y, on=["planet", "row", "molecule"]); a["y"] = a.v > -4; a["u"] = a.planet.map(ML.BASE)
        for p in ("JWST", "Tier-1"):
            for mol in ["all"] + MOLS:
                g = a[(a.precision == p) & ((a.molecule == mol) | (mol == "all"))]; s = [f"{c:<11}{p:<7}{mol:<4} no decline {bal(g):.3f}"]; K = {}
                for r in ("own", "margin", "ensemble", "knn", "mahalanobis", "pca_recon"):
                    K[r] = g[g[r] <= g[r].quantile(KEEP)]; s.append(f"{r} {bal(K[r]):.3f}")
                if mol == "all":
                    us = sorted(set(g.u)); best = max(("margin", "ensemble", "knn", "mahalanobis", "pca_recon"), key=lambda r: bal(K[r])); go = {u: x for u, x in K["own"].groupby("u")}; gb = {u: x for u, x in K[best].groupby("u")}; d = []
                    for _ in range(300):
                        pick = rng.choice(us, len(us)); d.append(bal(pd.concat([go[u] for u in pick if u in go])) - bal(pd.concat([gb[u] for u in pick if u in gb])))
                    s.append(f"own minus best rule ({best}) {bal(K['own']) - bal(K[best]):+.3f} ({np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f})")
                Out.append("  ".join(s)); print(Out[-1], flush=True)
    open(os.path.join(R.OUT, "round3_rules.txt"), "w").write("\n".join(Out) + "\n")


if __name__ == "__main__":
    main()
