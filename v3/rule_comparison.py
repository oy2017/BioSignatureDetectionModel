"""Recognising unreliable verdicts against the existing rules, and the routing method on opacity-removed spectra (REVISION_EXPECTATIONS.md, section T). Writes results/round3/rule_comparison.txt and routing_counterfactual.txt."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import SEED  # noqa: E402
import crossfield_counterfactual as CC  # noqa: E402
import crossfield_test as CT  # noqa: E402
import goyal_test as GT  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402
import panel_tier1 as PT  # noqa: E402

MOLS = ML.MOLS; RULES = ["margin", "ensemble", "knn", "mahalanobis", "pca_recon"]; KEEP = 0.7; OUT = os.path.join("results", "round3")


def noisy(j, i, X, prec):
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; e = np.nan_to_num(err); r = np.random.default_rng(SEED + 1700 + i + (0 if prec == "JWST" else 5000)); B = X.copy(); B[:, filled] = B[:, meas].mean(1, keepdims=True); ee = np.zeros(X.shape); ee[:, meas] = r.normal(0, 1, (len(X), meas.sum())) * (np.sqrt(e[meas] ** 2 + s[meas] ** 2) if prec == "Tier-1" else e[meas])
    if prec == "Tier-1": ee[:, filled] = r.normal(0, 1, (len(X), filled.sum())) * s[filled]
    return B + ee


def rules(ctx, L, Z, cond):
    rows = []
    for i, j in enumerate(L):
        pl = j[1]
        if pl not in Z: continue
        for prec in ("JWST", "Tier-1"):
            B = noisy(j, i, Z[pl], prec); P = ctx.sc.probs(B); S, _, _ = ctx.sc.scores(B)
            for m in MOLS:
                v = np.sum([P["all"][(m, k)] >= 0.5 for k in PT.LEARNERS], axis=0); d = dict(cond=cond, planet=pl, precision=prec, molecule=m, row=np.arange(len(B)), present=(v >= 3).astype(float), absent=(v <= 1).astype(float))
                for r in RULES: d[r] = np.mean([S[(r, (m, k))] for k in PT.LEARNERS], axis=0) if r in ("margin", "ensemble") else S[(r, None)]
                rows.append(pd.DataFrame(d))
    return pd.concat(rows)


def balanced(g):
    c = np.where(g.y, g.present, g.absent) + 0.5 * (1 - g.present - g.absent); return 0.5 * (c[g.y].mean() + c[~g.y].mean())


def boot(a, b, rng):
    us = sorted(set(a.u)); ga, gb = {u: g for u, g in a.groupby("u")}, {u: g for u, g in b.groupby("u")}; o = []
    for _ in range(400):
        pick = rng.choice(us, len(us)); x = pd.concat([ga[u] for u in pick if u in ga]); y = pd.concat([gb[u] for u in pick if u in gb]); o.append(balanced(x) - balanced(y))
    return np.percentile(o, [2.5, 97.5])


def main():
    ctx = PT.Context(); L = ML.jobs_list(); Lines = []; rng = np.random.default_rng(405); Th = pd.read_csv(os.path.join(OUT, "thresholds_C+B-mlp.csv")).melt(id_vars=["planet", "precision"], var_name="molecule", value_name="t")
    for cond in ("local", "crossfield"):
        Z = np.load(os.path.join("results", "goyal_test", f"grid_{cond}.npz")); A = rules(ctx, L, Z, cond); Y = pd.read_csv(os.path.join("results", "goyal_test", f"truth_{cond}.csv")).melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in MOLS], var_name="molecule", value_name="v"); Y["molecule"] = Y.molecule.str[2:]; A = A.merge(Y, on=["planet", "row", "molecule"]); A["y"] = A.v > -4; A["u"] = A.planet.map(ML.BASE)
        E = pd.read_csv(os.path.join(OUT, f"votes_C+B-mlp_{cond}.csv.gz")).merge(Y, on=["planet", "row", "molecule"]).merge(Th, on=["planet", "precision", "molecule"]); E["y"] = E.v > -4; E["u"] = E.planet.map(ML.BASE); E["conf"] = np.abs(E.prob - E.t)
        for prec in ("JWST", "Tier-1"):
            for mol in ["all"] + MOLS:
                a = A[(A.precision == prec) & ((A.molecule == mol) | (mol == "all"))]; e = E[(E.precision == prec) & ((E.molecule == mol) | (mol == "all"))]; kept_e = e[e.conf >= e.conf.quantile(1 - KEEP)]; s = [f"{cond:<11}{prec:<7}{mol:<4} ensemble all {balanced(e):.3f}, kept {balanced(kept_e):.3f}; classifier all {balanced(a):.3f}"]; best, bn = -1, None; K = {}
                for r in RULES:
                    k = a[a[r] <= a[r].quantile(KEEP)]; K[r] = k; b = balanced(k); s.append(f"{r} {b:.3f}")
                    if b > best: best, bn = b, r
                if mol == "all":
                    lo, hi = boot(kept_e, K[bn], rng); s.append(f"ensemble kept minus best rule ({bn}) {balanced(kept_e) - best:+.3f} ({lo:+.3f},{hi:+.3f})")
                Lines.append("  ".join(s)); print(Lines[-1], flush=True)
    open(os.path.join(OUT, "rule_comparison.txt"), "w").write("\n".join(Lines) + "\n")
    Yc = pd.read_csv(os.path.join("results", "goyal_test", "truth_crossfield.csv")); M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name"); e, _ = J.tier3_layout(); F = pd.read_csv("results/jwst_panel/amplitude_flag.csv").set_index("planet"); Fl = pd.read_csv("results/jwst_panel_fourth/fourth_batch_flags.csv").set_index("planet"); rows = []
    Zc = np.load(os.path.join("results", "goyal_test", "grid_crossfield.npz"))
    for i, j in enumerate(L):
        C = CC.spectra(j, Yc, M, e); full = GT.run_frozen(ctx, j, i, Zc[j[1]], F, Fl); full = full[full.design == "the method"]
        for g in MOLS:
            q = GT.run_frozen(ctx, j, i, C[g], F, Fl); q = q[(q.design == "the method") & (q.molecule == g)].rename(columns=dict(present="removed")); f = full[full.molecule == g].rename(columns=dict(present="full")); rows.append(f[["planet", "precision", "molecule", "row", "full"]].merge(q[["precision", "row", "removed"]], on=["precision", "row"]))
    Q = pd.concat(rows); T = Yc.melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in MOLS], var_name="molecule", value_name="v"); T["molecule"] = T.molecule.str[2:]; Q = Q.merge(T, on=["planet", "row", "molecule"]); Q = Q[Q.v > -4]; L2 = []
    for p in ("JWST", "Tier-1"):
        for m in ["all"] + MOLS:
            g = Q[(Q.precision == p) & ((Q.molecule == m) | (m == "all"))]; L2.append(f"{p:<7}{m:<4} routing method: called present, full {g.full.mean():.3f}, opacity removed {g.removed.mean():.3f}, flips {(g.full.gt(0) & g.removed.eq(0)).sum() / max(g.full.sum(), 1):.3f}")
    t = "\n".join(L2); open(os.path.join(OUT, "routing_counterfactual.txt"), "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
