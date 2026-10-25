"""E3: does a diagnostic improve Tier-2 target selection against existing methods, at a fixed retrieval budget,
on a mixed population?

Population: POP-I, each planet observed either clean or with haze (3e7 m^-3) with probability q (many random
assignments). Pipeline: the classifier scores every planet; a fraction f of the planets, chosen by a rule that
ranks them "most suspect", is re-scored by a retrieval; 60 % of the planets get a Tier-2 slot, ranked by the
combined present-probability; the metric is CH4-bearing (and H2O, joint) targets lost. The retrieval is the
fallback a pipeline could run on any spectrum: the training forward model plus a haze with free density
(identifiability_calibrate.py --free-haze), taken from its run on the hazed spectra for hazed planets and on the
clean spectra for clean ones. Orderings: the five existing rules (margin, ensemble, Mahalanobis, k-NN, PCA), the
new ones (optical excess, input disagreement, band metric), random, and an oracle that knows which planets are
hazed. Baselines: classifier alone; the existing method, a clear-model retrieval of every planet (which fails on
hazed spectra); the free-haze retrieval of every planet (f = 1); the AIRS-only routing of F5 (no retrieval).

Usage: python e3_budget.py [--fallback freehaze|modelled|clear] [--q 0.1 0.2 0.5] [--draws 20]
Writes results/e3_budget.csv and results/e3_budget.txt.
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402

MOLS = D.MOLS; BUDGET = D.BUDGET
RULES = ["random", "optical", "disagree", "bandmetric", "margin", "ensemble", "mahalanobis", "knn", "pca_recon", "oracle (knows the hazed)"]
FS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
FILES = {"freehaze": ("identifiability_calibration_haze3e7_freehaze.csv", "identifiability_calibration_freehaze.csv"),
         "modelled": ("identifiability_calibration_haze3e7_modelled.csv", "identifiability_calibration_freehaze.csv"),
         "clear": ("identifiability_calibration_haze3e7.csv", "identifiability_calibration.csv")}


def posteriors(Pte, fn):
    """p_present per molecule aligned with Pte (NaN where the file has no posterior, or the retrieval errored)."""
    p = os.path.join(RESULTS, fn)
    if not os.path.exists(p):
        return None, 0
    C = pd.read_csv(p).drop_duplicates("name"); C = C[C["error"].isna()] if "error" in C else C
    idx = {n: i for i, n in enumerate(Pte.name)}; C = C[C.name.isin(idx)]; i = C.name.map(idx).to_numpy()
    out = {}
    for m in MOLS:
        v = np.full(len(Pte), np.nan); v[i] = C[f"{m}_p_present"].to_numpy(float); out[m] = v
    return out, len(C)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fallback", default="freehaze", choices=list(FILES))
    ap.add_argument("--q", type=float, nargs="+", default=[0.1, 0.2, 0.5]); ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    cases, Pte, e, sig = D.test_cases(); sc = D.Scorer(e); n = len(Pte)
    X = {c: cases[c] for c in ("clean", "haze3e7")}; S, P = {}, {}
    for c in X:
        S[c], P[c], _ = sc.scores(X[c]); print("scored", c, flush=True)
    y_true = {m: (Pte[f"atm {m}"].to_numpy() > D.THRESH if hasattr(D, "THRESH") else Pte[f"atm {m}"].to_numpy() > -4).astype(int) for m in MOLS}
    haz_fn, cln_fn = FILES[a.fallback]
    R_h, n_h = posteriors(Pte, haz_fn); R_c, n_c = posteriors(Pte, cln_fn)
    note = ""
    if R_c is None:
        R_c, n_c = posteriors(Pte, "identifiability_calibration.csv"); note = f"(clean-side posteriors from the clean retrieval; {cln_fn} not yet available)"
    R_clear_h, _ = posteriors(Pte, FILES["clear"][0]); R_clear_c, _ = posteriors(Pte, FILES["clear"][1])
    if R_h is None:
        sys.exit(f"no posteriors: {haz_fn}")
    # the population: planets with a fallback posterior on both spectra (so every assignment is scorable)
    have = np.isfinite(R_h["CH4"]) & np.isfinite(R_c["CH4"])
    print(f"fallback {a.fallback}: posteriors on {n_h} hazed and {n_c} clean spectra; population {int(have.sum())} of {n} planets {note}", flush=True)
    idx = np.where(have)[0]; N = len(idx); slots = int(round(BUDGET * N))
    rng = np.random.default_rng(SEED)
    cls = {c: {m: np.mean([P[c]["all"][(m, k)] for k in ("KNN", "MLP", "RFC")], axis=0) for m in MOLS} for c in X}
    airs = {c: {m: np.mean([P[c]["airs_only"][(m, k)] for k in ("KNN", "MLP", "RFC")], axis=0) for m in MOLS} for c in X}
    rule_s = {c: {r: (np.nanmean(np.column_stack([sc.get(S[c], r, m, k) for m in MOLS for k in ("KNN", "MLP", "RFC")]), axis=1)
                      if r not in ("random", "oracle (knows the hazed)") else None) for r in RULES} for c in X}
    rows = []
    for q in a.q:
        for d in range(a.draws):
            h = rng.uniform(0, 1, N) < q                      # which planets are hazed in this draw
            pick = lambda A_c, A_h: np.where(h, A_h[idx], A_c[idx])
            for target in ("CH4", "H2O", "joint"):
                mols = ["CH4", "H2O"] if target == "joint" else [target]
                y = np.clip(sum(y_true[m][idx] for m in mols), 0, 1); ntar = int(y.sum())
                score = sum(pick(cls["clean"][m], cls["haze3e7"][m]) for m in mols)
                fb = sum(pick(airs["clean"][m], airs["haze3e7"][m]) for m in mols)
                ret = sum(pick(R_c[m], R_h[m]) for m in mols)
                ret_clear = sum(pick(R_clear_c[m], R_clear_h[m]) for m in mols) if R_clear_h is not None else None
                strategies = {("classifier alone", 0.0): score, ("AIRS-only routing by optical excess (no retrieval)", 0.0): np.where(pick(rule_s["clean"]["optical"], rule_s["haze3e7"]["optical"]) > np.quantile(rule_s["clean"]["optical"], 0.9), fb, score),
                              (f"{a.fallback} retrieval of every planet", 1.0): ret}
                if ret_clear is not None:
                    strategies[("clear-model retrieval of every planet (existing method)", 1.0)] = np.where(np.isfinite(ret_clear), ret_clear, score)
                for r in RULES:
                    if r == "random": s = rng.uniform(0, 1, N)
                    elif r.startswith("oracle"): s = h + rng.uniform(0, 1e-3, N)
                    else: s = pick(rule_s["clean"][r], rule_s["haze3e7"][r]); s = np.where(np.isfinite(s), s, -np.inf)
                    order = np.argsort(-s)
                    for f in FS[1:-1]:
                        flag = np.zeros(N, bool); flag[order[:int(round(f * N))]] = True
                        strategies[(f"+ {r}", f)] = np.where(flag & np.isfinite(ret), ret, score)
                        if ret_clear is not None and r in ("random", "optical"):
                            strategies[(f"+ {r}, clear-model retrieval", f)] = np.where(flag & np.isfinite(ret_clear), ret_clear, score)
                for (name, f), s_ in strategies.items():
                    sel = np.zeros(N, bool); sel[np.argsort(-s_)[:slots]] = True
                    # classification by the combined pipeline (single-molecule targets): accuracy, and recall on the hazed targets
                    dec = s_ >= 0.5 if target != "joint" else None
                    acc = float((dec == (y == 1)).mean()) if dec is not None else np.nan
                    rec_h = float(dec[h & (y == 1)].mean()) if dec is not None and (h & (y == 1)).any() else np.nan
                    rec_c = float(dec[~h & (y == 1)].mean()) if dec is not None and (~h & (y == 1)).any() else np.nan
                    rows.append(dict(q=q, draw=d, target=target, strategy=name, f=f, targets=ntar, lost=int(ntar - (sel & (y == 1)).sum()),
                                     wasted=int((sel & (y == 0)).sum()), acc=acc, recall_hazed=rec_h, recall_clean=rec_c))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, f"e3_budget_{a.fallback}.csv"), index=False)
    # detection operating points of each rule on the mixed population
    L = [f"E3 fixed-budget selection on a mixed population: {N} POP-I planets, each hazed (3e7 m^-3) with probability q; {a.draws} draws; "
         f"{int(BUDGET*100)} % Tier-2 slots; fallback retrieval: {a.fallback} {note}", ""]
    g = df.groupby(["q", "target", "strategy", "f"]).agg(lost=("lost", "mean"), sd=("lost", "std"), targets=("targets", "mean"), acc=("acc", "mean"), rh=("recall_hazed", "mean"), rc=("recall_clean", "mean")).reset_index()
    for q in a.q:
        for target in ("CH4", "H2O", "joint"):
            gg = g[(g.q == q) & (g.target == target)]; L.append(f"== q = {q:.1f}, target {target} ({gg.targets.iloc[0]:.0f} targets on average): targets lost, mean over draws")
            base = gg[gg.strategy.str.startswith("classifier alone") | gg.strategy.str.contains("every planet") | gg.strategy.str.startswith("AIRS")]
            cl = lambda r: (f"   acc {100*r.acc:5.1f} %  recall hazed/clean targets {100*r.rh:3.0f}/{100*r.rc:3.0f} %" if target != "joint" else "")
            for _, r in base.iterrows():
                L.append(f"  {r.strategy:<58} {r.lost:6.1f} ± {r.sd:4.1f}" + cl(r))
            L.append(f"  {'ordering of the retrieval budget':<40} " + " ".join(f"f={f:.0%}".rjust(8) for f in FS[1:-1]) + ("   | at f=20 %: acc, recall hazed/clean" if target != "joint" else ""))
            for r in RULES + ["random, clear-model retrieval", "optical, clear-model retrieval"]:
                rr = gg[gg.strategy == f"+ {r}"]
                if len(rr):
                    r2 = rr[rr.f == 0.2].iloc[0]
                    L.append(f"  {('+ ' + r):<40} " + " ".join(f"{rr[rr.f == f].lost.iloc[0]:8.1f}" for f in FS[1:-1]) + (f"   | {100*r2.acc:5.1f} %  {100*r2.rh:3.0f}/{100*r2.rc:3.0f} %" if target != "joint" else ""))
            L.append("")
    # how selective is each rule: fraction of hazed and of clean planets inside the top f, q = the middle value
    q = a.q[len(a.q) // 2]; L.append(f"== detection: share of hazed / clean planets inside the fraction f each rule ranks most suspect (q = {q:.1f}, mean over draws)")
    for r in RULES[1:-1]:
        hz, cl = {f: [] for f in FS[1:-1]}, {f: [] for f in FS[1:-1]}
        for d in range(a.draws):
            h = np.random.default_rng(SEED + 1000 + d).uniform(0, 1, N) < q
            s = np.where(h, rule_s["haze3e7"][r][idx], rule_s["clean"][r][idx]); s = np.where(np.isfinite(s), s, -np.inf); order = np.argsort(-s)
            for f in FS[1:-1]:
                flag = np.zeros(N, bool); flag[order[:int(round(f * N))]] = True; hz[f].append(flag[h].mean()); cl[f].append(flag[~h].mean())
        L.append(f"  {r:<14} " + "  ".join(f"f={f:.0%}: {100*np.mean(hz[f]):3.0f} / {100*np.mean(cl[f]):3.0f} %" for f in FS[1:-1]))
    txt = "\n".join(L); print(txt); open(os.path.join(RESULTS, f"e3_budget_{a.fallback}.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
