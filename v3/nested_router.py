"""Nested-input disagreement as a localising detector of simulator departures, and the three-level router it defines.

The classifier exists in three nested input sets (all = photometry + NIRSpec + AIRS; no_optical = NIRSpec + AIRS;
airs_only). Two disagreement scores per (molecule, learner):
    D1 = |p_all - p_no_optical|       sensitive to a departure confined to the three optical points
    D2 = |p_no_optical - p_airs_only|  sensitive to a departure reaching the NIRSpec range (1.1-1.95 um)
each thresholded at the 90th percentile of its clean POP-I distribution. The nested router sends a cell to the
largest input set the nest agrees on: airs_only if D2 fires, else no_optical if D1 fires, else all. Compared with the
two-level 'either' router of alfnoor_diagnostic (optical excess or max-disagreement -> airs_only), and with each
variant used everywhere, on every departure case.

Usage: python nested_router.py   -> results/nested_router.{txt,csv}
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402

CASES = ["clean", "haze3e7", "haze2e6", "spots20", "cloud1e2", "compound", "white_x2"]


def main():
    cases, Pte, e, sig = D.test_cases(extra=("haze2e6",)); cases = {c: cases[c] for c in CASES if c in cases}
    sc = D.Scorer(e); y = {m: (Pte[f"atm {m}"] > -4).astype(int).to_numpy() for m in D.MOLS}
    S = {c: sc.scores(X) for c, X in cases.items()}
    q = lambda s: np.quantile(s[np.isfinite(s)], 1 - D.DECLINE)
    rows = []
    for c in cases:
        _, P, _ = S[c]; S0 = S["clean"][0]; Sc = S[c][0]
        f_opt = Sc[("optical", None)] > q(S0[("optical", None)])
        for m in D.MOLS:
            for k in D.LEARNERS:
                pa, pn, pr = P["all"][(m, k)], P["no_optical"][(m, k)], P["airs_only"][(m, k)]
                P0 = S["clean"][1]; d1_0 = np.abs(P0["all"][(m, k)] - P0["no_optical"][(m, k)]); d2_0 = np.abs(P0["no_optical"][(m, k)] - P0["airs_only"][(m, k)])
                d1 = np.abs(pa - pn); d2 = np.abs(pn - pr); f1 = d1 > q(d1_0); f2 = d2 > q(d2_0)
                f_dis = Sc[("disagree", (m, k))] > q(S0[("disagree", (m, k))]); f_either = f_opt | f_dis
                acc = lambda p: float(((p >= 0.5) == y[m]).mean())
                rec = lambda p: float(((p >= 0.5)[y[m] == 1]).mean())
                nested = np.where(f2, pr, np.where(f1, pn, pa)); either = np.where(f_either, pr, pa)
                opt_only = np.where(f_opt, pr, pa); d1_route = np.where(f1, pn, pa)
                rows.append(dict(case=c, molecule=m, learner=k, fire_D1=f1.mean(), fire_D2=f2.mean(), fire_optical=f_opt.mean(), fire_either=f_either.mean(),
                                 to_no_optical=(f1 & ~f2).mean(), to_airs=f2.mean(),
                                 acc_all=acc(pa), acc_no_optical=acc(pn), acc_airs=acc(pr), acc_optical_router=acc(opt_only), acc_either=acc(either),
                                 acc_D1_router=acc(d1_route), acc_nested=acc(nested),
                                 rec_all=rec(pa), rec_either=rec(either), rec_nested=rec(nested)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "nested_router.csv"), index=False)
    g = df.groupby("case", sort=False).mean(numeric_only=True).loc[[c for c in CASES if c in cases]]
    L = ["Nested-input disagreement (D1 = |p_all - p_no_optical|, D2 = |p_no_optical - p_airs_only|; thresholds = 90th percentile of clean POP-I per molecule and learner)",
         "mean over the four molecules and four learners; accuracy at the 1e-4 threshold on POP-I (954 planets)", "",
         f"{'case':<10} {'D1 fires':>9} {'D2 fires':>9} {'optical':>8} {'either':>7} | {'all':>6} {'no_opt':>7} {'AIRS':>6} {'opt->A':>7} {'either->A':>9} {'D1->noopt':>9} {'nested':>7} | {'recall all':>10} {'either':>7} {'nested':>7}"]
    for c, r in g.iterrows():
        L.append(f"{c:<10} {100*r.fire_D1:>8.1f}% {100*r.fire_D2:>8.1f}% {100*r.fire_optical:>7.1f}% {100*r.fire_either:>6.1f}% | {100*r.acc_all:>6.1f} {100*r.acc_no_optical:>7.1f} {100*r.acc_airs:>6.1f} {100*r.acc_optical_router:>7.1f} {100*r.acc_either:>9.1f} {100*r.acc_D1_router:>9.1f} {100*r.acc_nested:>7.1f} | {100*r.rec_all:>9.1f}% {100*r.rec_either:>6.1f}% {100*r.rec_nested:>6.1f}%")
    L += ["", "per molecule (mean over learners): D1 / D2 firing and accuracy all -> either -> nested"]
    for c in g.index:
        for m in D.MOLS:
            r = df[(df.case == c) & (df.molecule == m)].mean(numeric_only=True)
            L.append(f"  {c:<10} {m:<4} D1 {100*r.fire_D1:5.1f}%  D2 {100*r.fire_D2:5.1f}%   acc {100*r.acc_all:5.1f} -> {100*r.acc_either:5.1f} -> {100*r.acc_nested:5.1f}   recall {100*r.rec_all:5.1f} -> {100*r.rec_either:5.1f} -> {100*r.rec_nested:5.1f}")
    txt = "\n".join(L); print(txt); open(os.path.join(RESULTS, "nested_router.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
