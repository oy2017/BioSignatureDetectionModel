"""Draw-to-draw spread of the random ordering in the clean-spectrum triage (Table 8).

alfnoor_diagnostic.py scores the 'random' budget with one draw. This script repeats the triage with the same
scores and posteriors for many random draws and reports mean and standard deviation of the targets lost, next to
the deterministic orderings, so that differences of a few targets can be read against the spread.

Usage: python triage_spread.py [--draws 200]   -> results/triage_spread.txt
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=200); a = ap.parse_args()
    cases, Pte, e, sig = D.test_cases(); sc = D.Scorer(e); n = len(Pte); slots = int(round(D.BUDGET * n))
    S, P, _ = sc.scores(cases["clean"]); RET = D.retrieval_posteriors(Pte)
    rules = [r for r in ("margin", "ensemble", "mahalanobis", "knn", "pca_recon", "optical", "disagree", "bandmetric")]
    L = [f"Clean-spectrum triage: targets lost at the {int(D.BUDGET*100)} % selection when the top fraction f by each ordering is re-scored by the retrieval;",
         f"random = mean +/- sd over {a.draws} draws (alfnoor_diagnostic.py reports one draw)", ""]
    for target in ("CH4", "H2O"):
        pp, pt = RET[target]; y = (Pte[f"atm {target}"].to_numpy() > -4).astype(int)
        score = np.mean([P["all"][(target, k)] for k in ("KNN", "MLP", "RFC")], axis=0)
        def lost(order, f):
            flag = np.zeros(n, bool); flag[np.argsort(-order)[:int(round(f * n))]] = True; s_ = np.where(flag, pp, score)
            sel = np.zeros(n, bool); sel[np.argsort(-s_)[:slots]] = True; return int(y.sum() - (sel & (y == 1)).sum())
        fs = [0.1, 0.2, 0.3, 0.5, 0.7]
        rng = np.random.default_rng(SEED)
        R = np.array([[lost(rng.uniform(0, 1, n), f) for f in fs] for _ in range(a.draws)])
        L.append(f"== {target} ({int(y.sum())} targets)      " + "".join(f"f={int(100*f)}%      " for f in fs))
        L.append("  random              " + "".join(f"{m:5.1f} +/- {s:3.1f}  " for m, s in zip(R.mean(0), R.std(0))))
        L.append("  retrieval's own p_true (unattainable) " + "".join(f"{lost(-pt, f):5d}        " for f in fs))
        for r in rules:
            s_ = np.nanmean(np.column_stack([sc.get(S, r, target, k) for k in ("KNN", "MLP", "RFC")]), axis=1)
            if np.isfinite(s_).sum() > 50:
                L.append(f"  {r:<20}" + "".join(f"{lost(np.where(np.isfinite(s_), s_, -np.inf), f):5d}        " for f in fs))
        L.append("")
    open(os.path.join(RESULTS, "triage_spread.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
