"""Held-out check of the decline thresholds. Every rule's threshold in alfnoor_diagnostic.py is the 90th percentile of its
score on the clean POP-I test spectra, so the clean decline rate is 10 % by construction. Here POP-I is split in half at
random (20 splits): the threshold is set on one half and the decline rates on the other half are reported for every case,
against the in-sample rates of diagnostic.txt.

Usage: python threshold_holdout.py [--splits 20]   -> results/threshold_holdout.{txt,csv}
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402

RULES = ["margin", "ensemble", "mahalanobis", "knn", "pca_recon", "disagree", "optical", "bandmetric"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--splits", type=int, default=20); a = ap.parse_args()
    cases, Pte, e, sig = D.test_cases(); sc = D.Scorer(e); n = len(Pte)
    S = {c: sc.scores(cases[c], sig=sig, Pb=Pte)[0] for c in D.CASES}
    rows = []
    for rule in RULES:
        for mol in D.MOLS:
            for k in D.LEARNERS:
                s0 = sc.get(S["clean"], rule, mol, k); v0 = np.isfinite(s0)
                if v0.sum() < 100:
                    continue
                thr_in = np.quantile(s0[v0], 1 - D.DECLINE)
                for sp in range(a.splits):
                    A = np.random.default_rng(SEED + 500 + sp).uniform(0, 1, n) < 0.5; B = ~A
                    thr = np.quantile(s0[v0 & A], 1 - D.DECLINE)
                    for c in D.CASES:
                        s = sc.get(S[c], rule, mol, k); v = np.isfinite(s) & B
                        rows.append(dict(rule=rule, molecule=mol, learner=k, split=sp, case=c, declined_heldout=float((s[v] > thr).mean()), declined_insample=float((s[v] > thr_in).mean())))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "threshold_holdout.csv"), index=False)
    g = df.groupby(["rule", "case"]).agg(held=("declined_heldout", "mean"), held_sd=("declined_heldout", "std"), ins=("declined_insample", "mean")).reset_index()
    L = [f"Decline thresholds set on a random half of POP-I and applied to the other half ({a.splits} splits; mean over molecules, learners and splits): declined fraction, held-out (sd over splits) vs in-sample", ""]
    L.append(f"{'rule':<12}" + "".join(f"{c:>24}" for c in D.CASES)); L.append(f"{'':<12}" + "".join(f"{'held (sd) / in-sample':>24}" for c in D.CASES))
    for rule in RULES:
        gg = g[g.rule == rule].set_index("case")
        L.append(f"{rule:<12}" + "".join(f"{100*gg.loc[c].held:5.1f} ({100*gg.loc[c].held_sd:3.1f}) / {100*gg.loc[c].ins:5.1f}".rjust(24) for c in D.CASES))
    clean = g[g.case == "clean"]; L.append(""); L.append(f"clean decline rate held out: {100*clean.held.min():.1f}-{100*clean.held.max():.1f} % against 10.0 % in-sample by construction")
    txt = "\n".join(L); print(txt); open(os.path.join(RESULTS, "threshold_holdout.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
