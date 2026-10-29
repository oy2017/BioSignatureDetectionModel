"""Table 11 over test-noise draws: CH4-bearing targets lost at the 60 % Tier-2 selection under each departure, with no
retrieval, for the no-retrieval strategies of alfnoor_diagnostic's selection simulation (classifier alone, routed by the
optical flag to the AIRS-only variant, AIRS-only everywhere, the existing rules declining to the back of the queue, band
metric alone), repeated over `--draws` realisations of the test noise (the learners and their training draw fixed).

Usage: python table11_spread.py [--draws 10]   -> results/table11_spread.{txt,csv}
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402

CASES = ["clean", "haze3e7", "spots20", "white_x2"]; LEARN = ("KNN", "MLP", "RFC")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=10); a = ap.parse_args()
    sc = None; rows = []
    for d in range(a.draws):
        cases, Pte, e, sig = D.test_cases(noise_seed=d); sc = sc or D.Scorer(e); n = len(Pte); slots = int(round(D.BUDGET * n))
        y = (Pte["atm CH4"].to_numpy() > -4).astype(int)
        S, P, M = {}, {}, {}
        for c in CASES:
            S[c], P[c], M[c] = sc.scores(cases[c], sig=sig, Pb=Pte)
        rng = np.random.default_rng(SEED + d)
        for c in CASES:
            score = np.mean([P[c]["all"][("CH4", k)] for k in LEARN], axis=0); fb = np.mean([P[c]["airs_only"][("CH4", k)] for k in LEARN], axis=0)
            strat = {"classifier alone": score, "AIRS-only variant everywhere": fb, "band metric alone": M[c]["CH4"], "random": rng.uniform(0, 1, n)}
            for rule in D.OLD_RULES + ["optical", "disagree", "bandmetric", "muting"]:
                s = np.nanmean(np.column_stack([sc.get(S[c], rule, "CH4", k) for k in LEARN]), axis=1)
                s0 = np.nanmean(np.column_stack([sc.get(S["clean"], rule, "CH4", k) for k in LEARN]), axis=1); v = np.isfinite(s0)
                thr = np.quantile(s0[v], 1 - D.DECLINE); flag = np.isfinite(s) & (s > thr)
                strat[f"{rule}, declined to the back" if rule in D.OLD_RULES else f"routed by {rule} to AIRS-only"] = np.where(flag, score - 10, score) if rule in D.OLD_RULES else np.where(flag, fb, score)
            for name, s_ in strat.items():
                sel = np.zeros(n, bool); sel[np.argsort(-s_)[:slots]] = True
                rows.append(dict(draw=d, case=c, strategy=name, targets=int(y.sum()), lost=int(y.sum() - (sel & (y == 1)).sum())))
        print(f"draw {d} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "table11_spread.csv"), index=False)
    g = df.groupby(["strategy", "case"]).lost.agg(["mean", "std"]).reset_index()
    L = [f"Table 11 over {a.draws} test-noise draws: CH4-bearing targets lost (of {int(df.targets.iloc[0])}) at the 60 % selection, mean +/- sd over draws (draw 0 is the committed realisation)", ""]
    L.append(f"{'strategy':<42}" + "".join(f"{c:>16}" for c in CASES))
    for name in ["classifier alone", "routed by optical to AIRS-only", "AIRS-only variant everywhere", "routed by disagree to AIRS-only", "routed by bandmetric to AIRS-only", "routed by muting to AIRS-only",
                 "knn, declined to the back", "mahalanobis, declined to the back", "pca_recon, declined to the back", "margin, declined to the back", "ensemble, declined to the back", "band metric alone", "random"]:
        gg = g[g.strategy == name].set_index("case"); L.append(f"{name:<42}" + "".join(f"{gg.loc[c]['mean']:8.1f} ± {gg.loc[c]['std']:4.1f}".rjust(16) for c in CASES))
    d0 = df[df.draw == 0].groupby(["strategy", "case"]).lost.first()
    L.append(""); L.append("committed draw (Table 11): " + "; ".join(f"{c}: classifier {d0[('classifier alone', c)]}, routed {d0[('routed by optical to AIRS-only', c)]}" for c in CASES))
    txt = "\n".join(L); print(txt); open(os.path.join(RESULTS, "table11_spread.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
