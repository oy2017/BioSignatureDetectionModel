"""V of REVISION_EXPECTATIONS.md: the margin rule as the matched ensemble's diagnostic. Writes results/round3/round3_margin.txt. Needs round3_rules.py and rule_comparison.py first."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import matched_ariel as MA  # noqa: E402
import matched_learner as ML  # noqa: E402
import round3_ariel as RA  # noqa: E402
import round3_ariel_tests as RT  # noqa: E402
import round3_learner as R  # noqa: E402

MOLS = R.MOLS; KEEP = 0.7; OUT = R.OUT; RULES = ["margin", "ensemble", "knn", "mahalanobis", "pca_recon"]


def inc(g, r):
    k = g[g[r] <= g[r].quantile(KEEP)]; f = lambda x: 0.5 * (x[x.y].c.mean() + x[~x.y].c.mean()); return f(k) - f(g), f(k)


def grids(L):
    E = pd.read_csv(os.path.join(OUT, "round3_rules_rows.csv.gz")); E["absent"] = 1 - E.present
    for c in ("rainout", "local", "crossfield"):
        Y = pd.read_csv(os.path.join(R.GRIDS, f"truth_{c}.csv")).melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in MOLS], var_name="molecule", value_name="v"); Y["molecule"] = Y.molecule.str[2:]; C = pd.read_csv(os.path.join(OUT, f"rule_rows_{c}.csv.gz")).merge(Y, on=["planet", "row", "molecule"]); e = E[E.cond == c].merge(Y, on=["planet", "row", "molecule"])
        for q in (C, e): q["y"] = q.v > -4; q["c"] = np.where(q.y, q.present, q.absent) + 0.5 * (1 - q.present - q.absent); q["u"] = q.planet.map(ML.BASE)
        for p in ("JWST", "Tier-1"):
            a, b = e[e.precision == p], C[C.precision == p]; I = {r: inc(b, r)[0] for r in RULES}; best = max(I, key=I.get); da, kept = inc(a, "margin"); us = sorted(set(a.u)); ga = {u: x for u, x in a.groupby("u")}; gb = {u: x for u, x in b.groupby("u")}; rng = np.random.default_rng(405); o = []
            for _ in range(300):
                pick = rng.choice(us, len(us)); o.append(inc(pd.concat([ga[u] for u in pick]), "margin")[0] - inc(pd.concat([gb[u] for u in pick]), best)[0])
            L.append(f"{c:<11}{p:<7} ensemble margin adds {da:+.3f} (kept {kept:.3f}); classifier best rule {best} adds {I[best]:+.3f}; difference {da - I[best]:+.3f} ({np.percentile(o, 2.5):+.3f},{np.percentile(o, 97.5):+.3f})"); print(L[-1], flush=True)


def main():
    L = []; grids(L); cases, Pte, e, sig = D.test_cases(extra=("haze2e6", "plhaze29")); X, P = RA.population(e); s = AF.sigma(P, e, "radiometric"); Y = (P[[f"atm {m}" for m in MOLS]].to_numpy() > -4).astype(int); hold = np.random.default_rng(SEED + 77).uniform(size=len(X)) < 0.15; nets = []
    for k in range(R.MEMBERS):
        rng = np.random.default_rng(SEED + 8100 + k); Z = np.vstack([RA.draw(X[~hold], s[~hold], rng) for _ in range(R.COPIES)]); nets.append(MLPClassifier((256, 128), batch_size=256, early_stopping=True, n_iter_no_change=6, max_iter=60, random_state=SEED + 8100 + k).fit(Z, np.tile(Y[~hold], (R.COPIES, 1))))
    prob = lambda Z: np.mean([n.predict_proba(Z) for n in nets], axis=0); t = R.thresholds(prob(RA.draw(X[hold], s[hold], np.random.default_rng(SEED + 8199))), Y[hold]); cases = RT.mute(cases, e); sc = D.Scorer(e); LE = D.LEARNERS; lost = lambda v, y: int(y.sum() - y[np.argsort(-v)[:int(round(0.6 * len(y)))]].sum()); L.append("V3: per case: ensemble margin error AUROC by molecule, mean; re-implemented classifier margin AUROC mean; accuracy with no decline and on the kept 70 %")
    for c in RT.CC:
        p = prob(MA.feats(cases[c], sig)); Pc = sc.probs(cases[c])["all"]; a, b, ka, kb, na, nb = [], [], [], [], [], []
        for i, m in enumerate(MOLS):
            y = (Pte[f"atm {m}"] > -4).to_numpy(); q = p[:, i] >= t[i]; g = np.abs(p[:, i] - 0.5); a.append(auc(q != y, -g)); keep = g >= np.quantile(g, 0.3); ka.append((q == y)[keep].mean()); na.append((q == y).mean())
            for k in LE:
                pc = Pc[(m, k)]; q2 = pc >= 0.5; g2 = np.abs(pc - 0.5); b.append(auc(q2 != y, -g2)); keep2 = g2 >= np.quantile(g2, 0.3); kb.append((q2 == y)[keep2].mean()); nb.append((q2 == y).mean())
        L.append(f"  {c:<10} ensemble {' '.join(f'{v:.2f}' for v in a)} mean {np.mean(a):.2f}, accuracy {100 * np.mean(na):.1f} kept {100 * np.mean(ka):.1f}; classifier mean {np.mean(b):.2f}, accuracy {100 * np.mean(nb):.1f} kept {100 * np.mean(kb):.1f}")
    D_ = {c: [] for c in RT.CC}
    for ns in range(1, 11):
        cs, _, _, _ = D.test_cases(extra=("haze2e6", "plhaze29"), noise_seed=ns); cs = RT.mute(cs, e)
        for c in RT.CC:
            Pa = sc.probs(cs[c])["all"]; D_[c].append([lost(np.mean([Pa[(m, k)] for k in ("KNN", "MLP", "RFC")], axis=0), (Pte[f"atm {m}"] > -4).to_numpy()) for m in ("CH4", "H2O")])
        print("draw", ns, flush=True)
    L.append("V4: the re-implemented classifier, targets lost, mean +- sd over ten draws, CH4 / H2O"); L += [f"  {c:<10} {np.mean(v, 0)[0]:.1f} +- {np.std(v, 0)[0]:.1f} / {np.mean(v, 0)[1]:.1f} +- {np.std(v, 0)[1]:.1f}" for c, v in D_.items()]
    t_ = "\n".join(L); open(os.path.join(OUT, "round3_margin.txt"), "w").write(t_ + "\n"); print(t_)


if __name__ == "__main__":
    main()
