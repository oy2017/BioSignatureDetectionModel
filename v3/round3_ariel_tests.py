"""U2 to U6 of REVISION_EXPECTATIONS.md, for the design at Ariel coverage: declining in selection, error ranking under departures, recognising undecidable labels, ten noise draws, and the held-out grids through all bins. Writes results/round3/round3_ariel_tests.txt."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_ariel as MA  # noqa: E402
import matched_learner as ML  # noqa: E402
import panel_tier1 as PT  # noqa: E402
import round3_ariel as RA  # noqa: E402
import round3_learner as R  # noqa: E402

MOLS = R.MOLS; CC = MA.CASES + ["plhaze29"]


def mute(cases, e):
    from bin_spectra import bin_native
    wl = np.load(os.path.join(R.DATA, "native_wl.npy")); X0 = np.load(os.path.join(AF.DIRS["faithful"], "pop1_native.npy")).astype(float); nf = bin_native(X0[np.isfinite(X0).all(1)], wl, e); eps = cases["clean"] - nf; mu = nf.mean(1, keepdims=True); cases = dict(cases); cases["mute3"] = mu + (nf - mu) / 3 + eps; return cases


def main():
    cases, Pte, e, sig = D.test_cases(extra=("haze2e6", "plhaze29")); X, P = RA.population(e); s = AF.sigma(P, e, "radiometric"); Y = (P[[f"atm {m}" for m in MOLS]].to_numpy() > -4).astype(int); hold = np.random.default_rng(SEED + 77).uniform(size=len(X)) < 0.15; nets = []
    for k in range(R.MEMBERS):
        rng = np.random.default_rng(SEED + 8100 + k); Z = np.vstack([RA.draw(X[~hold], s[~hold], rng) for _ in range(R.COPIES)]); nets.append(MLPClassifier((256, 128), batch_size=256, early_stopping=True, n_iter_no_change=6, max_iter=60, random_state=SEED + 8100 + k).fit(Z, np.tile(Y[~hold], (R.COPIES, 1)))); print(k, flush=True)
    prob = lambda Z: np.mean([n.predict_proba(Z) for n in nets], axis=0); t = R.thresholds(prob(RA.draw(X[hold], s[hold], np.random.default_rng(SEED + 8199))), Y[hold]); cases = mute(cases, e); L = []; slots = int(round(0.6 * len(Pte))); Rc = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv")).drop_duplicates("name").set_index("name").reindex(Pte.name.to_numpy()); pc = prob(MA.feats(cases["clean"], sig)); thr = [np.quantile(-np.abs(pc[:, i] - t[i]), 0.9) for i in range(4)]
    lost = lambda sc, y: int(y.sum() - y[np.argsort(-sc)[:slots]].sum()); L.append("U2, U3: per case, targets lost CH4 / H2O with the ensemble, with its 10 % least confident sent to the back, and error AUROC of its distance from the threshold, per molecule")
    for c in CC:
        p = prob(MA.feats(cases[c], sig)); a, b, u = [], [], []
        for i, m in enumerate(MOLS):
            y = (Pte[f"atm {m}"] > -4).to_numpy(); conf = -np.abs(p[:, i] - t[i]); d = conf > thr[i]; u.append(auc((p[:, i] >= t[i]) != y, conf))
            if m in ("CH4", "H2O"): a.append(lost(p[:, i], y)); b.append(lost(p[:, i] - 10 * d, y))
        L.append(f"  {c:<10} {a[0]} / {a[1]}  declined {b[0]} / {b[1]}  error AUROC {' '.join(f'{v:.2f}' for v in u)}  mean {np.mean(u):.2f}")
    L.append("U4: AUROC for recognising undecidable labels (p_true <= .95) on clean spectra, CH4 H2O CO2 NH3: " + " ".join(f"{auc((Rc[f'{m}_p_true'].to_numpy() <= 0.95), -np.abs(pc[:, i] - t[i])):.2f}" for i, m in enumerate(MOLS)))
    D_ = {c: [] for c in CC}
    for ns in range(1, 11):
        cs, _, _, _ = D.test_cases(extra=("haze2e6", "plhaze29"), noise_seed=ns); cs = mute(cs, e)
        for c in CC:
            p = prob(MA.feats(cs[c], sig)); D_[c].append([lost(p[:, i], (Pte[f"atm {m}"] > -4).to_numpy()) for i, m in enumerate(MOLS) if m in ("CH4", "H2O")])
        print("draw", ns, flush=True)
    L.append("U5: targets lost, mean +- sd over ten draws, CH4 / H2O"); L += [f"  {c:<10} {np.mean(v, 0)[0]:.1f} +- {np.std(v, 0)[0]:.1f} / {np.mean(v, 0)[1]:.1f} +- {np.std(v, 0)[1]:.1f}" for c, v in D_.items()]
    open(os.path.join(R.OUT, "round3_ariel_tests.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)
    ctx = PT.Context(); Jl = ML.jobs_list(); edges, _ = J.tier3_layout(); L2 = ["", "U6: held-out grids through all 104 bins at Tier-1 noise, balanced credit (present, false-present), and ensemble minus design with a 95 % bootstrap over planets"]
    for cond in ("local", "crossfield"):
        Z = np.load(os.path.join(R.GRIDS, f"grid_{cond}.npz")); T = pd.read_csv(os.path.join(R.GRIDS, f"truth_{cond}.csv")).melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in MOLS], var_name="molecule", value_name="v"); T["molecule"] = T.molecule.str[2:]; rows = []
        for i, j in enumerate(Jl):
            pl = j[1]; sg = j[6]; G = Z[pl]; r = np.random.default_rng(SEED + 2700 + i); B = G + r.normal(0, 1, G.shape) * sg; p = prob(MA.feats(B, np.tile(sg, (len(B), 1)))); res = PT.score(ctx, B, np.ones(104, bool), ML.BASE(pl), float(np.mean(sg ** 2)) * 1e12, np.inf, 0.0)
            for a, m in enumerate(MOLS):
                for d, pr, ab in (("ensemble", p[:, a] >= t[a], p[:, a] < t[a]), ("classifier", res[m][0]["published"] >= 3, res[m][0]["published"] <= 1), ("haze+deck", res[m][0]["haze+deck-retrained"] >= 3, res[m][0]["haze+deck-retrained"] <= 1), ("AIRS-only", res[m][0]["AIRS-only"] >= 3, res[m][0]["AIRS-only"] <= 1)):
                    rows.append(pd.DataFrame(dict(planet=pl, molecule=m, design=d, row=np.arange(len(B)), present=pr.astype(float), absent=ab.astype(float))))
        A = pd.concat(rows).merge(T, on=["planet", "row", "molecule"]); A["y"] = A.v > -4; A["u"] = A.planet.map(ML.BASE); A["c"] = np.where(A.y, A.present, A.absent) + 0.5 * (1 - A.present - A.absent); S = A.groupby(["design", "u", "y"]).c.agg(["sum", "count"]); us = sorted(A.u.unique()); ix = np.random.default_rng(405).integers(0, len(us), (4000, len(us)))
        def arr(d):
            q = S.loc[d].unstack("y").reindex(us).fillna(0); return q["sum"][True].to_numpy(), q["count"][True].to_numpy(), q["sum"][False].to_numpy(), q["count"][False].to_numpy()
        def B_(a, i=None):
            i = np.arange(len(us)) if i is None else i; return 0.5 * (a[0][i].sum(-1) / a[1][i].sum(-1) + a[2][i].sum(-1) / a[3][i].sum(-1))
        for d in ("ensemble", "classifier", "AIRS-only", "haze+deck"):
            g = A[A.design == d]; dif = B_(arr("ensemble"), ix) - B_(arr(d), ix); L2.append(f"  {cond:<11}{d:<12} {B_(arr(d)):.3f} ({g[g.y].c.mean():.3f}, {g[~g.y].present.mean():.3f})  ensemble minus this {B_(arr('ensemble')) - B_(arr(d)):+.3f} ({np.percentile(dif, 2.5):+.3f},{np.percentile(dif, 97.5):+.3f})")
        print(cond, flush=True)
    open(os.path.join(R.OUT, "round3_ariel_tests.txt"), "a").write("\n".join(L2) + "\n"); print("\n".join(L2))


if __name__ == "__main__":
    main()
