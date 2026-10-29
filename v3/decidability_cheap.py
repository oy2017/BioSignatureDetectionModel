"""Can anything cheap recognise the irrecoverable (undecidable) class? Three tests on the clean POP-I spectra, with the
retrieval of every planet (identifiability_calibration.csv) supplying the truth: a label is undecidable if the posterior
probability of its true class is <= 0.95.
  1. AUROC for undecidability of every existing score (the classifier's margin and ensemble spread, the three distance
     rules, the optical excess, input disagreement, band metric, muting) and of the bulk-feature regressor.
  2. A learned flag: a random forest trained out of fold (5-fold) on the normalised 104-bin spectrum plus the bulk
     features, with the retrieval's undecidable labels as the target; AUROC, and the accepted accuracy of the classifier
     when the planets it ranks most undecidable are declined (10-30 % coverage loss), against the margin at equal coverage.
  3. The learned flag as a retrieval-budget ordering (CH4 and H2O targets lost at the 60 % selection, f = 10-50 %),
     against random, the margin, k-NN, the measured dchi2 and the retrieval's own p_true.
Usage: python decidability_cheap.py   -> results/decidability_cheap.{txt,csv}
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D, alfnoor_trust as AT  # noqa: E402

LEARN = ("KNN", "MLP", "RFC"); FS = [0.1, 0.2, 0.3, 0.5]


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    cases, Pte, e, sig = D.test_cases(); sc = D.Scorer(e); X = cases["clean"]; n = len(Pte)
    S, P, M = sc.scores(X, sig=sig, Pb=Pte)
    C = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv")).drop_duplicates("name"); idx = {nm: i for i, nm in enumerate(Pte.name)}
    C = C[C.name.isin(idx)]; ci = C.name.map(idx).to_numpy()
    p_true, p_pres = {}, {}
    for m in D.MOLS:
        a = np.full(n, np.nan); b = np.full(n, np.nan); a[ci] = C[f"{m}_p_true"].to_numpy(float); b[ci] = C[f"{m}_p_present"].to_numpy(float); p_true[m] = a; p_pres[m] = b
    have = np.isfinite(p_true["CH4"]); y_true = {m: (Pte[f"atm {m}"].to_numpy() > -4).astype(int) for m in D.MOLS}
    dec, feats = D.decidability_regressor(Pte, sig, X); Fb = feats.drop(columns="index").to_numpy(float)
    I = pd.read_csv(os.path.join(RESULTS, "identifiability_none_tier1.csv")); I = I[I.name.isin(idx)]; I["i"] = I.name.map(idx)
    Z = AT.norm(X); Xf = np.hstack([Z, Fb, np.log10(sig.mean(1))[:, None]])
    L = ["Cheap recognition of the irrecoverable class on clean POP-I spectra (truth: retrieval of every planet; undecidable = p_true <= 0.95)", ""]
    rows = []
    L.append("== 1. AUROC for undecidability (mean over the three learners where a score is per learner); n planets with a posterior")
    L.append(f"  {'mol':<4} {'n':>4} {'undec%':>7} " + " ".join(f"{r:>11}" for r in ["margin", "ensemble", "mahalan.", "knn", "pca", "optical", "disagree", "bandmetric", "muting", "bulk-regr"]))
    for m in D.MOLS:
        u = (p_true[m] <= 0.95)[have]; cells = []
        for r in ["margin", "ensemble", "mahalanobis", "knn", "pca_recon", "optical", "disagree", "bandmetric", "muting"]:
            v = [AT.auroc(u[np.isfinite(sc.get(S, r, m, k)[have])], sc.get(S, r, m, k)[have][np.isfinite(sc.get(S, r, m, k)[have])]) if np.isfinite(sc.get(S, r, m, k)[have]).sum() > 50 else np.nan for k in LEARN]; a = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan; cells.append(a); rows.append(dict(test="auroc_undecidable", molecule=m, score=r, auroc=a))
        if dec is not None and (dec.molecule == m).any():
            Dm = dec[dec.molecule == m]; s = np.full(n, np.nan); s[Dm.i.to_numpy()] = -Dm.pred.to_numpy(); v = np.isfinite(s) & have
            a = AT.auroc((p_true[m] <= 0.95)[v], s[v]); cells.append(a); rows.append(dict(test="auroc_undecidable", molecule=m, score="bulk-regressor", auroc=a))
        else:
            cells.append(np.nan)
        L.append(f"  {m:<4} {int(have.sum()):>4} {100*u.mean():7.1f} " + " ".join(f"{c:11.3f}" for c in cells))
    L.append(""); L.append("== 2. learned flag (random forest, 5-fold out of fold): AUROC for undecidability from the spectrum + bulk features, spectrum only, bulk only;")
    L.append("      then the classifier's accuracy (mean of KNN/MLP/RFC probabilities) on the planets kept when the most undecidable 10/20/30 % are declined, vs declining by the margin at equal coverage")
    oof = {}
    for m in D.MOLS:
        u = (p_true[m] <= 0.95).astype(int); v = np.where(have)[0]; res = {}
        for name, F_ in (("spectrum+bulk", Xf), ("spectrum", Z), ("bulk", Fb)):
            o = np.full(n, np.nan)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(F_[v], u[v]):
                o[v[te]] = RandomForestClassifier(500, random_state=SEED, n_jobs=4, min_samples_leaf=3).fit(F_[v[tr]], u[v[tr]]).predict_proba(F_[v[te]])[:, 1]
            res[name] = AT.auroc(u[v], o[v]); rows.append(dict(test="auroc_learned", molecule=m, score=name, auroc=res[name]))
            if name == "spectrum+bulk": oof[m] = o
        pm = np.mean([P["all"][(m, k)] for k in LEARN], axis=0); wrong = ((pm >= 0.5).astype(int) != y_true[m])
        marg = 1 - np.abs(2 * pm - 1); acc_all = 1 - wrong[v].mean(); cells = []
        for q in (0.1, 0.2, 0.3):
            keep_f = oof[m][v] <= np.quantile(oof[m][v], 1 - q); keep_m = marg[v] <= np.quantile(marg[v], 1 - q)
            cells.append((1 - wrong[v][keep_f].mean(), 1 - wrong[v][keep_m].mean()))
            rows.append(dict(test="accepted_accuracy", molecule=m, score="learned", coverage=1 - q, auroc=1 - wrong[v][keep_f].mean())); rows.append(dict(test="accepted_accuracy", molecule=m, score="margin", coverage=1 - q, auroc=1 - wrong[v][keep_m].mean()))
        L.append(f"  {m:<4} AUROC spectrum+bulk {res['spectrum+bulk']:.3f}  spectrum {res['spectrum']:.3f}  bulk {res['bulk']:.3f}   accuracy all {100*acc_all:5.1f} %; declining 10/20/30 %: learned "
                 + " / ".join(f"{100*a:5.1f}" for a, _ in cells) + "   margin " + " / ".join(f"{100*b:5.1f}" for _, b in cells))
    L.append(""); L.append("== 3. retrieval budget: targets lost at the 60 % selection when the fraction f ranked most suspect is re-scored by the retrieval (planets with a posterior)")
    v = np.where(have)[0]; N = len(v); slots = int(round(D.BUDGET * N)); rng = np.random.default_rng(SEED)
    for target in ("CH4", "H2O"):
        y = y_true[target][v]; score = np.mean([P["all"][(target, k)] for k in LEARN], axis=0)[v]; ret = p_pres[target][v]
        dch = np.full(n, np.nan); Dm = I[I.molecule == target]; dch[Dm.i.to_numpy()] = -np.log10(Dm.dchi2.clip(lower=1e-3).to_numpy()); dch = dch[v]
        orders = {"random (mean of 200)": None, "learned flag": oof[target][v], "margin": (1 - np.abs(2 * score - 1)), "knn": S[("knn", None)][v], "mahalanobis": S[("mahalanobis", None)][v],
                  "dchi2 measured (oracle)": np.where(np.isfinite(dch), dch, -np.inf), "p_true (retrieval's own)": -p_true[target][v]}
        L.append(f"-- {target} ({int(y.sum())} targets): lost at f = " + "  ".join(f"{int(100*f):3d}%" for f in FS) + f"   (classifier alone {int(y.sum() - (y[np.argsort(-score)[:slots]] == 1).sum())})")
        for name, s in orders.items():
            out = []
            for f in FS:
                k = int(round(f * N)); vals = []
                for rep in range(200 if s is None else 1):
                    o = np.argsort(-(rng.uniform(0, 1, N) if s is None else s)); flag = np.zeros(N, bool); flag[o[:k]] = True
                    s_ = np.where(flag, ret, score); sel = np.zeros(N, bool); sel[np.argsort(-s_)[:slots]] = True; vals.append(int(y.sum() - (sel & (y == 1)).sum()))
                out.append(float(np.mean(vals))); rows.append(dict(test="budget", molecule=target, score=name, coverage=f, auroc=float(np.mean(vals))))
            L.append(f"   {name:<28} " + "  ".join(f"{x:4.0f}" for x in out))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "decidability_cheap.csv"), index=False)
    txt = "\n".join(L); print(txt); open(os.path.join(RESULTS, "decidability_cheap.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
