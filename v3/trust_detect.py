"""Test 2 of the trust programme: can the screen tell when it is outside its simulator?

Five out-of-distribution scores are computed for the frozen clean-trained screen on every
shifted test set, and judged on the only thing that matters for deployment: do they rank the
planets the screen gets WRONG above the ones it gets right, and what accuracy does the
accepted set reach when the least typical planets are declined at a threshold fixed on clean
data alone?

Scores (higher = less trustworthy):
  margin        1 - |2p - 1| from the frozen model (the naive baseline)
  ensemble      std of p across norm_xgb, norm_rf, norm_mlp (disagreement)
  mahalanobis   distance in the frozen model's feature space to the training cloud
  pca_recon     reconstruction error from a PCA of the training features (99 % variance)
  knn           mean distance to the 10 nearest training planets in feature space

Threshold: the 90th percentile of each score on the pooled CLEAN test spectra (declines 10 %
of clean planets), never tuned on a shifted set.

Usage: python trust_detect.py --config ariel
Writes results/ariel_trust_detect.txt / .csv
"""
import argparse, json, os, sys
import joblib, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, centres, load_split, metrics  # noqa: E402
from augment import shifted_test, corr_noise  # noqa: E402
from pipeline import MLPWrap  # noqa: E402

RERENDERED = ["cloud_1e3Pa", "cloud_1e4Pa", "haze_3p0e7", "haze_2p4e8", "tlse_spots10", "tlse_spots20", "tlse_mixed",
              "exotransmit", "exomol", "quenched", "compound_spots20_haze3e7"]
NOISE = [("white", 8), ("white", 5), ("correlated", 8), ("correlated", 5)]
DECLINE_CLEAN = 0.10


def load_pipe(cfg, key):
    d = joblib.load(os.path.join(MODELS, f"{cfg}_{key}.joblib"))
    if "keras_path" in d:
        import tensorflow as tf
        d["model"] = MLPWrap(tf.keras.models.load_model(d["keras_path"]))
    return d["features"], d["model"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel"); a = ap.parse_args(); cfg = a.config
    cen = centres(cfg)
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    ff, fm = load_pipe(cfg, best)
    ens = [load_pipe(cfg, k) for k in ("norm_xgb", "norm_rf", "norm_mlp")]

    Xtr, ytr, _ = load_split("train", cfg); Ztr = ff.transform(Xtr)
    mu = Ztr.mean(axis=0); C = np.cov(Ztr, rowvar=False) + 1e-3 * np.eye(Ztr.shape[1]); Ci = np.linalg.inv(C)
    pca = PCA(n_components=0.99, random_state=SEED).fit(Ztr)
    knn = NearestNeighbors(n_neighbors=10).fit(Ztr)

    def scores(X):
        Z = ff.transform(X); p = fm.predict_proba(Z)[:, 1]
        P = np.column_stack([m.predict_proba(f.transform(X))[:, 1] for f, m in ens])
        d = Z - mu
        return dict(margin=1 - np.abs(2 * p - 1), ensemble=P.std(axis=1),
                    mahalanobis=np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d)),
                    pca_recon=np.linalg.norm(Z - pca.inverse_transform(pca.transform(Z)), axis=1),
                    knn=knn.kneighbors(Z)[0].mean(axis=1)), p

    Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
    Xc_nf = np.vstack([np.load(os.path.join(DATA, f"{t}_{cfg}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    S0, p0 = scores(Xc)
    thr = {k: np.quantile(v, 1 - DECLINE_CLEAN) for k, v in S0.items()}

    cases = [("clean", Xc, yc)]
    for case in RERENDERED:
        if all(os.path.exists(os.path.join(DATA, f"{t}_native_{case}.npy")) for t in TESTS):
            Xs, ys = shifted_test(case, cfg); cases.append((case, Xs, ys))
    rng = np.random.default_rng(SEED + 61)
    for kind, s in NOISE:
        cases.append((f"{kind}_snr{s}", corr_noise(Xc, rng, s, kind=kind, Xnf=Xc_nf, params=Pc, cen=cen), yc))

    rows = []
    for name, X, y in cases:
        S, p = scores(X); wrong = ((p >= 0.5).astype(int) != y); acc_all = 1 - wrong.mean()
        for k, v in S.items():
            keep = v <= thr[k]
            rows.append(dict(case=name, score=k, accuracy_all=acc_all, coverage=keep.mean(),
                             accuracy_accepted=1 - wrong[keep].mean() if keep.any() else np.nan,
                             accuracy_declined=1 - wrong[~keep].mean() if (~keep).any() else np.nan,
                             auroc_error=roc_auc_score(wrong, v) if 0 < wrong.sum() < len(wrong) else np.nan,
                             auroc_shift=roc_auc_score(np.r_[np.zeros(len(S0[k])), np.ones(len(v))], np.r_[S0[k], v]) if name != "clean" else 0.5))
        print(f"{name:<26} acc {acc_all*100:6.2f}  " + "  ".join(
            f"{k}: keep {rows[-5+i]['coverage']*100:3.0f}% -> {rows[-5+i]['accuracy_accepted']*100:5.2f} (auroc_err {rows[-5+i]['auroc_error']:.2f})"
            for i, k in enumerate(S)), flush=True)

    df = pd.DataFrame(rows)
    L = [f"Can the screen tell when it is outside its simulator? configuration {cfg}, frozen pipeline {best}",
         f"Decline threshold: {DECLINE_CLEAN:.0%} of CLEAN test planets declined, per score; never tuned on a shifted set.", "",
         "For each case: accuracy of all planets, then per score the coverage kept, accepted-set accuracy,",
         "and the AUROC of the score for ranking the screen's own errors (0.5 = uninformative).", ""]
    for name in df.case.unique():
        g = df[df.case == name].set_index("score")
        L.append(f"{name:<26} all {g.accuracy_all.iloc[0]*100:6.2f}%")
        for k in g.index:
            r = g.loc[k]
            L.append(f"    {k:<12} keep {r.coverage*100:5.1f}%  accepted {r.accuracy_accepted*100:6.2f}%  declined {r.accuracy_declined*100:6.2f}%"
                     f"  AUROC(error) {r.auroc_error:.3f}  AUROC(shift) {r.auroc_shift:.3f}")
    # summary: which score gives the best accepted accuracy averaged over shifted cases, and worst-case
    sh = df[df.case != "clean"]
    summ = sh.groupby("score").agg(mean_accepted=("accuracy_accepted", "mean"), min_accepted=("accuracy_accepted", "min"),
                                   mean_coverage=("coverage", "mean"), mean_auroc_error=("auroc_error", "mean")).sort_values("min_accepted", ascending=False)
    L += ["", "Over the shifted cases (clean excluded):", f"{'score':<12}{'mean accepted':>14}{'worst accepted':>15}{'mean coverage':>14}{'mean AUROC(err)':>16}"]
    for k, r in summ.iterrows():
        L.append(f"{k:<12}{r.mean_accepted*100:13.2f}%{r.min_accepted*100:14.2f}%{r.mean_coverage*100:13.1f}%{r.mean_auroc_error:16.3f}")
    frozen_worst = sh.groupby("case").accuracy_all.first().min()
    L += ["", f"Without declining anything the worst shifted-case accuracy is {frozen_worst*100:.2f}%."]
    open(os.path.join(RESULTS, f"{cfg}_trust_detect.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, f"{cfg}_trust_detect.csv"), index=False)
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
