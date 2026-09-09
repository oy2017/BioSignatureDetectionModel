"""Compare retrieval-derived labels with the frozen classifier on the identical planets.

Reads the retrieval output (retrieve_labels_balanced.py) and, for exactly the same
pooled-index planets, runs the frozen headline pipeline (StandardScaler -> PCA(102)
-> StandardScaler -> tuned XGBoost, trained on the training set only) to get a
predicted label and a predicted probability. Reports, per class:

  - retrieval-derived label vs injected threshold label
  - classifier label vs injected threshold label   (head-to-head on the same planets)
  - classifier label vs retrieval label             (agreement between the two methods)
  - correlation of the classifier probability with the retrieval posterior P+  (the
    probabilistic-label view)
  - injection-recovery: how often the true abundance lands in the retrieval's 68% CI

Works on a partial CSV too, so it can be run for an interim peek.
Run AFTER (or during) retrieve_labels_balanced.py. No arguments.
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

from retrieve_labels_balanced import load_pooled, spectral_cols, CH4_THR, O3_THR

SEED = 42
RET = "final_results/H2_retrieval_balanced.csv"
OUT = "final_results/H2_retrieval_vs_classifier.txt"


def frozen_classifier(df_train, cols):
    """Train the tuned headline pipeline on the training set; return a predict fn."""
    Xtr = df_train[cols].values
    ytr = (df_train["biosignature"] == "yes").astype(int).values
    keep = (Xtr <= 1.0).all(axis=1)
    Xtr, ytr = Xtr[keep], ytr[keep]
    sr = StandardScaler().fit(Xtr)
    pca = PCA(n_components=102, random_state=SEED).fit(sr.transform(Xtr))
    sp = StandardScaler().fit(pca.transform(sr.transform(Xtr)))
    Ztr = sp.transform(pca.transform(sr.transform(Xtr)))
    Ztr, ytr = shuffle(Ztr, ytr, random_state=SEED)
    clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                        eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(Ztr, ytr)

    def predict(Xraw):
        Z = sp.transform(pca.transform(sr.transform(Xraw)))
        return clf.predict(Z), clf.predict_proba(Z)[:, 1]
    return predict


def rate(mask, cond):
    m = mask.values if hasattr(mask, "values") else mask
    return 100 * cond[m].mean() if m.sum() else float("nan")


def main():
    ret = pd.read_csv(RET)
    n = len(ret)
    df = load_pooled()
    df_train = pd.read_parquet("multirex_spectra_H2_train.parquet")
    cols = spectral_cols(df_train)

    # classifier on the identical planets
    Xsel = df.iloc[ret["pool_idx"].values][cols].values
    predict = frozen_classifier(df_train, cols)
    clf_label, clf_prob = predict(Xsel)
    ret["clf_label"] = clf_label
    ret["clf_prob"] = clf_prob

    truth = ret["true_label"].values
    pos = truth == 1
    neg = truth == 0
    L = []
    L.append(f"Retrieval vs. classifier on {n} identical planets "
             f"({int(pos.sum())} positive, {int(neg.sum())} negative; seed {SEED}, pooled test sets)")
    L.append("Retrieval: best-case (own forward model, nuisance params fixed at truth, clear-sky,")
    L.append("SNR 15, npoints=100) -> the label disagreement below is a conservative lower bound.")
    L.append("")

    def block(name, pred):
        acc = 100 * (pred == truth).mean()
        rec = rate(pos, pred == truth)
        spec = rate(neg, pred == truth)
        L.append(f"{name}: {acc:.0f}% agree with injected label   "
                 f"(positives {rec:.0f}%, negatives {spec:.0f}%)")

    block("Retrieval-derived label vs truth", ret["ret_label"].values)
    block("Classifier label        vs truth", ret["clf_label"].values)
    agree = 100 * (ret["clf_label"].values == ret["ret_label"].values).mean()
    L.append(f"Classifier vs retrieval label: {agree:.0f}% agree with each other")
    L.append("")

    # probabilistic-label view
    if ret["p_pos"].nunique() > 1 and ret["clf_prob"].nunique() > 1:
        r, p = pearsonr(ret["clf_prob"], ret["p_pos"])
        L.append(f"Probabilistic labels: classifier probability vs retrieval posterior P+  "
                 f"Pearson r = {r:.2f} (p = {p:.1e})")
    L.append("")

    # injection-recovery coverage on the retrieval
    for g in ["ch4", "o3"]:
        cov = 100 * ((ret[f"{g}_inj"] >= ret[f"{g}_lo"]) & (ret[f"{g}_inj"] <= ret[f"{g}_hi"])).mean()
        mede = np.median(np.abs(ret[f"{g}_med"] - ret[f"{g}_inj"]))
        L.append(f"{g.upper()} recovery: median |err| {mede:.2f} dex, truth in 68% CI {cov:.0f}% (ideal ~68%)")

    report = "\n".join(L)
    print(report)
    with open(OUT, "w") as fh:
        fh.write(report + "\n")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
