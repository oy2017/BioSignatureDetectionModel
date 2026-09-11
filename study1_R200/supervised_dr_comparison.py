"""
Supervised dimensionality reduction versus variance-ranked PCA.

Addresses the outstanding ablation in Reviewer 1's R1-7, which asks for
"supervised dimensionality reduction techniques such as Linear Discriminant
Analysis or Partial Least Squares" alongside the whitening and leading-component
ablations already run.

The question this answers: PCA orders components by explained variance, which is
an unsupervised criterion. The component ablation showed that variance rank and
discriminative rank are substantially decoupled - the two highest-variance
components classify at chance. If that is true, a *supervised* projection, which
selects directions by their covariance with the label rather than by variance,
should reach comparable accuracy with far fewer components.

Three approaches are compared at matched dimensionality:

    PCA + XGBoost   unsupervised projection, the manuscript's pipeline
    PLS-DA          supervised; maximises covariance with the label
    LDA             supervised; maximises between-class separation

PLS produces a continuous response rather than a probability, so a logistic
regression is fitted on the PLS scores to obtain calibrated probabilities. This
is standard PLS-DA practice and makes the Brier scores comparable across methods.

All projections are fit on the training set only and applied unchanged to the
five held-out test sets.

Usage:
    python supervised_dr_comparison.py
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
TEST_FILE_FMT = "multirex_spectra_H2_test_set_{}.parquet"
N_COMPS = [2, 5, 10, 20, 50, 102]


def load():
    df = pd.read_parquet(TRAIN_FILE)
    fp = re.compile(r"^-?\d+\.\d+$")
    cols = sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)
    X = df[cols].values
    y = (df["biosignature"] == "yes").astype(int).values
    tests = []
    for i in range(1, 6):
        dt = pd.read_parquet(TEST_FILE_FMT.format(i))
        tests.append((dt[cols].values,
                      (dt["biosignature"] == "yes").astype(int).values))
    return X, y, tests


def evaluate(prob_fn, tests_scaled):
    acc, f1s, bri = [], [], []
    for Xt, yt in tests_scaled:
        p = np.clip(prob_fn(Xt), 1e-6, 1 - 1e-6)
        pred = (p > 0.5).astype(int)
        acc.append(accuracy_score(yt, pred))
        f1s.append(f1_score(yt, pred, zero_division=0))
        bri.append(brier_score_loss(yt, p))
    return np.mean(acc), np.std(acc), np.mean(f1s), np.mean(bri)


def main():
    os.makedirs("final_results", exist_ok=True)
    X, y, tests = load()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    tests_s = [(scaler.transform(Xt), yt) for Xt, yt in tests]

    rows = []
    header = f"{'Method':<16} {'n comp':>7} {'accuracy':>17} {'F1':>9} {'Brier':>9}"
    lines = [
        "Supervised dimensionality reduction vs variance-ranked PCA",
        "",
        "PCA selects directions by explained variance (unsupervised); PLS and LDA",
        "select by association with the label (supervised). If variance rank were",
        "the right criterion, PCA would not be beaten at low component counts.",
        "",
        header, "-" * len(header),
    ]
    print("\n".join(lines))

    for n in N_COMPS:
        # --- PCA + XGBoost: the manuscript's unsupervised pipeline ---
        pca = PCA(n_components=n, random_state=SEED).fit(Xs)
        Ptr = pca.transform(Xs)
        Pte = [(pca.transform(Xt), yt) for Xt, yt in tests_s]
        Xtr, ytr = shuffle(Ptr, y, random_state=SEED)
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                            subsample=0.8, eval_metric="logloss",
                            random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        a, sd, f, b = evaluate(lambda Z: xgb.predict_proba(Z)[:, 1], Pte)
        rows.append(("PCA + XGBoost", n, a, sd, f, b))

        # --- PLS-DA: supervised projection, linear classifier ---
        pls = PLSRegression(n_components=n, scale=False).fit(Xs, y.astype(float))
        Ttr = pls.transform(Xs)
        lr = LogisticRegression(max_iter=2000).fit(Ttr, y)
        a2, sd2, f2, b2 = evaluate(
            lambda Z: lr.predict_proba(pls.transform(Z))[:, 1], tests_s)
        rows.append(("PLS-DA (linear)", n, a2, sd2, f2, b2))

        # --- PLS + XGBoost: supervised projection, SAME nonlinear classifier.
        # This is the clean comparison against PCA + XGBoost - only the
        # projection differs, so the classifier is not a confound. Without this
        # arm, PLS-DA losing would only show the task is nonlinear.
        Tte = [(pls.transform(Xt), yt) for Xt, yt in tests_s]
        Xtr2, ytr2 = shuffle(Ttr, y, random_state=SEED)
        xgb2 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                             subsample=0.8, eval_metric="logloss",
                             random_state=SEED, n_jobs=-1).fit(Xtr2, ytr2)
        a3, sd3, f3, b3 = evaluate(lambda Z: xgb2.predict_proba(Z)[:, 1], Tte)
        rows.append(("PLS + XGBoost", n, a3, sd3, f3, b3))

        for name, nn, aa, ss, ff, bb in rows[-3:]:
            print(f"{name:<16} {nn:>7} {aa:>10.2%} ±{ss:>5.2%} {ff:>8.2%} {bb:>9.4f}")

    # --- LDA: a binary problem admits only one discriminant direction ---
    for label, fit_X, fit_tests in [
        ("LDA (raw bins)", Xs, tests_s),
        ("LDA (102 PCs)",
         PCA(n_components=102, random_state=SEED).fit(Xs).transform(Xs), None),
    ]:
        if fit_tests is None:
            p102 = PCA(n_components=102, random_state=SEED).fit(Xs)
            fit_tests = [(p102.transform(Xt), yt) for Xt, yt in tests_s]
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(fit_X, y)
        a, sd, f, b = evaluate(lambda Z: lda.predict_proba(Z)[:, 1], fit_tests)
        rows.append((label, 1, a, sd, f, b))
        print(f"{label:<16} {1:>7} {a:>10.2%} ±{sd:>5.2%} {f:>8.2%} {b:>9.4f}")

    for name, n, a, sd, f, b in rows:
        lines.append(f"{name:<16} {n:>7} {a:>10.2%} ±{sd:>5.2%} {f:>8.2%} {b:>9.4f}")

    # --- the comparison that answers R1-7 ---
    df = pd.DataFrame(rows, columns=["method", "n", "acc", "sd", "f1", "brier"])
    df.to_csv("final_results/H2_supervised_dr.csv", index=False)
    pca_full = df[(df.method == "PCA + XGBoost") & (df.n == 102)].acc.iloc[0]
    lines += ["", "=" * 72,
              "Supervised vs unsupervised projection at matched dimensionality,",
              "holding the classifier fixed (XGBoost). (PCA + XGBoost at 102 = "
              f"{pca_full:.2%})", ""]
    for n in N_COMPS:
        pls_a = df[(df.method == "PLS + XGBoost") & (df.n == n)].acc.iloc[0]
        pca_a = df[(df.method == "PCA + XGBoost") & (df.n == n)].acc.iloc[0]
        lin_a = df[(df.method == "PLS-DA (linear)") & (df.n == n)].acc.iloc[0]
        lines.append(f"  n={n:<4} supervised proj (PLS+XGB) {pls_a:.2%}   "
                     f"unsupervised proj (PCA+XGB) {pca_a:.2%}   "
                     f"diff {pls_a - pca_a:+.2%}   [linear PLS-DA {lin_a:.2%}]")

    out = "\n".join(lines)
    print("\n" + "\n".join(lines[lines.index("=" * 72) - 1:]))
    with open("final_results/H2_supervised_dr.txt", "w") as fh:
        fh.write(out + "\n")
    print("\nWrote final_results/H2_supervised_dr.{txt,csv}")


if __name__ == "__main__":
    main()
