"""
Selective principal-component ablation.

Retrains the tree ensembles on restricted PCA index ranges to test where the
label-discriminative signal actually lives. Addresses Reviewer 1's request for
"reconstruction studies after selectively removing principal components."

The preprocessing chain replicates get_data() in run_master_5set_evaluation.py
exactly (StandardScaler -> PCA(102) -> slice -> StandardScaler), but is
self-contained so it does not pull in TensorFlow.

Usage:
    python ablate_pc_ranges.py
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
TEST_FILE_FMT = "multirex_spectra_H2_test_set_{}.parquet"

# (start, end) half-open PCA index ranges.
RANGES = [
    (0, 102),   # baseline: the manuscript's unified feature space
    (2, 102),   # drop the two high-variance components (the key comparison)
    (0, 2),     # the two high-variance components alone
    (1, 102),
    (3, 102),
    (5, 102),
    (10, 102),
    (0, 10),
    (0, 20),
    (0, 50),
    (2, 52),
]


def load_spectra():
    """Return raw spectra, labels, and the shared spectral column list."""
    df_train = pd.read_parquet(TRAIN_FILE)
    float_pattern = re.compile(r"^-?\d+\.\d+$")
    spectral_cols = [
        c for c in df_train.columns
        if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))
    ]

    y_train = (df_train["biosignature"] == "yes").astype(int).values
    X_train_raw = df_train[spectral_cols].values

    test_raw = []
    for i in range(1, 6):
        df_test = pd.read_parquet(TEST_FILE_FMT.format(i))
        y_test = (df_test["biosignature"] == "yes").astype(int).values
        test_raw.append((df_test[spectral_cols].values, y_test))

    return X_train_raw, y_train, test_raw


def fit_pca(X_train_raw, test_raw):
    """Fit the scaler and PCA on training data only; project every split."""
    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)

    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)

    test_pca = [
        (pca.transform(scaler_raw.transform(X_raw)), y)
        for X_raw, y in test_raw
    ]
    return pca, X_train_pca, test_pca


def slice_and_whiten(X_train_pca, test_pca, start, end):
    """Apply the manuscript's post-PCA StandardScaler to a component slice."""
    scaler_pca = StandardScaler()
    X_train = scaler_pca.fit_transform(X_train_pca[:, start:end])
    tests = [(scaler_pca.transform(Xp[:, start:end]), y) for Xp, y in test_pca]
    return X_train, tests


def build(name):
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
            eval_metric="logloss", random_state=SEED, n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=300, min_samples_split=2, min_samples_leaf=2,
        max_depth=None, random_state=SEED, n_jobs=-1,
    )


def evaluate(model, tests):
    accs, f1s = [], []
    for X_test, y_test in tests:
        preds = model.predict(X_test)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, zero_division=0))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def main():
    os.makedirs("final_results", exist_ok=True)

    X_train_raw, y_train, test_raw = load_spectra()
    pca, X_train_pca, test_pca = fit_pca(X_train_raw, test_raw)
    evr = pca.explained_variance_ratio_

    lines = [
        "Selective principal-component ablation",
        f"Training samples: {len(y_train)}   Test sets: {len(test_pca)}"
        f" ({[len(y) for _, y in test_pca]})",
        f"PC0 + PC1 explained variance: {evr[:2].sum():.5%}",
        f"PCs 2-101 explained variance: {evr[2:102].sum():.5%}",
        "",
        f"{'PC range':>12} {'n':>4} {'var %':>9} "
        f"{'XGB acc':>16} {'XGB F1':>16} {'RF acc':>16}",
        "-" * 82,
    ]
    print("\n".join(lines))

    for start, end in RANGES:
        X_train, tests = slice_and_whiten(X_train_pca, test_pca, start, end)
        X_train_s, y_train_s = shuffle(X_train, y_train, random_state=SEED)

        row_var = evr[start:end].sum()
        cells = []
        for name in ("XGBoost", "Random Forest"):
            model = build(name).fit(X_train_s, y_train_s)
            acc, acc_sd, f1, f1_sd = evaluate(model, tests)
            cells.append((acc, acc_sd, f1, f1_sd))

        (xa, xas, xf, xfs), (ra, ras, _, _) = cells
        row = (
            f"[{start:>3}:{end:>3})".rjust(12)
            + f" {end - start:>4} {row_var:>8.4%}"
            + f" {xa:>8.2%} ±{xas:>5.2%}"
            + f" {xf:>8.2%} ±{xfs:>5.2%}"
            + f" {ra:>8.2%} ±{ras:>5.2%}"
        )
        print(row)
        lines.append(row)

    out = "final_results/H2_pc_range_ablation.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
