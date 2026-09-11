"""
Sensitivity of the benchmark to where the abundance cutoff is drawn (R1-4).

The positive class is defined by fixed cutoffs (CH4 >= 1e-6, O3 >= 1e-7)
applied to the abundances given to the forward model. Reviewer 1's objection is
that this is a labelling convention rather than a detection; a fair question
that follows is whether the reported performance is an artefact of those
particular cutoff values.

This script shifts both cutoffs together by -0.5, -0.25, 0, +0.25 and +0.5 dex,
relabels the training and test sets, retrains the pipeline from scratch at each
setting, and reports accuracy against the majority-class baseline (which moves,
because shifting the cutoff unbalances the classes that the generation plan
balanced at the original values).

The spectra are untouched: only the labels move. Accuracy that holds up across
the range shows the result does not depend on the specific cutoff chosen;
the majority-baseline column is what makes the comparison fair.

Frozen-pipeline conventions as elsewhere: scaler and PCA are fit on the
training split of each labelling, models are never retrained on test data.

Usage:
    python analyze_threshold_sensitivity.py
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
CLEAN_TEST_FMT = "multirex_spectra_H2_test_set_{}.parquet"
BIO_CH4, BIO_O3 = -6.0, -7.0
SHIFTS = [-0.5, -0.25, 0.0, 0.25, 0.5]


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def relabel(df, shift):
    """Label under cutoffs moved by `shift` dex (both gases together)."""
    return ((df["atm CH4"].astype(float) >= BIO_CH4 + shift)
            & (df["atm O3"].astype(float) >= BIO_O3 + shift)).astype(int).values


def main():
    df_tr = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df_tr)
    X_tr = df_tr[cols].values
    tests = pd.concat([pd.read_parquet(CLEAN_TEST_FMT.format(i))
                       for i in range(1, 6)], ignore_index=True)
    X_te = tests[spectral_cols(tests)].values

    # Preprocessing depends only on the spectra, so it is fit once and shared;
    # only the labels change between settings.
    scaler_raw = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(
        scaler_raw.transform(X_tr))
    P_tr = pca.transform(scaler_raw.transform(X_tr))
    P_te = pca.transform(scaler_raw.transform(X_te))

    lines = ["Sensitivity of performance to the abundance cutoff", "",
             "Both cutoffs are moved together by the stated shift and the data",
             "relabelled; the spectra are unchanged. The majority baseline is the",
             "accuracy obtainable with no spectral information, and moves because",
             "shifting the cutoff unbalances the classes.", "",
             f"{'shift (dex)':>12} {'CH4 / O3 cutoff':>18} {'train pos':>10} "
             f"{'test pos':>9} {'accuracy':>10} {'majority':>10} {'gain':>7} "
             f"{'F1':>7} {'Brier':>8}", "-" * 96]
    rows = []
    for s in SHIFTS:
        y_tr = relabel(df_tr, s)
        y_te = relabel(tests, s)
        Xs, ys = shuffle(P_tr, y_tr, random_state=SEED)
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                              subsample=0.8, eval_metric="logloss",
                              random_state=SEED, n_jobs=-1).fit(Xs, ys)
        p = model.predict_proba(P_te)[:, 1]
        pred = (p > 0.5).astype(int)
        acc = accuracy_score(y_te, pred)
        base = max(y_te.mean(), 1 - y_te.mean())
        f1 = f1_score(y_te, pred, zero_division=0)
        brier = brier_score_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6))
        cutoff = f"{BIO_CH4 + s:.2f} / {BIO_O3 + s:.2f}"
        lines.append(f"{s:>+12.2f} {cutoff:>18} {y_tr.mean():>10.1%} "
                     f"{y_te.mean():>9.1%} {acc:>10.2%} {base:>10.2%} "
                     f"{acc - base:>+7.2%} {f1:>7.3f} {brier:>8.4f}")
        rows.append((s, BIO_CH4 + s, BIO_O3 + s, float(y_tr.mean()),
                     float(y_te.mean()), float(acc), float(base),
                     float(acc - base), float(f1), float(brier)))

    accs = [r[5] for r in rows]
    gains = [r[7] for r in rows]
    lines += ["",
              f"Accuracy across the range: {min(accs):.2%} to {max(accs):.2%} "
              f"(spread {max(accs) - min(accs):.2%} points).",
              f"Gain over the majority baseline: {min(gains):.2%} to "
              f"{max(gains):.2%}.",
              "",
              "Class balance is enforced at the original cutoffs by the",
              "generation plan, so shifted labellings are progressively",
              "unbalanced; the gain column is therefore the meaningful",
              "comparison, not raw accuracy."]

    out = "\n".join(lines)
    print(out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_threshold_sensitivity.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["shift_dex", "ch4_cutoff", "o3_cutoff",
                                "train_positive_frac", "test_positive_frac",
                                "accuracy", "majority_baseline", "gain",
                                "f1", "brier"]).to_csv(
        "final_results/H2_threshold_sensitivity.csv", index=False)
    print("\nWrote final_results/H2_threshold_sensitivity.{txt,csv}")


if __name__ == "__main__":
    main()
