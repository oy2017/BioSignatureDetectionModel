"""Precision-recall curve and candidate operating points for the XGBoost triage filter.

Section 4.5 argues that the decision threshold would be moved in deployment --
lowered for a recall-first pass over the Tier 3 catalogue, raised when
down-selecting for Tier 4 or retrieval. This makes that argument quantitative:
it reports the precision cost of each operating point, in domain and under one
domain shift.

Rewritten from the original version, which predated the five-set validation
framework: it read a single test file and used untuned hyperparameters. This
version uses the same pooled five test sets, the same cleaning, and the same
tuned configuration as Section 4.1 and Figure 6, so the numbers are comparable
with the rest of the paper.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILES = [f"multirex_spectra_{FILL_GAS}_test_set_{i}.parquet" for i in range(1, 6)]
SHIFT_FILE = f"multirex_spectra_{FILL_GAS}_paired_cloudy_1e4Pa.parquet"
OUT = "final_results/plots/pr_curve_xgboost.png"

FLOAT_COL = re.compile(r"^-?\d+\.\d+$")


def label(df):
    return df["biosignature"].apply(lambda x: 1 if x == "yes" else 0).values


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print("--- Loading data ---")
    df_train = pd.read_parquet(TRAIN_FILE)
    df_test = pd.concat([pd.read_parquet(f) for f in TEST_FILES], ignore_index=True)
    cols = [c for c in df_train.columns
            if isinstance(c, float) or (isinstance(c, str) and FLOAT_COL.match(c))]

    # Same physically-impossible-depth cut as the main pipeline.
    df_train = df_train[(df_train[cols].values <= 1.0).all(axis=1)].reset_index(drop=True)
    df_test = df_test[(df_test[cols].values <= 1.0).all(axis=1)].reset_index(drop=True)
    y_train, y_test = label(df_train), label(df_test)
    print(f"    train {len(df_train)}   test {len(df_test)}   positive rate {y_test.mean():.4f}")

    print("--- Preprocessing (frozen on training data) ---")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(df_train[cols].values)
    pca = PCA(n_components=102, random_state=SEED)
    Xtr = pca.fit_transform(Xtr)
    post = StandardScaler()
    Xtr = post.fit_transform(Xtr)

    def project(df):
        return post.transform(pca.transform(scaler.transform(df[cols].values)))

    Xte = project(df_test)

    print("--- Training XGBoost (tuned configuration of Table 2) ---")
    Xtr, y_train = shuffle(Xtr, y_train, random_state=SEED)
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                          random_state=SEED, n_jobs=-1, eval_metric="logloss")
    model.fit(Xtr, y_train)

    score = model.predict_proba(Xte)[:, 1]
    ap = average_precision_score(y_test, score)
    print(f"    average precision {ap:.4f}   accuracy at 0.5 {100*((score>=0.5)==y_test).mean():.2f}%")

    # Domain-shifted comparison: same planets, 1e4 Pa grey deck.
    shift = None
    if os.path.exists(SHIFT_FILE):
        df_s = pd.read_parquet(SHIFT_FILE)
        df_s = df_s[(df_s[cols].values <= 1.0).all(axis=1)].reset_index(drop=True)
        y_s = label(df_s)
        sc_s = model.predict_proba(project(df_s))[:, 1]
        shift = (y_s, sc_s, average_precision_score(y_s, sc_s))
        print(f"    cloud deck 1e4 Pa: average precision {shift[2]:.4f}")

    def operating_points(y, s, targets=(0.90, 0.95, 0.99)):
        """Precision and threshold at the highest-precision point meeting each recall target."""
        prec, rec, thr = precision_recall_curve(y, s)
        out = []
        for t in targets:
            ok = np.where(rec[:-1] >= t)[0]
            i = ok[-1] if len(ok) else 0
            out.append((t, thr[i], prec[i], rec[i]))
        return out

    print("\n--- Candidate operating points (in domain) ---")
    print(f"{'target recall':>14} {'threshold':>11} {'precision':>11} {'recall':>9}")
    for t, th, p, r in operating_points(y_test, score):
        print(f"{t:>14.2f} {th:>11.3f} {100*p:>10.1f}% {100*r:>8.1f}%")
    d = (score >= 0.5)
    print(f"{'default 0.5':>14} {0.5:>11.3f} "
          f"{100*(y_test[d].mean() if d.sum() else 0):>10.1f}% "
          f"{100*(d[y_test==1].mean()):>8.1f}%")

    if shift:
        print("\n--- Same targets under a 1e4 Pa cloud deck ---")
        for t, th, p, r in operating_points(shift[0], shift[1]):
            print(f"{t:>14.2f} {th:>11.3f} {100*p:>10.1f}% {100*r:>8.1f}%")

    print("\n--- Plotting ---")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    prec, rec, _ = precision_recall_curve(y_test, score)
    ax.plot(rec, prec, color="#1f4e79", lw=2.2, label=f"Clear (AP = {ap:.3f})")
    if shift:
        ps, rs, _ = precision_recall_curve(shift[0], shift[1])
        ax.plot(rs, ps, color="#c0392b", lw=2.0, ls="--",
                label=f"Cloud deck 10$^4$ Pa (AP = {shift[2]:.3f})")
    ax.axhline(y_test.mean(), color="grey", lw=1, ls=":",
               label=f"No-skill baseline ({y_test.mean():.3f})")

    # Annotations sit in the empty lower-left region so they do not overlap either curve.
    for (t, th, p, r), ytext in zip(operating_points(y_test, score, (0.95, 0.99)), (0.54, 0.40)):
        ax.plot(r, p, "o", ms=8, color="#1f4e79", zorder=5)
        ax.annotate(f"{t:.0%} recall at threshold {th:.2f}\nprecision {p:.0%}",
                    xy=(r, p), xycoords="data",
                    xytext=(0.04, ytext), textcoords="axes fraction",
                    fontsize=9.5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="#1f4e79",
                                    connectionstyle="arc3,rad=-0.15"))

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.35, 1.02)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {OUT}")


if __name__ == "__main__":
    main()
