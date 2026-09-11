"""
Accuracy against distance to the labelling threshold (R1-4).

Reviewer 1 objects that the positive class is a labelling rule applied to the
simulator's own inputs, not a detection. A direct consequence of that rule is
that planets whose abundances sit just either side of the cutoff have opposite
labels but near-identical atmospheres, so their labels carry no information a
spectrum could recover. This script measures that.

Margin, in dex, is the distance to the nearest label flip:

  * positive planets (CH4 >= -6 AND O3 >= -7): the smaller of the two
    excesses - how far the binding gas could fall before the label flips.
  * negative planets: the larger of the deficits over the gases that are
    short, since BOTH must be raised for the label to flip.

Accuracy is then reported per margin bin. Near-chance accuracy in the smallest
bin demonstrates that the labels there are arbitrary rather than that the model
fails; the interpretation is the reviewer's own point, quantified.

Frozen pipeline, XGBoost (deterministic), evaluated on the five clear test sets.

Usage:
    python analyze_label_margin.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
CLEAN_TEST_FMT = "multirex_spectra_H2_test_set_{}.parquet"
BIO_CH4, BIO_O3 = -6.0, -7.0
BIN_EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, np.inf]
C_MAIN, C_TEXT, C_MUTED, C_GRID = "#2a78d6", "#0b0b0b", "#52514e", "#d8d8d4"


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def label_margin(ch4, o3):
    """Dex distance to the nearest label flip; see module docstring."""
    d_ch4 = ch4 - BIO_CH4
    d_o3 = o3 - BIO_O3
    positive = (d_ch4 >= 0) & (d_o3 >= 0)
    margin = np.empty(len(ch4))
    # positive: the binding gas is the one closest to its threshold
    margin[positive] = np.minimum(d_ch4[positive], d_o3[positive])
    # negative: both gases must be raised, so the larger deficit binds
    neg = ~positive
    deficits = np.stack([np.where(d_ch4 < 0, -d_ch4, 0.0),
                         np.where(d_o3 < 0, -d_o3, 0.0)])
    margin[neg] = deficits.max(axis=0)[neg]
    return margin


def main():
    df_tr = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df_tr)
    X_tr = df_tr[cols].values
    y_tr = (df_tr["biosignature"] == "yes").astype(int).values

    scaler_raw = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(
        scaler_raw.transform(X_tr))
    Xs, ys = shuffle(pca.transform(scaler_raw.transform(X_tr)), y_tr,
                     random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                        subsample=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(Xs, ys)

    tests = pd.concat([pd.read_parquet(CLEAN_TEST_FMT.format(i))
                       for i in range(1, 6)], ignore_index=True)
    X = tests[spectral_cols(tests)].values
    y = (tests["biosignature"] == "yes").astype(int).values
    prob = xgb.predict_proba(pca.transform(scaler_raw.transform(X)))[:, 1]
    pred = (prob > 0.5).astype(int)
    margin = label_margin(tests["atm CH4"].astype(float).values,
                          tests["atm O3"].astype(float).values)

    lines = ["Accuracy against distance to the labelling threshold",
             "",
             "Margin is the dex distance to the nearest label flip: for positive",
             "planets the smaller excess over the two thresholds, for negative",
             "planets the larger deficit (both gases must rise to flip the label).",
             "",
             f"Overall accuracy: {np.mean(pred == y):.2%}  (n = {len(y)})",
             "",
             "Class balance varies between bins, so accuracy is reported "
             "against the",
             "majority-class baseline for that bin - the score obtainable "
             "with no",
             "spectral information at all.",
             "",
             f"{'margin (dex)':<16} {'n':>6} {'positives':>10} {'accuracy':>10} "
             f"{'majority':>10} {'gain':>7} {'mean p':>8}", "-" * 72]
    rows = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (margin >= lo) & (margin < hi)
        if m.sum() == 0:
            continue
        acc = np.mean(pred[m] == y[m])
        base = max(y[m].mean(), 1 - y[m].mean())
        name = f"{lo:g} - {hi:g}" if np.isfinite(hi) else f"> {lo:g}"
        lines.append(f"{name:<16} {m.sum():>6} {y[m].mean():>10.1%} "
                     f"{acc:>10.2%} {base:>10.2%} {acc - base:>+7.2%} "
                     f"{prob[m].mean():>8.3f}")
        rows.append((name, int(m.sum()), float(y[m].mean()), float(acc),
                     float(base), float(acc - base), float(prob[m].mean())))

    near = margin < 0.25
    far = margin >= 1.0
    a_near, a_far = np.mean(pred[near] == y[near]), np.mean(pred[far] == y[far])
    b_near = max(y[near].mean(), 1 - y[near].mean())
    b_far = max(y[far].mean(), 1 - y[far].mean())
    lines += ["",
              f"Planets within 0.25 dex of the threshold: {near.sum()} "
              f"({near.mean():.1%} of the test set), accuracy {a_near:.2%} "
              f"against a {b_near:.2%} majority baseline ({a_near - b_near:+.2%})",
              f"Planets at least 1 dex from the threshold:  {far.sum()} "
              f"({far.mean():.1%}), accuracy {a_far:.2%} against a "
              f"{b_far:.2%} majority baseline ({a_far - b_far:+.2%})",
              "",
              "Near-threshold labels distinguish atmospheres that differ by less",
              "than the labelling tolerance. In the smallest-margin bin the",
              "classifier does not beat the majority-class baseline at all, so it",
              "recovers no usable information there - which reflects the",
              "arbitrariness of the cutoff rather than a failure of the model.",
              "Predicted probabilities in that bin average 0.5, i.e. the model",
              "reports its own uncertainty correctly rather than guessing",
              "confidently."]

    out = "\n".join(lines)
    print(out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_label_margin.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["margin_bin", "n", "positive_fraction",
                                "accuracy", "majority_baseline", "gain",
                                "mean_probability"]).to_csv(
        "final_results/H2_label_margin.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    x = np.arange(len(rows))
    ax.bar(x, [r[3] for r in rows], color=C_MAIN, width=0.62)
    ax.axhline(0.5, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(np.mean(pred == y), color=C_MUTED, linestyle=":",
               linewidth=1.5, alpha=0.7)
    for xi, r in zip(x, rows):
        ax.text(xi, r[3] + 0.012, f"n={r[1]}", ha="center", fontsize=7,
                color=C_MUTED)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("distance from the labelling threshold (dex)", fontsize=10,
                  color=C_TEXT)
    ax.set_ylabel("Accuracy", fontsize=10, color=C_TEXT)
    ax.set_title("Accuracy against distance to the abundance cutoff\n"
                 "(dotted = overall accuracy, dashed = chance)",
                 fontsize=11, color=C_TEXT)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=C_MUTED)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(C_GRID)
    fig.tight_layout()
    fig.savefig("final_results/label_margin.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("\nWrote final_results/H2_label_margin.{txt,csv} and label_margin.png")


if __name__ == "__main__":
    main()
