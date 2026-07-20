"""
Per-component discriminative power of the PCA feature space.

For every principal component this measures how much label information it
carries, independently of how much variance it explains:

  * single-feature AUC  - rank-based, so completely invariant to feature scale.
    0.5 means the component orders the two classes no better than chance.
    Reported as max(auc, 1 - auc) since the direction of the relationship is
    irrelevant to whether information is present.
  * mutual information  - captures non-monotonic dependence that AUC misses.
  * Pearson r           - included only to reproduce the statistic currently
    quoted in the manuscript, for comparison.

Addresses Reviewer 1's request for explicit quantification of per-component
information content, and replaces the label-correlation evidence in Section 4.2.

Preprocessing replicates get_data() in run_master_5set_evaluation.py
(StandardScaler -> PCA(102)); the post-PCA whitening is omitted because every
statistic here is invariant to per-feature affine rescaling.

Usage:
    python analyze_pc_discriminative_power.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
N_COMPONENTS = 102
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
TEST_FILE_FMT = "multirex_spectra_H2_test_set_{}.parquet"

# Reference palette, slots 1 and 2 in documented order (light mode).
C_SERIES_1 = "#2a78d6"
C_SERIES_2 = "#008300"
C_TEXT = "#0b0b0b"
C_TEXT_MUTED = "#52514e"
C_GRID = "#d8d8d4"


def load_and_project():
    """Fit scaler + PCA on training data only; project train and pooled test."""
    df_train = pd.read_parquet(TRAIN_FILE)
    float_pattern = re.compile(r"^-?\d+\.\d+$")
    spectral_cols = [
        c for c in df_train.columns
        if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))
    ]

    y_train = (df_train["biosignature"] == "yes").astype(int).values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[spectral_cols].values)

    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)

    X_test_parts, y_test_parts = [], []
    for i in range(1, 6):
        df_test = pd.read_parquet(TEST_FILE_FMT.format(i))
        X_test_parts.append(
            pca.transform(scaler.transform(df_test[spectral_cols].values))
        )
        y_test_parts.append((df_test["biosignature"] == "yes").astype(int).values)

    return pca, X_train_pca, y_train, np.vstack(X_test_parts), np.concatenate(y_test_parts)


def per_component_stats(pca, X_train, y_train, X_test, y_test):
    rows = []
    mi_train = mutual_info_classif(X_train, y_train, random_state=SEED)

    for k in range(N_COMPONENTS):
        auc_tr = roc_auc_score(y_train, X_train[:, k])
        auc_te = roc_auc_score(y_test, X_test[:, k])
        r, _ = pearsonr(X_train[:, k], y_train)
        rows.append({
            "component": k,
            "explained_variance_ratio": pca.explained_variance_ratio_[k],
            "auc_train": max(auc_tr, 1 - auc_tr),
            "auc_test": max(auc_te, 1 - auc_te),
            "mutual_information": mi_train[k],
            "pearson_r": r,
        })
    return pd.DataFrame(rows)


def make_figure(df, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True, dpi=300)
    x = df["component"].values

    def style(ax, ylabel):
        ax.set_ylabel(ylabel, fontsize=10, color=C_TEXT)
        ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors=C_TEXT_MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(C_GRID)
        # Shade the two components that dominate the variance.
        ax.axvspan(-0.5, 1.5, color=C_TEXT_MUTED, alpha=0.10, linewidth=0)

    # Panel 1 - variance.
    ax = axes[0]
    ax.plot(x, df["explained_variance_ratio"], color=C_SERIES_1, linewidth=1.8)
    ax.set_yscale("log")
    style(ax, "Explained variance ratio")
    ax.annotate(
        f"PC0–PC1: {df['explained_variance_ratio'][:2].sum():.2%} of variance",
        xy=(1.5, df["explained_variance_ratio"][0]), xytext=(12, -6),
        textcoords="offset points", fontsize=9, color=C_TEXT_MUTED,
    )

    # Panel 2 - AUC.
    ax = axes[1]
    ax.axhline(0.5, color=C_TEXT_MUTED, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(x, df["auc_train"], color=C_SERIES_1, linewidth=1.8, label="Training set")
    ax.plot(x, df["auc_test"], color=C_SERIES_2, linewidth=1.8, alpha=0.85,
            label="Pooled test sets")
    style(ax, "Single-feature AUC")
    ax.set_ylim(0.487, None)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=C_TEXT)
    ax.annotate("chance (AUC = 0.5)", xy=(3, 0.5), xytext=(0, -12),
                textcoords="offset points", fontsize=8, color=C_TEXT_MUTED)

    # Panel 3 - mutual information.
    ax = axes[2]
    ax.plot(x, df["mutual_information"], color=C_SERIES_1, linewidth=1.8)
    style(ax, "Mutual information (nats)")
    ax.set_xlabel("Principal component index", fontsize=10, color=C_TEXT)
    ax.set_xlim(-1, N_COMPONENTS)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs("final_results", exist_ok=True)

    pca, X_train, y_train, X_test, y_test = load_and_project()
    df = per_component_stats(pca, X_train, y_train, X_test, y_test)

    csv_path = "final_results/H2_pc_discriminative_power.csv"
    df.to_csv(csv_path, index=False)

    lines = [
        "Per-component discriminative power of the PCA feature space",
        f"Training samples: {len(y_train)}   Pooled test samples: {len(y_test)}",
        "",
        "Leading components (the ones carrying the variance):",
        f"{'PC':>4} {'var %':>10} {'AUC train':>11} {'AUC test':>10} "
        f"{'MI':>8} {'Pearson r':>11}",
    ]
    for k in range(6):
        row = df.iloc[k]
        lines.append(
            f"{k:>4} {row['explained_variance_ratio']:>9.4%}"
            f" {row['auc_train']:>11.4f} {row['auc_test']:>10.4f}"
            f" {row['mutual_information']:>8.4f} {row['pearson_r']:>11.4f}"
        )

    top = df.sort_values("auc_train", ascending=False).head(15)
    lines += ["", "Top 15 components by single-feature AUC (training set):",
              f"{'PC':>4} {'var %':>10} {'AUC train':>11} {'AUC test':>10} {'MI':>8}"]
    for _, row in top.iterrows():
        lines.append(
            f"{int(row['component']):>4} {row['explained_variance_ratio']:>9.4%}"
            f" {row['auc_train']:>11.4f} {row['auc_test']:>10.4f}"
            f" {row['mutual_information']:>8.4f}"
        )

    top20_auc = set(df.sort_values("auc_train", ascending=False).head(20)["component"])
    top20_var = set(range(20))  # components are already ordered by variance
    lines += [
        "",
        f"Of the 20 most discriminative components, "
        f"{len(top20_auc - top20_var)} fall outside the 20 highest-variance components.",
        f"Mean AUC of PC0–PC1: {df['auc_train'][:2].mean():.4f}",
        f"Mean AUC of PC2–PC101: {df['auc_train'][2:].mean():.4f}",
        f"Max AUC anywhere: {df['auc_train'].max():.4f} (PC{int(df['auc_train'].idxmax())})",
    ]

    report = "\n".join(lines)
    print(report)
    with open("final_results/H2_pc_discriminative_power.txt", "w") as fh:
        fh.write(report + "\n")

    fig_path = "final_results/pc_discriminative_power.png"
    make_figure(df, fig_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote final_results/H2_pc_discriminative_power.txt")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
