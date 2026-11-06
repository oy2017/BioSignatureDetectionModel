"""Figures for the ML4PS 2026 camera-ready version.

figures/transfer.png  A) Spearman correlation of the per-case loss between every pair of
                      the seven pipelines (results/ariel_transfer_matrix.csv, from
                      transfer_mlp.py); B) rank of each shift's loss under six pipelines
                      against the frozen pipeline's rank (the panel B of the submission).
figures/pipeline.png  Block diagram of the setup, for Section 2.

Usage: python plots/fig_camera_ready.py   (from v2/)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import RESULTS  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "paper", "figures")
KEYS = ["norm_xgb", "norm_rf", "norm_mlp", "pca_xgb", "pca_rf", "raw_xgb", "raw_rf"]
LABEL = {"norm_xgb": "normalized, XGBoost", "norm_rf": "normalized, RF", "norm_mlp": "normalized, MLP",
         "pca_xgb": "PCA, XGBoost", "pca_rf": "PCA, RF", "raw_xgb": "raw, XGBoost", "raw_rf": "raw, RF"}
SHORT = {"norm_xgb": "XGB", "norm_rf": "RF", "norm_mlp": "MLP", "pca_xgb": "XGB", "pca_rf": "RF",
         "raw_xgb": "XGB", "raw_rf": "RF"}
COL = {"norm_rf": "#1b9e77", "norm_mlp": "#7570b3", "pca_xgb": "#d95f02", "pca_rf": "#e7298a",
       "raw_xgb": "#e6ab02", "raw_rf": "#d62728"}


def loss(key):
    f = "ariel_shifts.csv" if key == "norm_xgb" else f"ariel_{key}_shifts.csv"
    d = pd.read_csv(os.path.join(RESULTS, f))
    d = d[(d.axis != "clean") & (d.axis != "extrapolation")]
    return d.set_index(d.axis + "|" + d.case)["d_acc"]


def transfer():
    M = pd.read_csv(os.path.join(RESULTS, "ariel_transfer_matrix.csv"), index_col=0).loc[KEYS, KEYS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    im = a.imshow(M.values, cmap="viridis", vmin=0.75, vmax=1.0)
    n = len(KEYS)
    a.set_xticks(range(n)); a.set_yticks(range(n))
    a.set_xticklabels([SHORT[k] for k in KEYS]); a.set_yticklabels([LABEL[k] for k in KEYS])
    for i in range(n):
        for j in range(n):
            if i != j:
                a.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                       color="white" if M.values[i, j] < 0.9 else "black")
    for x in (2.5, 4.5):
        a.axvline(x, color="white", lw=2); a.axhline(x, color="white", lw=2)
    for x, t in ((1, "normalized"), (3.5, "PCA"), (5.5, "raw")):
        a.text(x, -0.9, t, ha="center", va="bottom", fontsize=9)
    a.set_ylim(n - 0.5, -1.3); a.set_xlim(-0.5, n - 0.5)
    a.tick_params(length=0)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.set_label(r"Spearman $\rho$ of the loss over 48 shifts")
    a.set_title("A)", loc="left", fontweight="bold")

    ref = loss("norm_xgb"); cases = ref.index
    rref = (-ref).rank()
    for k in KEYS[1:]:
        lk = loss(k)[cases]; rk = (-lk).rank()
        rho = pd.Series(ref).corr(lk, method="spearman")
        b.scatter(rref, rk, s=18, color=COL[k], label=f"{LABEL[k]} ($\\rho$={rho:.2f})", alpha=0.85, edgecolor="none")
    b.plot([0, 49], [0, 49], color="0.6", lw=0.8, zorder=0)
    b.set_xlabel("Loss rank, frozen pipeline (normalized, XGBoost)"); b.set_ylabel("Loss rank, other pipeline")
    b.legend(fontsize=7.5, loc="upper left", frameon=True); b.set_xlim(0, 49); b.set_ylim(0, 49)
    b.set_title("B)", loc="left", fontweight="bold")
    for s in ("top", "right"):
        a.spines[s].set_visible(False); b.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer.png"), dpi=300)


def box(ax, x, y, w, h, text, color, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.6, edgecolor=color, facecolor="white"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="0.3"))
    if text:
        if x1 == x0:      # vertical arrow: label beside it
            ax.text(x0 + 0.08, (y0 + y1) / 2, text, ha="left", va="center", fontsize=7.5, color="0.3")
        else:             # diagonal arrow: label to its left
            ax.text((x0 + x1) / 2 - 0.15, (y0 + y1) / 2, text, ha="right", va="center", fontsize=7.5, color="0.3")


def pipeline():
    blue, orange, green = "#2b6fd6", "#e8641b", "#1b9e77"
    fig, ax = plt.subplots(figsize=(10.5, 3.3)); ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.3); ax.axis("off")
    w, h, y1, y0 = 2.35, 1.1, 1.95, 0.25
    xs = [0.1, 2.7, 5.3, 7.9]
    box(ax, xs[0], y1, w, h, "Parameter grid\n15 parameters: 9 bulk, 6 gases\nindependent draws, max |r| = 0.07\n18,156 train; 5 × ~1,815 test", blue)
    box(ax, xs[1], y1, w, h, "Forward model\nMultiREx / TauREx 3\nnoise-free, R ≈ 1000", blue)
    box(ax, xs[2], y1, w, h, "Ariel layout + noise\n102 bins: 3 bands, R = 15, 100, 30\nExoRad-shaped noise, SNR 15\nplanet's own seed", blue)
    box(ax, xs[3], y1, w, h, "Representation × model\nnormalized / PCA / raw bins\nXGBoost / random forest / MLP\n5-fold CV, then frozen", blue)
    for i in range(3):
        arrow(ax, xs[i] + w, y1 + h / 2, xs[i + 1], y1 + h / 2)
    box(ax, xs[0], y0, w, h, "Re-rendered, same planets\nRT code; line lists;\nclouds, hazes; stellar spots", orange)
    box(ax, xs[1], y0, w, h, "Injected on clean spectra\nwhite / correlated noise;\ngain ramp; offsets", orange)
    box(ax, xs[2], y0, w, h, "Augmented retraining\nshift randomized over\ntraining set, ⅓ left clean", orange)
    box(ax, xs[3], y0, w, h, "Scoring, paired per planet\nΔ accuracy vs clean 90.33 %\nflips, pred.-positive rate,\nfeature amplitude", green)
    arrow(ax, xs[0] + w / 2, y1, xs[0] + w / 2, y0 + h, "test planets")
    arrow(ax, xs[2] + w / 2, y1, xs[1] + w / 2 + 0.5, y0 + h, "test spectra")
    arrow(ax, xs[0] + w, y0 + h / 2, xs[1], y0 + h / 2)
    arrow(ax, xs[1] + w, y0 + h / 2, xs[2], y0 + h / 2)
    arrow(ax, xs[2] + w, y0 + h / 2, xs[3], y0 + h / 2)
    arrow(ax, xs[3] + w / 2, y1, xs[3] + w / 2, y0 + h, "frozen")
    fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    transfer(); pipeline()
    print("wrote", OUT)
