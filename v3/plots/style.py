"""Shared figure style for the manuscript: 6.0 in wide, 300 dpi, light surface only,
no in-image titles (captions carry them), text in ink tokens, marks in the
validated categorical palette (slots 1-3 validate all-pairs)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
os.makedirs(OUT, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#d8d8d4", "#ffffff"
WIDTH = 6.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.dpi": 300, "savefig.facecolor": SURFACE, "figure.dpi": 100,
})


def figure(height_ratio=0.5, ncols=1, nrows=1, **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=(WIDTH, WIDTH * height_ratio), **kw)
    return fig, ax


def panel_label(ax, letter):
    ax.text(-0.02, 1.04, f"{letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom", color=INK)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", p)
    return p
