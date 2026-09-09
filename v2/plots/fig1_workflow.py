"""Figure 1: workflow schematic. Pure geometry, no data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.patches as pat  # noqa: E402
from style import INK, INK2, SERIES, figure, save  # noqa: E402

BOXES = [
    # (x, y, w, h, text, colour)
    (0.01, 0.60, 0.225, 0.32, "Independent draws\n20k train, 5 × 2k test\n9 bulk + 6 gas parameters", SERIES[0]),
    (0.26, 0.60, 0.225, 0.32, "Forward model\nMultiREx / TauREx 3\nnoise-free, R ≈ 1000", SERIES[0]),
    (0.51, 0.60, 0.225, 0.32, "Bin + Ariel-shaped noise\nAriel delivered, R = 100,\nR = 200; SNR 15", SERIES[0]),
    (0.76, 0.60, 0.23, 0.32, "Classifiers\nXGBoost, Random Forest,\nMLP; 4 feature sets", SERIES[0]),
    (0.01, 0.08, 0.30, 0.34, "Same planets re-rendered\nExo-Transmit; ExoMol, HITRAN O₃;\nclouds, hazes; stellar spots", SERIES[1]),
    (0.345, 0.08, 0.30, 0.34, "Injected on clean spectra\nwhite, correlated noise;\ngain ramp, offset, colouring", SERIES[1]),
    (0.68, 0.08, 0.31, 0.34, "Frozen pipeline scored\nΔ accuracy, Δ Brier, flips;\nmechanism per axis", SERIES[2]),
]


def box(ax, x, y, w, h, text, colour):
    ax.add_patch(pat.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    linewidth=1.2, edgecolor=colour, facecolor="white"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.0, color=INK, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, shrinkA=0, shrinkB=0))


def main():
    fig, ax = figure(0.42)
    ax.set_axis_off(); ax.grid(False); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for b in BOXES:
        box(ax, *b)
    for x in (0.235, 0.485, 0.735):
        arrow(ax, x + 0.004, 0.76, x + 0.022, 0.76)
    arrow(ax, 0.12, 0.595, 0.12, 0.425)
    arrow(ax, 0.62, 0.595, 0.50, 0.425)
    arrow(ax, 0.875, 0.595, 0.875, 0.425)
    arrow(ax, 0.314, 0.25, 0.341, 0.25)
    arrow(ax, 0.649, 0.25, 0.676, 0.25)
    ax.text(0.128, 0.51, "test planets", ha="left", va="center", fontsize=5.8, color=INK2)
    ax.text(0.575, 0.51, "test spectra", ha="left", va="center", fontsize=5.8, color=INK2)
    ax.text(0.883, 0.51, "frozen", ha="left", va="center", fontsize=5.8, color=INK2)
    ax.text(0.5, 0.985, "In-domain study and resolution ladder", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.text(0.5, 0.005, "Domain-shift map (seven axes, paired per planet)", ha="center", va="bottom", fontsize=7.5, color=INK2)
    save(fig, "fig1_workflow.png")


if __name__ == "__main__":
    main()
