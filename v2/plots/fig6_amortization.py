"""Figure: a cheap method and an expensive one under an ingredient neither models.

The point is not that both degrade. It is that the rate at which they disagree
barely moves while the agreements stop being right, so concurrence between them
carries no information about whether either is correct.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, SERIES, figure, save  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    d = pd.read_csv(os.path.join(RES, "ariel_amortization.csv")).set_index("condition")
    labels = {"clean": "Clean spectra", "haze": "Haze, $3\\times10^{7}$ m$^{-3}$"}

    both_right, both_wrong, disagree = [], [], []
    for c in ("clean", "haze"):
        r = d.loc[c]
        agree = 100 * r["agreement"]
        br = 100 * r["both_correct"]
        both_right.append(br)
        both_wrong.append(agree - br)
        disagree.append(100 - agree)

    fig, ax = figure(0.32)
    y = np.arange(2)[::-1]
    h = 0.55
    segs = [("Both correct", both_right, SERIES[2]),
            ("Both wrong, and they agree", both_wrong, SERIES[1]),
            ("They disagree", disagree, "#c9c9c4")]
    left = np.zeros(2)
    for name, vals, col in segs:
        ax.barh(y, vals, left=left, height=h, color=col, label=name,
                edgecolor="white", linewidth=1.1)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 6:
                ax.text(l + v / 2, y[yi], f"{v:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="white" if col != "#c9c9c4" else INK2,
                        fontweight="bold")
        left = left + np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels([labels["clean"], labels["haze"]], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the 80 retrieved planets (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8,
              frameon=False, handletextpad=0.5, columnspacing=1.4)
    fig.tight_layout()
    save(fig, "fig6_amortization.png")


if __name__ == "__main__":
    main()
