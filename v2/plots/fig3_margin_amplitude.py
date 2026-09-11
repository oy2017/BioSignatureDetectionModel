"""Figure 3: A) accuracy vs margin to the nearest label flip; B) accuracy and the
majority baseline as both thresholds are moved; C) error rate vs feature
amplitude under the absolute-noise convention (and, for contrast, under the
peak-to-peak convention where amplitude is normalized away)."""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import GRID, INK2, SERIES, figure, panel_label, save  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def threshold_table(path):
    rows = []
    for line in open(path):
        m = re.match(r"\s*([+-]\d\.\d\d)\s+(-?\d+\.\d\d)/\s*(-?\d+\.\d\d)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", line)
        if m:
            rows.append(dict(shift=float(m.group(1)), acc=float(m.group(6)), maj=float(m.group(7))))
    return pd.DataFrame(rows)


def main():
    d = pd.read_parquet(os.path.join(RES, "ariel_labels.parquet"))
    has_abs = os.path.exists(os.path.join(RES, "ariel_abs50_labels.parquet"))
    fig, axes = figure(0.36, ncols=3)
    a, b, c = axes
    # A: margin
    edges = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    idx = np.digitize(d["margin"], edges) - 1
    xs, acc, maj = [], [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() < 30:
            continue
        pos = d.loc[m, "y"].mean()
        xs.append(0.5 * (edges[i] + edges[i + 1])); acc.append(d.loc[m, "correct"].mean() * 100); maj.append(max(pos, 1 - pos) * 100)
    a.plot(xs, acc, color=SERIES[0], marker="o", ms=4, label="classifier", markeredgecolor="white", markeredgewidth=1)
    a.plot(xs, maj, color=INK2, marker="s", ms=3, lw=1.0, ls="--", label="majority baseline")
    a.set_xlabel("Margin to label flip (dex)"); a.set_ylabel("Accuracy in bin (%)"); a.set_ylim(28, 100)
    a.legend(loc="lower right", bbox_to_anchor=(1.02, 0.02)); panel_label(a, "A")
    # B: threshold shift
    t = threshold_table(os.path.join(RES, "ariel_labels.txt"))
    b.plot(t["shift"], t["acc"], color=SERIES[0], marker="o", ms=4, label="accuracy", markeredgecolor="white", markeredgewidth=1)
    b.plot(t["shift"], t["maj"], color=INK2, marker="s", ms=3, lw=1.0, ls="--", label="majority baseline")
    b.set_xlabel("Shift of both thresholds (dex)"); b.set_ylabel("Accuracy (%)"); b.set_ylim(40, 100)
    b.legend(loc="center right"); panel_label(b, "B")
    # C: amplitude
    def decile_curve(df):
        q = pd.qcut(df["amplitude"], 10, labels=False)
        return ([df.loc[q == i, "amplitude"].median() for i in range(10)],
                [(1 - df.loc[q == i, "correct"].mean()) * 100 for i in range(10)])
    x0, e0 = decile_curve(d)
    c.plot(x0, e0, color=INK2, marker="s", ms=3, lw=1.0, ls="--", label="peak-to-peak SNR")
    if has_abs:
        da = pd.read_parquet(os.path.join(RES, "ariel_abs50_labels.parquet"))
        x1, e1 = decile_curve(da)
        c.plot(x1, e1, color=SERIES[1], marker="o", ms=4, label="50 ppm floor", markeredgecolor="white", markeredgewidth=1)
    c.set_xscale("log"); c.set_xlabel("Feature amplitude"); c.set_ylabel("Error rate (%)")
    c.set_ylim(0, 55); c.legend(loc="upper right", bbox_to_anchor=(1.02, 0.85)); panel_label(c, "C")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_margin_amplitude.png")


if __name__ == "__main__":
    main()
