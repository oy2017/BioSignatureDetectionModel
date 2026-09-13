"""Note Figure 2: the consortium classifier under the trust test (1e-4 threshold, mean over the four
classifiers and four molecules). For each mismatch: the loss of the classifier as published (open circle),
of the same design retrained on a randomized grid (filled circle), and the irreducible part, i.e. what a
design retrained at the test condition still loses (tick). Error bars: standard deviation of the loss
across the four molecules."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, MUTED, SERIES, figure, save  # noqa: E402
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ROWS = [("cloud1e3", "cloud deck, 10$^3$ Pa", 0), ("cloud1e2", "cloud deck, 10$^2$ Pa", 0),
        ("haze2e6", "haze, 2×10$^6$ m$^{-3}$", 1), ("haze3e7", "haze, 3×10$^7$ m$^{-3}$", 1), ("haze2p4e8", "haze, 2.4×10$^8$ m$^{-3}$", 1),
        ("spots10", "star spots, 10 %", 1), ("spots20", "star spots, 20 %", 1), ("compound", "spots 20 % + haze 3×10$^7$", 1),
        ("white_x2", "white noise ×2", 2), ("white_x3", "white noise ×3", 2), ("corr_x2", "correlated noise ×2", 2), ("ramp_x2", "gain ramp, 2 noise levels", 2),
        ("exomol", "other opacity database", 3), ("exotransmit", "other radiative-transfer code", 3), ("absorbers", "HCN + C$_2$H$_2$ omitted", 3)]

def main():
    fz = pd.read_csv(os.path.join(RES, "alfnoor_trust_frozen.csv"), dtype={"threshold": str}); fz = fz[fz.threshold == "1e-4"]
    rd = pd.read_csv(os.path.join(RES, "alfnoor_trust_randomized.csv")); rd = rd[rd.variant == "full"]
    ce = pd.read_csv(os.path.join(RES, "alfnoor_trust_ceiling.csv"))
    fm = fz.groupby("molecule").mean(numeric_only=True); rm = rd.groupby("molecule").mean(numeric_only=True)
    cm = ce.groupby(["case", "molecule"]).ceiling.mean().unstack()
    fig, ax = figure(0.78)
    y = np.arange(len(ROWS))[::-1]
    for yi, (k, lab, grp) in zip(y, ROWS):
        lf = (fm["clean"] - fm[k]) * 100; lr = (fm["clean"] - rm[k]) * 100
        col = SERIES[grp]
        ax.plot([min(lf.mean(), lr.mean()), max(lf.mean(), lr.mean())], [yi, yi], color=col, lw=1.1, alpha=.55, zorder=1)
        ax.errorbar(lf.mean(), yi, xerr=lf.std(), fmt="none", ecolor=col, elinewidth=.8, capsize=2)
        ax.scatter(lf.mean(), yi, s=34, facecolor="white", edgecolor=col, linewidth=1.3, zorder=3, label="as published" if yi == y[0] else None)
        ax.scatter(lr.mean(), yi, s=30, color=col, zorder=3, label="randomized training" if yi == y[0] else None)
        if k in cm.index:
            irr = ((fm["clean"] - cm.loc[k]) * 100).mean()
            ax.plot([irr, irr], [yi - .32, yi + .32], color=INK, lw=1.6, zorder=4, label="irreducible (retrained at test condition)" if k == "cloud1e2" else None)
    ax.axvline(0, color=INK2, lw=.6)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab, _ in ROWS]); ax.tick_params(axis="y", length=0); ax.set_ylim(-.7, len(ROWS) - .3); ax.set_xlim(-1, 25)
    ax.set_xlabel("Accuracy lost (points, relative to the classifier's clean accuracy)")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right")
    save(fig, "note_fig2_consortium_map.png")

if __name__ == "__main__":
    main()
