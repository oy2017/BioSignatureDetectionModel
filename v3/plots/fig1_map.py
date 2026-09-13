"""Figure 1: the reliability map at Tier 3. One row per mismatch; the frozen screen's loss, the
randomized screen's loss, and the irreducible part (clean minus the oracle trained at the test
condition). Rows grouped by region: modelled mismatch, mismatch that helps, omitted physics."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, MUTED, SERIES, figure, save  # noqa: E402
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

ROWS = [  # (envelope/randomized case key, label, region)
    ("cloud_1e3Pa", "cloud deck, 10$^3$ Pa", 1), ("haze_3p0e7", "haze, 3×10$^7$ m$^{-3}$", 1), ("haze_2p4e8", "haze, 2.4×10$^8$ m$^{-3}$", 1),
    ("tlse_spots10", "star spots, 10 %", 1), ("tlse_spots20", "star spots, 20 %", 1), ("compound_spots20_haze3e7", "spots 20 % + haze", 1),
    ("white_snr5", "white noise, SNR 5", 1), ("correlated_snr5", "correlated noise, SNR 5", 1),
    ("exotransmit", "other RT code", 1), ("exomol", "other opacity tables", 1),
    ("quenched", "quenched chemistry", 2),
    ("absorbers", "HCN + C$_2$H$_2$ omitted", 3), ("absorbers_quenched", "HCN + C$_2$H$_2$, quenched", 3)]


def main():
    env = pd.read_csv(os.path.join(RES, "ariel_trust_envelope.csv")); env = env[env.score == "ensemble"].set_index("case")
    det = pd.read_csv(os.path.join(RES, "ariel_trust_detect.csv")); det = det[det.score == "margin"].set_index("case")
    orc = pd.read_csv(os.path.join(RES, "ariel_oracle.csv")).set_index("case")
    clean = det.loc["clean", "accuracy_all"]
    fig, ax = figure(0.62)
    y = np.arange(len(ROWS))[::-1]
    for yi, (k, lab, reg) in zip(y, ROWS):
        fr = (clean - det.loc[k, "accuracy_all"]) * 100
        rd = (clean - env.loc[k, "accuracy_all"]) * 100
        irr = orc.loc[k, "irreducible"] if k in orc.index else np.nan
        col = {1: SERIES[0], 2: SERIES[2], 3: SERIES[1]}[reg]
        ax.plot([min(fr, rd), max(fr, rd)], [yi, yi], color=col, lw=1.2, alpha=.6, zorder=1)
        ax.scatter(fr, yi, marker="o", s=34, facecolor="white", edgecolor=col, lw=1.4, zorder=3, label="clean-trained screen" if yi == y[0] else None)
        ax.scatter(rd, yi, marker="o", s=34, color=col, zorder=4, label="randomized screen" if yi == y[0] else None)
        if np.isfinite(irr):
            ax.scatter(irr, yi, marker="|", s=140, color=INK, lw=1.6, zorder=5, label="irreducible (oracle)" if yi == y[0] else None)
    ax.axvline(0, color=INK2, lw=.8)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in ROWS], fontsize=7.5)
    ax.set_xlabel("Accuracy Lost Relative to the Clean-Trained Screen on Clean Spectra (Points)")
    ax.set_xlim(-4, 33); ax.grid(axis="y", visible=False)
    for yy, txt, col in ((y[0] + .55, "modelled mismatch", SERIES[0]), (y[10] + .55, "helps", SERIES[2]), (y[11] + .55, "omitted physics", SERIES[1])):
        ax.text(31.8, yy, txt, ha="right", va="center", fontsize=7.5, color=col, fontweight="bold")
    for yb in (y[9] - .5, y[10] - .5):
        ax.axhline(yb, color=MUTED, lw=.6, ls=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, 1.0), fontsize=7, frameon=False, ncol=3)
    fig.tight_layout(); save(fig, "fig1_map.png")


if __name__ == "__main__":
    main()
