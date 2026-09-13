"""Figure 3: confidence vs distance. (a) error-ranking AUROC of the margin and the Mahalanobis score for
the clean-trained screen on each mismatch; (b) credit of the ensemble and k-NN decline rules against
the clean selective baseline at equal coverage, randomized screen. The omitted-species axes are marked."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, MUTED, SERIES, figure, panel_label, save  # noqa: E402
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CASES = ["cloud_1e3Pa", "haze_3p0e7", "haze_2p4e8", "tlse_spots10", "tlse_spots20", "compound_spots20_haze3e7", "exotransmit", "exomol",
         "white_snr5", "correlated_snr5", "absorbers", "absorbers_quenched"]
LAB = {"cloud_1e3Pa": "cloud", "haze_3p0e7": "haze 3e7", "haze_2p4e8": "haze 2.4e8", "tlse_spots10": "spots 10 %", "tlse_spots20": "spots 20 %",
       "compound_spots20_haze3e7": "spots+haze", "exotransmit": "other code", "exomol": "other tables", "white_snr5": "white SNR 5",
       "correlated_snr5": "corr. SNR 5", "absorbers": "HCN+C$_2$H$_2$", "absorbers_quenched": "HCN+C$_2$H$_2$ quenched"}


def main():
    det = pd.read_csv(os.path.join(RES, "ariel_trust_detect.csv"))
    env = pd.read_csv(os.path.join(RES, "ariel_trust_envelope.csv"))
    fig, (a, b) = figure(0.48, ncols=2)
    x = np.arange(len(CASES))
    for ax, (df, col_y, s1, s2, ylab, title) in zip((a, b), (
            (det, "auroc_error", "margin", "mahalanobis", "error-ranking AUROC", None),
            (env, "credit", "ensemble", "knn", "credit vs clean baseline (points)", None))):
        d = df.set_index(["case", "score"])
        v1 = [d.loc[(c, s1), col_y] * (100 if col_y == "credit" else 1) for c in CASES]
        v2 = [d.loc[(c, s2), col_y] * (100 if col_y == "credit" else 1) for c in CASES]
        ax.scatter(x - .12, v1, color=SERIES[0], s=30, zorder=3, label="confidence (" + s1 + ")")
        ax.scatter(x + .12, v2, color=SERIES[1], s=30, marker="s", zorder=3, label="distance (" + s2 + ")")
        for xi, u, w in zip(x, v1, v2):
            ax.plot([xi - .12, xi + .12], [u, w], color=MUTED, lw=.8, zorder=2)
        ax.axvspan(9.5, 11.5, color=SERIES[1], alpha=.08, lw=0)
        ax.set_xticks(x); ax.set_xticklabels([LAB[c] for c in CASES], rotation=60, ha="right", fontsize=6.8)
        ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    a.axhline(0.5, color=INK2, lw=.8, ls="--"); a.set_ylim(0.3, 1.02); b.axhline(0, color=INK2, lw=.8)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=6.8, ncol=2); b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=6.8, ncol=2)
    panel_label(a, "a"); panel_label(b, "b")
    fig.tight_layout(); save(fig, "fig3_detect.png")


if __name__ == "__main__":
    main()
