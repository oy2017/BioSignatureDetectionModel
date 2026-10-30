"""Manuscript figure: CH4 targets lost at the 60 % Tier-2 selection against the retrieval budget f on the mixed
population with a fifth of the planets hazed (e3_budget.py, results/e3_budget_freehaze.csv). Lines: the orderings
that choose which planets the free-haze retrieval re-scores (mean over draws, band = +/- 1 sd for random and the
optical score); dashed: the same optical ordering routed to the clear-model retrieval; horizontal: the
classifier alone, the clear-model retrieval of every planet, and that retrieval re-run with the haze density free where its evidence fails."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, MUTED, SERIES, figure, save  # noqa: E402
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(V3, "results")
LINES = [("+ optical", "Optical Excess", SERIES[0], "o"), ("+ oracle (knows the hazed)", "Oracle (Knows the Hazed)", INK, "s"),
         ("+ mahalanobis", "Mahalanobis", SERIES[2], "^"), ("+ random", "Random", MUTED, "D"), ("+ margin", "Margin", SERIES[3], "v")]


def main():
    E = pd.read_csv(os.path.join(RES, "e3_budget_freehaze.csv")); E = E[(E.q == 0.2) & (E.target == "CH4")]
    g = E.groupby(["strategy", "f"]).lost.agg(["mean", "std"]).reset_index()
    fig, ax = figure(0.55)
    for key, lab, col, mk in LINES:
        d = g[g.strategy == key].sort_values("f"); ax.plot(100 * d.f, d["mean"], color=col, marker=mk, label=lab, lw=1.4)
        if key in ("+ random", "+ optical"):
            ax.fill_between(100 * d.f, d["mean"] - d["std"], d["mean"] + d["std"], color=col, alpha=0.15, lw=0)
    d = g[g.strategy == "+ optical, clear-model retrieval"].sort_values("f")
    ax.plot(100 * d.f, d["mean"], color=SERIES[1], marker="o", ls="--", label="Optical Excess → Clear-Model Retrieval", lw=1.4)
    for key, lab, ls, xt, ha, dy in (("classifier alone", "Classifier Alone, No Retrieval", ":", 51, "right", 0.8), ("clear-model retrieval of every planet (existing method)", "Clear-Model Retrieval of Every Planet", "-.", 51, "right", 3.2),
                                     ("freehaze retrieval of every planet", "Clear-Model Retrieval, Re-Run With Haze Free Where the Evidence Fails", (0, (1, 1)), 8.6, "left", 0.9)):
        v = E[E.strategy == key].lost.mean(); ax.axhline(v, color=INK2, ls=ls, lw=1.0); ax.text(xt, v + dy, lab, fontsize=6.4, color=INK2, ha=ha, va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=.4, alpha=.85))
    ax.set_xlabel("Retrieval Budget f (% of Planets Re-Scored)"); ax.set_ylabel("CH$_4$-Bearing Targets Lost (of 359)")
    ax.set_xlim(8, 52); ax.set_ylim(0, 80); ax.legend(loc="upper right", ncol=2, fontsize=6.8, bbox_to_anchor=(1.0, 0.62))
    save(fig, "fig6_budget.png")


if __name__ == "__main__":
    main()
