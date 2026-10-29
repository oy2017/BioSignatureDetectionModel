"""Manuscript figure: CH4 targets lost at the 60 % Tier-2 selection against the retrieval budget f on the mixed
population with a fifth of the planets hazed (e3_budget.py, results/e3_budget_freehaze.csv). Lines: the orderings
that choose which planets the free-haze retrieval re-scores (mean over draws, band = +/- 1 sd for random and the
optical score); dashed: the same optical ordering routed to the existing clear-model retrieval; horizontal: the
classifier alone, the clear-model retrieval of every planet and the free-haze retrieval of every planet."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, MUTED, SERIES, figure, save  # noqa: E402
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(V3, "results")
LINES = [("+ optical", "optical excess", SERIES[0], "o"), ("+ oracle (knows the hazed)", "oracle (knows the hazed)", INK, "s"),
         ("+ mahalanobis", "Mahalanobis", SERIES[2], "^"), ("+ random", "random", MUTED, "D"), ("+ margin", "margin", SERIES[3], "v")]


def main():
    E = pd.read_csv(os.path.join(RES, "e3_budget_freehaze.csv")); E = E[(E.q == 0.2) & (E.target == "CH4")]
    g = E.groupby(["strategy", "f"]).lost.agg(["mean", "std"]).reset_index()
    fig, ax = figure(0.55)
    for key, lab, col, mk in LINES:
        d = g[g.strategy == key].sort_values("f"); ax.plot(100 * d.f, d["mean"], color=col, marker=mk, label=lab, lw=1.4)
        if key in ("+ random", "+ optical"):
            ax.fill_between(100 * d.f, d["mean"] - d["std"], d["mean"] + d["std"], color=col, alpha=0.15, lw=0)
    d = g[g.strategy == "+ optical, clear-model retrieval"].sort_values("f")
    ax.plot(100 * d.f, d["mean"], color=SERIES[1], marker="o", ls="--", label="optical excess → clear-model retrieval", lw=1.4)
    for key, lab, ls in (("classifier alone", "classifier alone, no retrieval", ":"), ("clear-model retrieval of every planet (existing method)", "clear-model retrieval of every planet", "-."),
                         ("freehaze retrieval of every planet", "free-haze retrieval of every planet", (0, (1, 1)))):
        v = E[E.strategy == key].lost.mean(); ax.axhline(v, color=INK2, ls=ls, lw=1.0); ax.text(51, v + 1, lab, fontsize=6.8, color=INK2, ha="right", va="bottom")
    ax.set_xlabel("Retrieval budget f (% of planets re-scored)"); ax.set_ylabel("CH$_4$-bearing targets lost (of 359)")
    ax.set_xlim(8, 52); ax.set_ylim(0, 80); ax.legend(loc="upper right", ncol=2, fontsize=6.8, bbox_to_anchor=(1.0, 0.62))
    save(fig, "fig6_budget.png")


if __name__ == "__main__":
    main()
