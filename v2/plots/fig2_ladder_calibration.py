"""Figure 2: A) resolution ladder for three pipelines at the three configurations;
B) reliability curves of the three normalized-feature models at the Ariel
configuration (equal-count bins, Wilson 95 % intervals)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import GRID, INK2, SERIES, figure, panel_label, save  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CFG = [("ariel", "Ariel delivered\n(102 bins)"), ("r100", "uniform R = 100\n(275 bins)"), ("r200", "uniform R = 200\n(550 bins)")]
LINES = [("norm_xgb", "XGBoost, per-spectrum normalized"), ("norm_mlp", "MLP, per-spectrum normalized"),
         ("pca_xgb", "XGBoost, PCA (earlier preprocessing)")]
REL = [("norm_xgb", "XGBoost"), ("norm_rf", "Random Forest"), ("norm_mlp", "MLP")]


def main():
    fig, (a, b) = figure(0.46, ncols=2, gridspec_kw={"width_ratios": [1.05, 1]})
    summ = {}
    for cfg, _ in CFG:
        p = os.path.join(RES, f"{cfg}_summary.json")
        if os.path.exists(p):
            summ[cfg] = json.load(open(p))["models"]
    xs = [i for i, (cfg, _) in enumerate(CFG) if cfg in summ]
    for k, (key, name) in enumerate(LINES):
        vals = [summ[cfg][key]["accuracy"][0] * 100 for cfg, _ in CFG if cfg in summ]
        errs = [summ[cfg][key]["accuracy"][1] * 100 for cfg, _ in CFG if cfg in summ]
        a.errorbar(xs, vals, yerr=errs, color=SERIES[k], marker="o", ms=5, lw=1.6, capsize=2, label=name,
                   markeredgecolor="white", markeredgewidth=1.2)
        for x, v in zip(xs, vals):
            a.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 7 if k != 1 else -12),
                       ha="center", fontsize=7, color=INK2)
    a.set_xticks(xs); a.set_xticklabels([lab for cfg, lab in CFG if cfg in summ])
    a.set_ylabel("Five-set mean accuracy (%)"); a.set_ylim(70, 100); a.legend(loc="lower right"); panel_label(a, "A")
    rel = pd.read_csv(os.path.join(RES, "ariel_reliability.csv"))
    b.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls=":")
    for k, (key, name) in enumerate(REL):
        r = rel[rel["model"] == key]
        b.errorbar(r["mean_pred"], r["observed"], yerr=[r["observed"] - r["lo"], r["hi"] - r["observed"]],
                   color=SERIES[k], marker="o", ms=4.5, lw=1.4, capsize=1.5, label=name,
                   markeredgecolor="white", markeredgewidth=1.0)
    b.set_xlabel("Mean predicted probability"); b.set_ylabel("Observed positive fraction")
    b.set_xlim(0, 1); b.set_ylim(0, 1); b.legend(loc="upper left"); panel_label(b, "B")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_ladder_calibration.png")


if __name__ == "__main__":
    main()
