"""Manuscript figure: balanced credit on the two held-out model grids through the real masks, with 95 % bootstrap intervals over planets (results/goyal_test, results/round3)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from style import INK2, SERIES, save  # noqa: E402
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = ["published", "AIRS-only", "haze+deck-retrained", "the method", "matched+amp", "C+B-mlp"]; LAB = ["Re-Implemented Classifier", "AIRS-Only Input", "Haze+Deck-Retrained", "Routing Method", "Per-Reduction", "Matched Ensemble"]; GR = [("local", "Local Condensation"), ("crossfield", "Photochemical")]; COL = ["#8a8a8a", SERIES[0], SERIES[2], SERIES[3], SERIES[4], "#111111"]; BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")


def read(tag):
    g = os.path.join(V3, "results", "goyal_test"); V = pd.concat([pd.read_csv(os.path.join(g, f"votes_{k}_{tag}.csv.gz")) for k in ("frozen", "matched")] + [pd.read_csv(os.path.join(V3, "results", "round3", f"votes_C+B-mlp_{tag}.csv.gz")).drop(columns="prob")]); V = V[V.design.isin(D)]; Y = pd.read_csv(os.path.join(g, f"truth_{tag}.csv")).melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in ("CH4", "H2O", "CO2", "NH3")], var_name="molecule", value_name="v"); Y["molecule"] = Y.molecule.str[2:]; A = V.merge(Y, on=["planet", "row", "molecule"]); A["y"] = A.v > -4; A["c"] = np.where(A.y, A.present, A.absent) + 0.5 * (1 - A.present - A.absent); A["u"] = A.planet.map(BASE); S = A.groupby(["precision", "design", "u", "y"]).c.agg(["sum", "count"]); us = sorted(A.u.unique()); ix = np.random.default_rng(405).integers(0, len(us), (4000, len(us))); o = {}
    for p in ("JWST", "Tier-1"):
        for d in D:
            q = S.loc[(p, d)].unstack("y").reindex(us).fillna(0); a, b, c_, e = (q["sum"][True].to_numpy(), q["count"][True].to_numpy(), q["sum"][False].to_numpy(), q["count"][False].to_numpy()); F = lambda i: 0.5 * (a[i].sum(-1) / b[i].sum(-1) + c_[i].sum(-1) / e[i].sum(-1)); o[(p, d)] = (F(np.arange(len(us))), *np.percentile(F(ix), [2.5, 97.5]))
    return o


def main():
    R = {t: read(t) for t, _ in GR}; fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for a, prec, title in zip(ax, ("JWST", "Tier-1"), ("(a) JWST Precision", "(b) Tier-1 Noise")):
        for j, (d, lab) in enumerate(zip(D, LAB)):
            xs = np.arange(2) + (j - 2.5) * 0.13; v = np.array([R[t][(prec, d)] for t, _ in GR]); a.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4.5, capsize=2.5, lw=1.1, color=COL[j], label=lab)
        a.axhline(0.5, color=INK2, ls=":", lw=1); a.set_xticks(np.arange(2)); a.set_xticklabels([g for _, g in GR]); a.set_title(title, fontsize=9); a.set_ylim(0.47, 0.84); a.set_xlim(-0.55, 1.55)
    ax[0].set_ylabel("Balanced Credit"); ax[1].legend(fontsize=6.3, loc="upper left", ncol=2, frameon=False); fig.tight_layout(); save(fig, "fig7_grids.png")


if __name__ == "__main__":
    main()
