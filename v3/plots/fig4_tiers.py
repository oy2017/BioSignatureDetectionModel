"""Figure 4: Tier 3 vs Tier 1 for the same mismatches (error bars: standard deviation across the five test sets) — clean-trained screen, randomized screen, and
the ceiling from a screen trained at the test condition; right panel: the screen on Ariel's known
targets under the mission's noise definition, by tier and host type."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, MUTED, SERIES, figure, panel_label, save  # noqa: E402
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CASES = [("clean", "clean"), ("haze_3p0e7", "haze 3e7"), ("tlse_spots20", "spots 20 %"), ("compound_spots20_haze3e7", "spots+haze"),
         ("white_snr5", "white SNR 5"), ("exomol", "other tables"), ("absorbers", "HCN+C$_2$H$_2$")]


def main():
    fig, (a, b) = figure(0.52, ncols=2, gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(CASES)); w = .36
    for j, (cfg, tag, col) in enumerate((("ariel", "Tier 3 (102 points)", SERIES[0]), ("tier1", "Tier 1 (7 points)", SERIES[1]))):
        rnd = pd.read_csv(os.path.join(RES, f"{cfg}_trust_randomized.csv")).set_index("case")
        env = pd.read_csv(os.path.join(RES, f"{cfg}_trust_envelope.csv")); env = env[env.score == "ensemble"].set_index("case")
        orc = pd.read_csv(os.path.join(RES, f"{cfg}_oracle.csv")).set_index("case")
        det = pd.read_csv(os.path.join(RES, f"{cfg}_trust_detect.csv")) if os.path.exists(os.path.join(RES, f"{cfg}_trust_detect.csv")) else None
        pp = os.path.join(RES, f"{cfg}_trust_randomized_persplit.csv"); po = os.path.join(RES, f"{cfg}_oracle_persplit.csv")
        sd_f = sd_r = sd_o = {}
        if os.path.exists(pp):
            g = pd.read_csv(pp).groupby("case"); sd_f = (g.frozen.std() * 100).to_dict(); sd_r = (g.full.std() * 100).to_dict()
        if os.path.exists(po):
            sd_o = (pd.read_csv(po).groupby("case").oracle.std() * 100).to_dict()
        for i, (c, lab) in enumerate(CASES):
            xi = x[i] + (j - .5) * w
            if c in rnd.index: fr = rnd.loc[c, "frozen"] * 100
            elif det is not None: fr = det[det.score == "margin"].set_index("case").loc[c, "accuracy_all"] * 100
            else: fr = env.loc[c, "accuracy_all"] * 100 if c in env.index else np.nan
            rd = env.loc[c, "accuracy_all"] * 100 if c in env.index else np.nan
            a.bar(xi, fr, w * .92, color=col, alpha=.35, edgecolor=col, lw=.8, yerr=sd_f.get(c, 0), error_kw=dict(elinewidth=.8, capsize=2, ecolor=INK2), label=f"{tag}: clean-trained" if i == 0 else None)
            a.errorbar(xi, rd, yerr=sd_r.get(c, 0), fmt="none", ecolor=col, elinewidth=.8, capsize=2, zorder=3)
            a.scatter(xi, rd, color=col, s=26, zorder=4, label=f"{tag}: randomized" if i == 0 else None)
            if c in orc.index:
                a.errorbar(xi, orc.loc[c, "oracle"] * 100, yerr=sd_o.get(c, 0), fmt="none", ecolor=INK, elinewidth=.8, capsize=2, zorder=4)
                a.scatter(xi, orc.loc[c, "oracle"] * 100, marker="_", s=160, color=INK, lw=1.5, zorder=5, label="ceiling (trained at test condition)" if (i == 1 and j == 0) else None)
    a.axhline(50, color=INK2, lw=.7, ls=":"); a.text(len(CASES) - .6, 51, "chance", fontsize=6.5, color=INK2, ha="right")
    a.set_xticks(x); a.set_xticklabels([l for _, l in CASES], rotation=45, ha="right", fontsize=7); a.set_ylabel("Accuracy (%)"); a.set_ylim(40, 100)
    a.grid(axis="x", visible=False); a.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    m = pd.read_csv(os.path.join(RES, "ariel_mcs.csv")).set_index("case")
    rows = [("mission noise at N2, 102 bins", "Tier 3 binning"), ("mission noise at N2, tier-2 binning (51)", "Tier 2"), ("mission noise at N1, tier-1 binning (7)", "Tier 1")]
    xb = np.arange(len(rows)); bands = [("acc_M", "M hosts"), ("acc_K", "K"), ("acc_G", "G"), ("acc_F+", "F+")]
    for k, (col_, lab) in enumerate(bands):
        b.bar(xb + (k - 1.5) * .2, [m.loc[r, col_] * 100 for r, _ in rows], .19, color=[SERIES[1], SERIES[3], SERIES[2], SERIES[0]][k], label=lab)
    b.plot(xb, [m.loc[r, "accuracy"] * 100 for r, _ in rows], color=INK, marker="D", ms=4, lw=1, label="all targets", zorder=5)
    b.set_xticks(xb); b.set_xticklabels([l for _, l in rows], fontsize=7); b.set_ylim(40, 100); b.set_ylabel("Accuracy on Ariel's Known Targets (%)")
    b.grid(axis="x", visible=False); b.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    panel_label(a, "a"); panel_label(b, "b")
    fig.tight_layout(); save(fig, "fig4_tiers.png")


if __name__ == "__main__":
    main()
