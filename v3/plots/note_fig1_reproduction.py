"""Note Figure 1: the re-implemented consortium classifier against its published Table 6 (48 cells:
4 molecules x 4 classifiers x 3 abundance thresholds). (a) Re-implemented against published accuracy, grey band
+/- 5 points. (b) Difference (re-implemented minus published) for every cell, grouped by molecule; marker shape
gives the threshold."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK2, MUTED, GRID, SERIES, figure, panel_label, save  # noqa: E402
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PUB = {"KNN": {"CH4": (79, 83, 85), "CO2": (77, 79, 82), "H2O": (64, 71, 82), "NH3": (75, 82, 84)},
       "MLP": {"CH4": (78, 85, 87), "CO2": (77, 81, 83), "H2O": (70, 76, 84), "NH3": (80, 86, 87)},
       "RFC": {"CH4": (77, 82, 87), "CO2": (76, 79, 83), "H2O": (69, 74, 82), "NH3": (78, 85, 87)},
       "SVC": {"CH4": (79, 86, 89), "CO2": (79, 83, 84), "H2O": (69, 78, 84), "NH3": (81, 87, 87)}}
TH = ["1e-5", "1e-4", "1e-3"]; MARK = {"1e-5": "o", "1e-4": "s", "1e-3": "^"}
MOLS = ["CH4", "H2O", "CO2", "NH3"]; LAB = {"CH4": "CH$_4$", "H2O": "H$_2$O", "CO2": "CO$_2$", "NH3": "NH$_3$"}
THLAB = {"1e-5": "> 10$^{-5}$", "1e-4": "> 10$^{-4}$", "1e-3": "> 10$^{-3}$"}


def main():
    d = pd.read_csv(os.path.join(RES, "alfnoor_faithful_faithful_tier3_r20_radiometric.csv"), dtype={"threshold": str})
    d = d[~d.model.str.startswith("vote")]
    fig, (a, b) = figure(0.42, ncols=2, gridspec_kw={"width_ratios": [1, 1.15]})
    a.fill_between([55, 95], [50, 90], [60, 100], color=GRID, alpha=.55, lw=0, zorder=0)
    a.plot([55, 95], [55, 95], color=INK2, lw=.8, ls="--", zorder=1)
    rng = np.random.default_rng(3); diffs = []
    for k, mol in enumerate(MOLS):
        for th in TH:
            xs, ys = [], []
            for c in PUB:
                r = d[(d.model == c) & (d.molecule == mol) & (d.threshold == th)].iloc[0]
                xs.append(PUB[c][mol][TH.index(th)]); ys.append(r.clean * 100)
            dd = np.array(ys) - np.array(xs); diffs += list(dd)
            a.scatter(xs, ys, s=18, marker=MARK[th], color=SERIES[k], edgecolor="white", linewidth=.5, zorder=3)
            xj = k + (TH.index(th) - 1) * 0.22 + rng.uniform(-.05, .05, len(dd))
            b.scatter(xj, dd, s=18, marker=MARK[th], color=SERIES[k], edgecolor="white", linewidth=.5, zorder=3)
    a.set_xlim(55, 95); a.set_ylim(55, 95); a.set_aspect("equal")
    a.set_xlabel("Published accuracy (%)"); a.set_ylabel("Re-implementation (%)"); panel_label(a, "a")
    b.axhspan(-5, 5, color=GRID, alpha=.55, lw=0, zorder=0); b.axhline(0, color=INK2, lw=.8, ls="--", zorder=1)
    b.set_xticks(range(4)); b.set_xticklabels([LAB[m] for m in MOLS]); b.set_ylim(-6, 9); b.set_yticks([-4, -2, 0, 2, 4]); b.set_xlim(-.6, 3.6)
    b.set_ylabel("Difference (points)"); panel_label(b, "b")
    for th in TH:
        b.scatter([], [], marker=MARK[th], color=MUTED, s=18, label=THLAB[th])
    b.legend(loc="upper center", ncol=3, fontsize=6.8, handletextpad=.2, columnspacing=.9, title="abundance threshold", title_fontsize=6.8)
    fig.tight_layout(w_pad=1.2)
    diffs = np.array(diffs); print(f"mean diff {diffs.mean():+.2f}, max |diff| {np.abs(diffs).max():.1f}")
    save(fig, "note_fig1_reproduction.png")


if __name__ == "__main__":
    main()
