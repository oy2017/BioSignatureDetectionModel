"""Note Figure 1: the re-implemented consortium classifier against its published Table 6 (48 cells:
4 molecules x 4 classifiers x 3 abundance thresholds). Dashed band: +/- 5 accuracy points."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK2, MUTED, GRID, SERIES, figure, save  # noqa: E402
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PUB = {"KNN": {"CH4": (79, 83, 85), "CO2": (77, 79, 82), "H2O": (64, 71, 82), "NH3": (75, 82, 84)},
       "MLP": {"CH4": (78, 85, 87), "CO2": (77, 81, 83), "H2O": (70, 76, 84), "NH3": (80, 86, 87)},
       "RFC": {"CH4": (77, 82, 87), "CO2": (76, 79, 83), "H2O": (69, 74, 82), "NH3": (78, 85, 87)},
       "SVC": {"CH4": (79, 86, 89), "CO2": (79, 83, 84), "H2O": (69, 78, 84), "NH3": (81, 87, 87)}}
TH = ["1e-5", "1e-4", "1e-3"]; MARK = {"1e-5": "o", "1e-4": "s", "1e-3": "^"}
MOLS = ["CH4", "H2O", "CO2", "NH3"]; LAB = {"CH4": "CH$_4$", "H2O": "H$_2$O", "CO2": "CO$_2$", "NH3": "NH$_3$"}

def main():
    d = pd.read_csv(os.path.join(RES, "alfnoor_faithful_faithful_tier3_r20_radiometric.csv"), dtype={"threshold": str})
    d = d[~d.model.str.startswith("vote")]
    fig, ax = figure(0.72)
    ax.fill_between([55, 95], [50, 90], [60, 100], color=GRID, alpha=.55, lw=0, zorder=0)
    ax.plot([55, 95], [55, 95], color=INK2, lw=.8, ls="--", zorder=1)
    diffs = []
    for k, mol in enumerate(MOLS):
        for th in TH:
            xs, ys = [], []
            for c in PUB:
                r = d[(d.model == c) & (d.molecule == mol) & (d.threshold == th)].iloc[0]
                xs.append(PUB[c][mol][TH.index(th)]); ys.append(r.clean * 100); diffs.append(ys[-1] - xs[-1])
            ax.scatter(xs, ys, s=26, marker=MARK[th], color=SERIES[k], edgecolor="white", linewidth=.6, zorder=3,
                       label=LAB[mol] if th == "1e-4" else None)
    for th in TH:
        ax.scatter([], [], marker=MARK[th], color=MUTED, s=26, label=f"abundance > {th.replace('1e-', '10$^{-')}" + "}$")
    ax.set_xlim(55, 95); ax.set_ylim(55, 95); ax.set_aspect("equal")
    ax.set_xlabel("Published accuracy (%)"); ax.set_ylabel("Re-implementation accuracy (%)")
    ax.legend(loc="upper left", ncol=2, handletextpad=.3, columnspacing=.8)
    diffs = np.array(diffs); print(f"mean diff {diffs.mean():+.2f}, max |diff| {np.abs(diffs).max():.1f}, within 3: {(np.abs(diffs)<=3).sum()}/48")
    save(fig, "note_fig1_reproduction.png")

if __name__ == "__main__":
    main()
