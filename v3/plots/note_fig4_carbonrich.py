"""Note Figure 4: the answer depends on the classifier. (a) Accuracy on carbon-rich planets of the carbon-rich
screen without and with HCN and C2H2 in the atmosphere: on the simulated test sets at Tier-3 binning and on the
known Ariel targets under the payload noise model at Tier-3, Tier-2 and Tier-1 binning. The consortium classifier's
mean loss to the same omission is 0.1 points. (b) Accuracy lost to haze at 3e7 m^-3 with all inputs and with the
three photometric points removed, for the consortium classifier and for the carbon-rich screen at Tier-3 and
Tier-1 binning; the clean accuracy each version gives up by removing the points is printed above the bars."""
import os, re, sys, json
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, V3)
from style import INK, INK2, MUTED, SERIES, figure, panel_label, save  # noqa: E402
RES = os.path.join(V3, "results")

def grid_crich():
    from common import TESTS, load_split, MODELS
    from augment import shifted_test
    best = json.load(open(os.path.join(RES, "ariel_best.json")))["best"]; fr = joblib.load(os.path.join(MODELS, f"ariel_{best}.joblib"))
    Xc = np.vstack([load_split(t, "ariel")[0] for t in TESTS]); yc = np.concatenate([load_split(t, "ariel")[1] for t in TESTS])
    Xa, ya = shifted_test("absorbers", "ariel")
    acc = lambda X, y: ((fr["model"].predict_proba(fr["features"].transform(X))[:, 1] >= .5) == y)[y == 1].mean() * 100
    return acc(Xc, yc), acc(Xa, ya)

def main():
    c0, c1 = grid_crich()
    txt = open(os.path.join(RES, "mcs_absorbers.txt")).read()
    real = [tuple(map(float, re.search(rf"^{cfg}.*?carbon-rich\s+([\d.]+) ->\s+([\d.]+)", txt, re.M).groups())) for cfg in ("ariel", "tier2", "tier1")]
    fig, axs = figure(0.5, ncols=2)
    a = axs[0]; labels = ["grid\nTier 3", "targets\nTier 3", "targets\nTier 2", "targets\nTier 1"]
    vals = [(c0, c1)] + real; x = np.arange(4)
    a.bar(x - .19, [v[0] for v in vals], .36, color=SERIES[0], label="without HCN, C$_2$H$_2$")
    a.bar(x + .19, [v[1] for v in vals], .36, color=SERIES[1], label="with HCN, C$_2$H$_2$")
    a.axhline(50, color=MUTED, lw=.7, ls="--"); a.set_xticks(x); a.set_xticklabels(labels, fontsize=6.8)
    a.set_ylabel("Carbon-rich planets correct (%)"); a.set_ylim(0, 125); a.set_yticks([0, 25, 50, 75, 100]); a.legend(loc="upper center", ncol=2, fontsize=6.6, handlelength=1.2, columnspacing=.8); panel_label(a, "a")
    g = pd.read_csv(os.path.join(RES, "haze_generality_carbonrich.csv")); m = pd.read_csv(os.path.join(RES, "alfnoor_haze_mechanism.csv"))
    mg = m.groupby(["variant", "case"]).accuracy.mean() * 100
    groups = [("consortium", (mg.loc[("A published", "clean")] - mg.loc[("A published", "haze3e7")], mg.loc[("C no_optical", "clean")] - mg.loc[("C no_optical", "haze3e7")]),
               mg.loc[("A published", "clean")] - mg.loc[("C no_optical", "clean")])]
    for cfg, lab in (("ariel", "carbon-rich\nTier 3"), ("tier1", "carbon-rich\nTier 1")):
        s = g[g.config == cfg].set_index(["variant", "case"])
        groups.append((lab, (s.loc[("all bins", "haze_3p0e7"), "loss"], s.loc[("no optical", "haze_3p0e7"), "loss"]),
                       s.loc[("all bins", "clean"), "accuracy"] - s.loc[("no optical", "clean"), "accuracy"]))
    b = axs[1]; x = np.arange(len(groups))
    b.bar(x - .19, [gg[1][0] for gg in groups], .36, color=SERIES[1], label="all inputs")
    b.bar(x + .19, [gg[1][1] for gg in groups], .36, color=SERIES[0], label="photometric points removed")
    for xi, gg in zip(x, groups):
        b.text(xi, max(gg[1]) + 1.0, f"clean −{gg[2]:.1f}", ha="center", fontsize=6.3, color=INK2)
    b.set_xticks(x); b.set_xticklabels([gg[0] for gg in groups], fontsize=6.8); b.set_ylabel("Accuracy lost to haze (points)")
    b.set_ylim(0, 42); b.set_yticks([0, 10, 20, 30]); b.legend(loc="upper center", ncol=2, fontsize=6.6, handlelength=1.2, columnspacing=.8); panel_label(b, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "note_fig4_carbonrich.png")

if __name__ == "__main__":
    main()
