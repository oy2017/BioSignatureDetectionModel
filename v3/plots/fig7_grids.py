"""Manuscript figure: balanced credit on the two held-out model grids through the real masks (results/goyal_test/goyal_{local,crossfield}.txt)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from style import INK2, SERIES, save  # noqa: E402
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = ["published", "AIRS-only", "haze+deck-retrained", "the method", "matched+amp", "C+B-mlp"]; LAB = ["Re-Implemented Classifier", "AIRS-Only Input", "Haze+Deck-Retrained", "Routing Method", "Per-Reduction", "Matched Ensemble"]; GR = [("local", "Local Condensation"), ("crossfield", "Photochemical")]


def read(tag, prec):
    out, on = {}, False
    for l in open(os.path.join(V3, "results", "goyal_test", f"goyal_{tag}.txt")):
        if l.startswith("=="): on = l.startswith(f"== {prec}, all:")
        elif on: k = l.split(" balanced ")[0].strip(); out[k] = float(l.split(" balanced ")[1].split()[0])
    return [out[d] for d in D]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True); w = 0.13
    for a, prec, title in zip(ax, ("JWST", "Tier-1"), ("(a) JWST Precision", "(b) Tier-1 Noise")):
        for j, (d, lab) in enumerate(zip(D, LAB)):
            a.bar(np.arange(2) + (j - 2.5) * w, [read(t, prec)[j] for t, _ in GR], w, label=lab, color=(["#b5b5b5", SERIES[0], SERIES[2], SERIES[3], SERIES[4], "#111111"])[j])
        a.axhline(0.5, color=INK2, ls=":", lw=1); a.set_xticks(np.arange(2)); a.set_xticklabels([g for _, g in GR]); a.set_title(title, fontsize=9); a.set_ylim(0.45, 0.82)
    ax[0].set_ylabel("Balanced Credit"); ax[1].legend(fontsize=6.5, loc="upper left", ncol=2); fig.tight_layout(); save(fig, "fig7_grids.png")


if __name__ == "__main__":
    main()
