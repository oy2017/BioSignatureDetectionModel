"""Figure 5: precision-recall of the frozen pipeline on clean spectra and on the
same planets under a 10^4 Pa cloud deck, with the 95 %-recall operating points."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402
from style import GRID, INK2, SERIES, figure, save  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    pr = pd.read_parquet(os.path.join(RES, "ariel_pr.parquet"))
    txt = open(os.path.join(RES, "ariel_calibration.txt")).read()
    fig, ax = figure(0.5)
    cases = list(dict.fromkeys(pr["case"]))
    for k, case in enumerate(cases):
        c = pr[pr["case"] == case]
        label = "clean" if case == "clean" else "same planets, 10⁴ Pa cloud deck"
        ax.plot(c["recall"], c["precision"], color=SERIES[k], label=label)
        m = re.search(rf"{re.escape(case)}:.*?recall >= 95%: threshold ([0-9.]+), precision ([0-9.]+)%", txt, re.S)
        if m:
            thr, prec = float(m.group(1)), float(m.group(2))
            ax.plot(0.95, prec / 100, marker="o", ms=7, color=SERIES[k], markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(f"{label}: 95 % recall at threshold {thr:.2f}, precision {prec:.0f} %",
                        (0.95, prec / 100), xytext=(0.53, 0.62 - 0.07 * k), textcoords="data", fontsize=7, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
    ax.axhline(0.5, color=GRID, lw=0.8, ls=":")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0.5, 1.0); ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig5_threshold_transfer.png")


if __name__ == "__main__":
    main()
