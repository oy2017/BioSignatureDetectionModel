"""Figure 4: A) the fidelity budget for the frozen pipeline; B) does that budget
transfer? Each point is one shifted case: the loss suffered by the frozen
pipeline against the loss suffered by another pipeline on the identical planets."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import GRID, INK, INK2, SERIES, figure, panel_label, save  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

ROWS = [
    # Labels match Table 1 in the paper word for word wherever a row appears in both,
    # so a reader matching a number across table and figure sees one name for it.
    ("Haze, 3\u00d710\u2077 m\u207b\u00b3", "haze", "haze_3p0e7"),
    ("Cloud deck, 10\u00b2 Pa", "cloud", "cloud_1e2Pa"),
    ("Stellar spots, 20%", "tlse", "tlse_spots20"),
    ("Correlated noise, SNR 5", "correlated noise", "snr5"),
    ("Alternative opacities + O\u2083 line list", "exomol", "exomol_o3"),
    ("White noise, SNR 5", "white noise", "snr5"),
    ("Absolute 200 ppm floor", "absolute noise", "200 ppm"),
    ("Cloud deck, 10\u00b3 Pa", "cloud", "cloud_1e3Pa"),
    ("Stellar spots, 5%", "tlse", "tlse_spots05"),
    ("Gain ramp, 2\u00d7 noise", "gain ramp", "x2.0"),
    ("Haze, 2\u00d710\u2076 m\u207b\u00b3", "haze", "haze_2p0e6"),
    ("Faculae, 10%", "tlse", "tlse_fac10"),
    ("Cloud deck, 10\u2074 Pa", "cloud", "cloud_1e4Pa"),
    ("Independent RT code", "exotransmit", "exotransmit"),
    ("Alternative opacities, 3 non-label gases", "exomol", "exomol"),
    ("Global additive offset, 2\u00d7 noise", "baseline offset", "x2.0"),
]
# Balanced grid: the frozen pipeline is XGBoost/normalized; these are the other
# five cells of {XGBoost, random forest} x {normalized, PCA, raw}.
# Short forms keep the legend clear of the data; the caption expands RF.
OTHERS = [("norm_rf", "RF, normalized", 2),
          ("pca_xgb", "XGBoost, PCA", 1),
          ("pca_rf", "RF, PCA", 4),
          ("raw_xgb", "XGBoost, raw", 3),
          ("raw_rf", "RF, raw", 7)]


def load(tag=None):
    f = "ariel_shifts.csv" if tag is None else f"ariel_{tag}_shifts.csv"
    p = os.path.join(RES, f)
    return pd.read_csv(p) if os.path.exists(p) else None


def get(df, axis, case):
    s = df[(df["axis"] == axis) & (df["case"] == case)]
    return float(s.iloc[0]["d_acc"]) if len(s) else np.nan


def main():
    base_df = load()
    base = base_df[base_df["axis"] == "clean"].iloc[0]["accuracy"] * 100
    vals = [(lab, get(base_df, ax, c)) for lab, ax, c in ROWS]
    vals = [(l, v) for l, v in vals if np.isfinite(v)]
    vals.sort(key=lambda t: t[1])

    fig, (a, b) = figure(0.46, ncols=2, gridspec_kw={"width_ratios": [1.32, 1]})
    y = np.arange(len(vals))
    colours = [SERIES[1] if abs(v) >= 10 else SERIES[0] for _, v in vals]
    a.barh(y, [v for _, v in vals], color=colours, height=0.62)
    a.set_yticks(y); a.set_yticklabels([l for l, _ in vals], fontsize=7.2); a.invert_yaxis()
    a.axvline(0, color=INK2, lw=0.8)
    for yi, (_, v) in zip(y, vals):
        if v <= -6:
            a.text(v + 0.7, yi, f"{v:+.1f}", va="center", ha="left", fontsize=7.2, color=INK)
        else:
            a.text(v - 0.5, yi, f"{v:+.1f}", va="center", ha="right", fontsize=7.2, color=INK2)
    a.set_xlim(min(v for _, v in vals) * 1.1, 5)
    a.set_xlabel(f"Accuracy change vs clean {base:.1f}% (points)")
    a.grid(axis="y", visible=False); panel_label(a, "A")

    from scipy.stats import rankdata, spearmanr
    # Correlations are computed on EVERY shared shift case, matching the text,
    # not only the cases plotted in panel A.
    def series(df):
        d = df[(df["axis"] != "clean") & (df["axis"] != "extrapolation")]
        return d.set_index(d["axis"] + "|" + d["case"])["d_acc"]
    ref_all = series(base_df)
    others = [(t, l, c, load(t)) for t, l, c in OTHERS]
    others = [(t, l, c, d) for t, l, c, d in others if d is not None]
    common = set(ref_all.index)
    for _, _, _, d in others:
        common &= set(series(d).index)
    common = sorted(common)
    r0 = ref_all[common]
    b.plot([0, len(common) + 1], [0, len(common) + 1], color=GRID, lw=0.9, ls=":", zorder=1)
    for tag, label, ci, d in others:
        v = series(d)[common]
        rho = spearmanr(r0, v).statistic
        b.scatter(rankdata(-r0), rankdata(-v), s=13, color=SERIES[ci], zorder=3,
                  edgecolor="white", linewidth=0.5, label=f"{label} (ρ={rho:.3f})")
    # A wide legend cannot fit either off-diagonal corner of a rank-rank plot without
    # clipping the diagonal, so give it a blank band above the data instead.
    b.set_xlim(0, len(common) + 1); b.set_ylim(0, (len(common) + 1) * 1.52)
    b.set_yticks([t for t in (0, 10, 20, 30, 40) if t <= len(common)])
    b.set_xlabel("Loss rank, frozen pipeline")
    b.set_ylabel("Loss rank, other pipeline")
    b.legend(loc="upper left", fontsize=6.8, handletextpad=0.3, borderpad=0.3,
             labelspacing=0.25, framealpha=0.9, frameon=True)
    panel_label(b, "B")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig4_fidelity_sweeps.png")


if __name__ == "__main__":
    main()
