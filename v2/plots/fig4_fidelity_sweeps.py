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
    ("Haze 3×10⁷ m⁻³", "haze", "haze_3p0e7"),
    ("Cloud deck 10² Pa", "cloud", "cloud_1e2Pa"),
    ("Stellar spots 20 %", "tlse", "tlse_spots20"),
    ("Correlated noise, SNR 5", "correlated noise", "snr5"),
    ("ExoMol + HITRAN O₃", "exomol", "exomol_o3"),
    ("White noise, SNR 5", "white noise", "snr5"),
    ("Absolute 200 ppm floor", "absolute noise", "200 ppm"),
    ("Cloud deck 10³ Pa", "cloud", "cloud_1e3Pa"),
    ("Stellar spots 5 %", "tlse", "tlse_spots05"),
    ("Gain ramp, 2× noise", "gain ramp", "x2.0"),
    ("Haze 2×10⁶ m⁻³", "haze", "haze_2p0e6"),
    ("Faculae 10 %", "tlse", "tlse_fac10"),
    ("Cloud deck 10⁴ Pa", "cloud", "cloud_1e4Pa"),
    ("Independent RT code", "exotransmit", "exotransmit"),
    ("ExoMol, 3 non-label gases", "exomol", "exomol"),
    ("Global offset, 2× noise", "baseline offset", "x2.0"),
]
OTHERS = [("norm_mlp", "MLP, same features", 0),
          ("norm_rf", "Random Forest, same features", 2),
          ("pca_xgb", "XGBoost, PCA features", 1)]


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

    fig, (a, b) = figure(0.58, ncols=2, gridspec_kw={"width_ratios": [1.32, 1]})
    y = np.arange(len(vals))
    colours = [SERIES[1] if abs(v) >= 10 else SERIES[0] for _, v in vals]
    a.barh(y, [v for _, v in vals], color=colours, height=0.62)
    a.set_yticks(y); a.set_yticklabels([l for l, _ in vals], fontsize=6.6); a.invert_yaxis()
    a.axvline(0, color=INK2, lw=0.8)
    for yi, (_, v) in zip(y, vals):
        if v <= -6:
            a.text(v + 0.7, yi, f"{v:+.1f}", va="center", ha="left", fontsize=6.3, color=INK)
        else:
            a.text(v - 0.5, yi, f"{v:+.1f}", va="center", ha="right", fontsize=6.3, color=INK2)
    a.set_xlim(min(v for _, v in vals) * 1.1, 5)
    a.set_xlabel(f"Accuracy change vs clean {base:.1f} % (points)")
    a.grid(axis="y", visible=False); panel_label(a, "A")

    from scipy.stats import rankdata, spearmanr
    ref = np.array([get(base_df, ax, c) for _, ax, c in ROWS])
    ok0 = np.isfinite(ref)
    rref = rankdata(-ref[ok0])
    b.plot([0, ok0.sum() + 1], [0, ok0.sum() + 1], color=GRID, lw=0.9, ls=":", zorder=1)
    for tag, label, ci in OTHERS:
        d = load(tag)
        if d is None:
            continue
        v = np.array([get(d, ax, c) for _, ax, c in ROWS])
        ok = ok0 & np.isfinite(v)
        rho = spearmanr(ref[ok], v[ok]).statistic
        b.scatter(rankdata(-ref[ok]), rankdata(-v[ok]), s=26, color=SERIES[ci], zorder=3,
                  edgecolor="white", linewidth=0.8, label=f"{label}  ρ = {rho:.2f}")
    b.set_xlim(0, ok0.sum() + 1); b.set_ylim(0, ok0.sum() + 1)
    b.set_xlabel("Rank of the loss, frozen pipeline")
    b.set_ylabel("Rank of the loss, other pipeline")
    b.legend(loc="upper left", fontsize=6.3); panel_label(b, "B")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig4_fidelity_sweeps.png")


if __name__ == "__main__":
    main()
