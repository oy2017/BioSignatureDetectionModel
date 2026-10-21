"""Manuscript Figure 4: the two error classes. (a) Fraction of consortium labels decidable at Tier-1 precision
(departure-aware dchi2 above the calibrated threshold) on clean spectra and under each departure, with the departure
in the counterfactual model; the noise cases are the clean dchi2 / k^2. (b) Share of the errors each departure
introduces that are recoverable (the label still decidable under the departure), mean over the four learners, with
the clean bar the posterior-based share of all clean errors. (c) Posterior probability of the true CH4 class from
the clear-model and the modelled-haze retrievals of the same hazed spectra, against the clean-spectrum posterior.
Reads results/identifiability_{none,haze3e7,spots20,cloud1e2}_tier1.csv, results/diagnostic_departure_classes.csv,
results/diagnostic_retrieval_classes.csv and results/identifiability_calibration{,_haze3e7,_haze3e7_modelled}.csv;
a departure whose files are missing is left out."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, MUTED, GRID, SERIES, figure, panel_label, save  # noqa: E402
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(V3, "results")
MOLS = ["CH4", "H2O", "CO2", "NH3"]; LAB = {"CH4": "CH$_4$", "H2O": "H$_2$O", "CO2": "CO$_2$", "NH3": "NH$_3$"}
CASES = [("clean", "clean", "none", 1.0), ("haze3e7", "haze 3×10$^7$ m$^{-3}$", "haze3e7", 1.0), ("spots20", "spots 20 %", "spots20", 1.0),
         ("cloud1e2", "deck 10$^2$ Pa", "cloud1e2", 1.0), ("white_x2", "noise × 2", "none", 0.25), ("white_x3", "noise × 3", "none", 1 / 9)]
COL = {"clean": INK2, "haze3e7": SERIES[1], "spots20": SERIES[3], "cloud1e2": SERIES[2], "white_x2": SERIES[0], "white_x3": "#184f95"}

def logit_clip(p, lo=1e-3):
    p = np.clip(np.asarray(p, float), lo, 1 - lo); return np.log(p / (1 - p))

def main():
    thr = pd.read_csv(os.path.join(RES, "identifiability_calibration_thresholds.csv")).set_index("molecule").dchi2_at_95
    cases = [c for c in CASES if os.path.exists(os.path.join(RES, f"identifiability_{c[2]}_tier1.csv"))]
    # (a) decidable fractions
    dec = {}
    for key, lab, base, scale in cases:
        I = pd.read_csv(os.path.join(RES, f"identifiability_{base}_tier1.csv")); I = I[np.isfinite(I.dchi2)]
        dec[key] = [100 * float((I[I.molecule == m].dchi2 * scale > thr[m]).mean()) for m in MOLS]
    # (b) recoverable share of the errors each departure introduces (dchi2 definition); clean: posterior definition of all errors
    D = pd.read_csv(os.path.join(RES, "diagnostic_departure_classes.csv")); R = pd.read_csv(os.path.join(RES, "diagnostic_retrieval_classes.csv"))
    rec = {"clean": [100 * R[(R.case == "clean") & (R.molecule == m)].recoverable.mean() for m in MOLS]}
    for key, lab, base, scale in cases[1:]:
        d = D[D.case == key]
        if len(d):
            rec[key] = [100 * d[d.molecule == m].recoverable_introduced.mean() for m in MOLS]
    fig, axs = figure(0.40, ncols=3, gridspec_kw={"width_ratios": [1.15, 1.15, 1]}, constrained_layout=True)
    x = np.arange(len(MOLS))
    for ax, data, ylab, letter in ((axs[0], dec, "Labels decidable (%)", "a"), (axs[1], rec, "Introduced errors recoverable (%)", "b")):
        keys = [c[0] for c in cases if c[0] in data]; w = 0.8 / len(keys)
        for j, k in enumerate(keys):
            lab = dict((c[0], c[1]) for c in CASES)[k]
            ax.bar(x + (j - (len(keys) - 1) / 2) * w, data[k], w, color=COL[k], label=lab if letter == "a" else None, lw=0)
        ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in MOLS]); ax.set_ylabel(ylab, fontsize=7.5); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
        panel_label(ax, letter)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(dec), fontsize=6.8, handlelength=1.0, columnspacing=1.0, bbox_to_anchor=(0.5, -0.07))
    # (c) CH4 p_true: clean posterior against the two haze retrievals of the same spectra
    c = axs[2]
    clean = pd.read_csv(os.path.join(RES, "identifiability_calibration.csv")).set_index("name")
    for fn, lab, col, mk in (("identifiability_calibration_haze3e7.csv", "clear model on hazed", SERIES[1], "o"),
                             ("identifiability_calibration_haze3e7_modelled.csv", "haze modelled", SERIES[0], "s")):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        H = pd.read_csv(p); H = H[H.name.isin(clean.index) & np.isfinite(H.CH4_p_true)]
        c.scatter(logit_clip(clean.loc[H.name, "CH4_p_true"]), logit_clip(H.CH4_p_true), s=11, color=col, marker=mk, alpha=.8, lw=0, label=f"{lab} (n = {len(H)})")
    ticks = [1e-3, 0.05, 0.5, 0.95, 1 - 1e-3]; tl = ["≤0.001", "0.05", "0.5", "0.95", "≥0.999"]
    for v in (0.05, 0.95):
        c.axhline(logit_clip(v), color=GRID, lw=0.8); c.axvline(logit_clip(v), color=GRID, lw=0.8)
    c.plot([logit_clip(1e-3), logit_clip(1 - 1e-3)], [logit_clip(1e-3), logit_clip(1 - 1e-3)], color=MUTED, lw=0.6, ls="--")
    c.set_xticks(logit_clip(ticks)); c.set_xticklabels(tl, fontsize=6.5); c.set_yticks(logit_clip(ticks)); c.set_yticklabels(tl, fontsize=6.5)
    c.set_xlabel("$p_{\\rm true}$(CH$_4$), clean spectrum", fontsize=7.5); c.set_ylabel("$p_{\\rm true}$(CH$_4$), hazed spectrum", fontsize=7.5); c.grid(False)
    c.legend(loc="lower left", bbox_to_anchor=(0.0, 0.10), fontsize=6.2, handlelength=0.8, markerscale=1.2, borderaxespad=0.2); panel_label(c, "c")
    save(fig, "fig4_identifiability.png")

if __name__ == "__main__":
    main()
