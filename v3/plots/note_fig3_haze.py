"""Note Figure 3: why haze blinds the consortium classifier. (a) One test planet with CH4 above 1e-4 at the
classifier's Tier-3 binning, without haze and with haze at 3e7 m^-3 (noise-free; the three photometric points
shaded). (b) Share of planets that truly contain each molecule which the classifier reports as present, against
haze density, for the published inputs, the inputs without the three photometric points, and the AIRS bins only
(mean over the four classifiers and four molecules)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, V3)
from style import INK, INK2, MUTED, GRID, SERIES, figure, panel_label, save  # noqa: E402
RES = os.path.join(V3, "results")

def main():
    import alfnoor_faithful as F, alfnoor_trust as AT
    from bin_spectra import bin_native
    wl = np.load(os.path.join(V3, "data", "native_wl.npy")); e = F.layout_edges("tier3_r20"); cen = 0.5 * (e[1:] + e[:-1])
    Pte = pd.read_parquet(os.path.join(AT.OUT, "pop1_params.parquet"))
    C = np.load(os.path.join(AT.OUT, "pop1_native.npy")).astype(float); H = np.load(os.path.join(AT.OUT, "pop1_native_haze3e7.npy")).astype(float)
    ok = np.isfinite(C).all(1) & np.isfinite(H).all(1) & (Pte["atm CH4"].to_numpy() > -4) & (Pte["cloud_pressure"].to_numpy() > 1e5)
    Cb, Hb = bin_native(C[ok], wl, e), bin_native(H[ok], wl, e)
    add = np.median((Hb - Cb)[:, cen < 1.1], axis=1); i = int(np.argsort(add)[len(add) // 2])      # median optical effect
    fig, axs = figure(0.46, ncols=2, gridspec_kw={"width_ratios": [1.2, 1]})
    a = axs[0]
    a.axvspan(0.5, 1.1, color=GRID, alpha=.6, lw=0)
    a.plot(cen, Cb[i] * 1e6, color=SERIES[0], marker="o", ms=2.4, lw=1.0, label="no haze")
    a.plot(cen, Hb[i] * 1e6, color=SERIES[1], marker="o", ms=2.4, lw=1.0, label="haze 3×10$^7$ m$^{-3}$")
    a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 3, 5, 7.8]); a.set_xticklabels(["0.5", "1", "2", "3", "5", "7.8"])
    a.set_xlabel("Wavelength (µm)"); a.set_ylabel("Transit depth (ppm)"); a.legend(loc="upper right")
    a.text(0.74, 6330, "photometric\npoints", ha="center", va="center", fontsize=6.5, color=INK2)
    panel_label(a, "a")
    b = axs[1]
    d = pd.read_csv(os.path.join(RES, "alfnoor_haze_mechanism.csv"))
    g = d.groupby(["variant", "case"]).recall.mean() * 100
    xs = {"clean": 1e4, "haze2e6": 2e6, "haze3e7": 3e7, "haze2p4e8": 2.4e8, "haze1e10": 1e10}
    for k, (v, lab) in enumerate((("A published", "published inputs"), ("C no_optical", "photometric points removed"), ("D airs_only", "AIRS bins only"))):
        b.plot(list(xs.values()), [g.loc[(v, c)] for c in xs], color=SERIES[[1, 0, 2][k]], marker="o", ms=3.5, label=lab)
    b.set_xscale("log"); b.set_xticks(list(xs.values())); b.set_xticklabels(["none", "2e6", "3e7", "2.4e8", "1e10"])
    b.set_xlabel("Haze density (m$^{-3}$)"); b.set_ylabel("Present molecules detected (%)"); b.set_ylim(0, 75)
    b.legend(loc="lower left", fontsize=6.8); panel_label(b, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "note_fig3_haze.png")

if __name__ == "__main__":
    main()
