"""Manuscript Figure 5: muting of the real JWST spectra against the simulator population. (a) For each real planet on
the panel, the dispersion of its clear H2/He twin over the measured Tier-3 bins divided by the noise-corrected
dispersion of the real spectrum (filled), and the same for the most muted atmosphere the population prior allows at
the planet's bulk parameters (open); a planet to the right of its open marker is muted beyond anything the simulator
can draw for it. (b) Cumulative distribution of the same muting over the 955 POP-I planets, each rendered with its
own cloud deck against its clear twin, with the real planets marked. Reads results/jwst_panel/muting_census.csv and
results/jwst_panel/muting_population.csv."""
import os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, INK2, MUTED, GRID, SERIES, figure, panel_label, save  # noqa: E402
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(V3, "results", "jwst_panel")
SUFFIX = {"-alt": " (Schmidt & MacDonald)", "-hu": " (Hu et al.)", "-hm": " (Holmberg & Madhusudhan)"}
GIANTS = ["HAT-P-26b", "WASP-107b", "WASP-39b", "GJ3470b", "WASP-166b", "WASP-96b", "WASP-17b", "WASP-80b", "HD209458b", "WASP-52b", "WASP-94Ab", "HAT-P-12b"]

def pretty(n):
    return {"TOI-270d": "TOI-270 d (Felix)", "TOI-270d-hm": "TOI-270 d (Holmberg)", "K2-18b": "K2-18 b (Madhusudhan)", "K2-18b-alt": "K2-18 b (Schmidt)", "K2-18b-hu": "K2-18 b (Hu)", "LTT3780c": "TOI-732 c",
            "GJ3470b": "GJ 3470 b", "GJ9827d": "GJ 9827 d", "HD189733b": "HD 189733 b", "HD209458b": "HD 209458 b", "HIP67522b": "HIP 67522 b", "WASP-94Ab": "WASP-94A b"}.get(n, n[:-1] + " " + n[-1])
def main():
    C = pd.read_csv(os.path.join(RES, "muting_census.csv")); P = pd.read_csv(os.path.join(RES, "muting_population.csv"))
    C = C.sort_values("muting").reset_index(drop=True); Bt = pd.read_csv(os.path.join(RES, "muting_bootstrap.csv")).set_index("planet")
    fig, axs = figure(0.50, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]}, constrained_layout=True)
    a = axs[0]; y = np.arange(len(C))
    a.axvspan(P.muting.quantile(0.05), P.muting.quantile(0.95), color=GRID, alpha=.5, lw=0, label="POP-I 5th–95th Percentile")
    a.axvline(P.muting.max(), color=MUTED, lw=0.8, ls=":", label="POP-I Most Muted")
    for i, r in C.iterrows():
        giant = r.planet in GIANTS
        a.plot([r.max_muting_in_prior, r.muting], [i, i], color=SERIES[1] if giant else INK2, lw=0.8, zorder=2)
        a.plot(r.max_muting_in_prior, i, marker="o", mfc="white", mec=SERIES[1] if giant else INK2, ms=4.5, mew=0.9, zorder=3, lw=0)
        if r.planet in Bt.index and np.isfinite(r.muting) and np.isfinite(Bt.lo[r.planet]) and np.isfinite(Bt.hi[r.planet]):
            a.plot([Bt.lo[r.planet], Bt.hi[r.planet]], [i, i], color=INK, lw=1.6, zorder=3.5, solid_capstyle="butt")
        a.plot(r.muting, i, marker="o", color=SERIES[1] if giant else INK2, ms=4.5, zorder=4, lw=0)
    a.plot([], [], marker="o", color=INK2, lw=0, label="Real Spectrum, Other Planets"); a.plot([], [], marker="o", color=SERIES[1], lw=0, label="Real Spectrum, the Twelve Giants"); a.plot([], [], marker="o", mfc="white", mec=INK2, lw=0, label="Most Muted in Prior"); a.plot([], [], color=INK, lw=1.6, label="68 % Interval")
    a.set_yticks(y); a.set_yticklabels([pretty(n) for n in C.planet], fontsize=6.6); a.set_ylim(-0.7, len(C) - 0.3)
    a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 3, 5]); a.set_xticklabels(["0.5", "1", "2", "3", "5"]); a.set_xlabel("Muting (Clear Twin / Real Spectrum)")
    a.grid(axis="y", visible=False); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.0, handlelength=1.2, columnspacing=1.2); panel_label(a, "a")
    b = axs[1]; m = np.sort(P.muting.to_numpy()); b.plot(m, np.arange(1, len(m) + 1) / len(m) * 100, color=SERIES[0], lw=1.4, label=f"POP-I ({len(m)} Planets)")
    G = C[C.planet.isin(GIANTS)].sort_values("muting").reset_index(drop=True)
    for i, r in G.iterrows():
        b.axvline(r.muting, color=SERIES[1], lw=0.7, alpha=.8, label="The Twelve Giants" if i == 0 else None)   # named in panel a
    b.set_xscale("log"); b.set_xticks([0.5, 1, 2, 3, 5]); b.set_xticklabels(["0.5", "1", "2", "3", "5"]); b.set_xlim(0.45, 7)
    b.set_xlabel("Muting by the Planet's\nOwn Deck"); b.set_ylabel("POP-I Planets Muted Less (%)"); b.set_ylim(0, 101)
    b.legend(loc="lower right", fontsize=6.2, framealpha=1).set_zorder(10); panel_label(b, "b")
    save(fig, "fig5_muting.png")

if __name__ == "__main__":
    main()
