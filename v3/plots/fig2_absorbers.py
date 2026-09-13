"""Figure 2: the omitted species. (a) A hot carbon-rich test planet at Ariel binning, rendered without
and with HCN + C2H2 at their equilibrium abundances. (b) The clean-trained screen's probability of
'carbon-rich' for the carbon-rich test planets, clean vs with the species present."""
import json, os, sys
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, V3)
from style import INK, INK2, MUTED, SERIES, figure, panel_label, save  # noqa: E402
from common import DATA, MODELS, RESULTS, TESTS, centres, configs, load_split  # noqa: E402
from augment import shifted_test  # noqa: E402
from bin_spectra import bin_native  # noqa: E402


def main():
    best = json.load(open(os.path.join(RESULTS, "ariel_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"ariel_{best}.joblib")); ff, fm = fr["features"], fr["model"]
    cen = np.array(centres("ariel")); edges = np.array(configs()["ariel"]["edges"]); wl = np.load(os.path.join(DATA, "native_wl.npy"))
    Xc = np.vstack([load_split(t, "ariel")[0] for t in TESTS]); yc = np.concatenate([load_split(t, "ariel")[1] for t in TESTS])
    Pc = pd.concat([load_split(t, "ariel", noisy=False)[2] for t in TESTS], ignore_index=True)
    Xs, ys = shifted_test("absorbers", "ariel")
    p_c = fm.predict_proba(ff.transform(Xc))[:, 1]; p_s = fm.predict_proba(ff.transform(Xs))[:, 1]
    # example planet: hot, carbon-rich, large-ish amplitude, noise-free renders
    Xn = np.vstack([np.load(os.path.join(DATA, f"{t}_native.npy")) for t in TESTS]).astype(float)
    Xa = np.vstack([np.load(os.path.join(DATA, f"{t}_native_absorbers.npy")) for t in TESTS]).astype(float)
    T = Pc["atm temperature"].to_numpy(); ok = (yc == 1) & (T > 1500) & np.all(np.isfinite(Xa), axis=1)
    bc, ba = bin_native(Xn, wl, edges), bin_native(Xa, wl, edges)
    amp = np.ptp(bc, axis=1); cand = np.where(ok & (amp > np.percentile(amp[ok], 60)) & (amp < np.percentile(amp[ok], 80)))[0]
    rel = np.abs(ba[cand] - bc[cand]).mean(1) / amp[cand]; i = cand[np.argsort(rel)[len(rel) // 2]]   # the median-effect planet, not the extreme
    fig, (a, b) = figure(0.46, ncols=2, gridspec_kw={"width_ratios": [1.45, 1.05]})
    a.plot(cen, bc[i] * 1e6, color=SERIES[0], lw=1.4, label="training forward model (no HCN, C$_2$H$_2$)")
    a.plot(cen, ba[i] * 1e6, color=SERIES[1], lw=1.4, label="with HCN + C$_2$H$_2$ at equilibrium abundance")
    a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 3, 5, 7.8]); a.set_xticklabels(["0.5", "1", "2", "3", "5", "7.8"])
    a.set_xlabel("Wavelength (µm)"); a.set_ylabel("Transit Depth (ppm)")
    r = Pc.iloc[i]; a.text(0.02, 0.97, f"T = {r['atm temperature']:.0f} K, C/O = {r['co_ratio']:.2f}\np(carbon-rich): {p_c[i]:.2f} → {p_s[i]:.2f}",
                           transform=a.transAxes, fontsize=7, color=INK2, va="top")
    a.legend(fontsize=6.8, loc="lower right")
    bins = np.linspace(0, 1, 26); m = yc == 1
    b.hist(p_c[m], bins, color=SERIES[0], alpha=.75, label=f"clean ({(p_c[m] >= .5).mean()*100:.0f} % called carbon-rich)")
    b.hist(p_s[m], bins, color=SERIES[1], alpha=.75, label=f"with HCN + C$_2$H$_2$ ({(p_s[m] >= .5).mean()*100:.0f} %)")
    b.axvline(0.5, color=INK2, lw=.8, ls="--"); b.set_xlabel("Probability of Carbon-Rich"); b.set_ylabel("Carbon-Rich Test Planets (Count)")
    b.legend(fontsize=6.5, loc="upper center"); b.grid(axis="x", visible=False)
    panel_label(a, "a"); panel_label(b, "b")
    fig.tight_layout(); save(fig, "fig2_absorbers.png")


if __name__ == "__main__":
    main()
