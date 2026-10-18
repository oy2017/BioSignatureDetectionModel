"""Muting census of the real JWST spectra against the simulator population (REVISION_FINDINGS.md, F7).

For every real planet with a converted spectrum: the noise-corrected dispersion of its measured Tier-3 bins
(std of the depths minus the per-bin errors in quadrature), the dispersion of its clear H2/He twin at the
published abundances over the same bins, the twins with a grey deck at the population's lowest allowed pressure
(5 x 10^2 Pa) and below it, the twin at the population's coldest allowed temperature (0.7 T_eq) with that deck, and the deck pressure a grey cloud would need to mute the twin down to the real
spectrum. The muting factor is clear twin / real. For comparison, the muting factor of a random sample of POP-I
planets: their rendered spectrum (with the population's own cloud deck) against their clear twin.

Usage: python jwst_muting_census.py [--pop 150]   -> results/jwst_panel/muting_census.{txt,csv}
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
V3 = os.path.dirname(os.path.abspath(__file__)); os.chdir(V3); sys.path.insert(0, V3)
import generate_grid as G, alfnoor_faithful as AF, forward_model as FM, jwst_to_ariel as J
from bin_spectra import bin_native
G.FILL_GAS = AF.FILL
edges, labels = J.tier3_layout(); wl = np.load("data/native_wl.npy")
M = pd.read_parquet("data/mcs/mcs_params.parquet")
# published abundances (log10 VMR) as in jwst_controls.py; the three marked take the published water abundance or
# metallicity with the other gases at the floor (sensitivity: the twin at log H2O -2 and -3 is reported for all)
TWINS = {"TOI-270d": {"CH4": -1.64, "H2O": -1.10, "CO2": -1.67, "NH3": -5.0},
         "K2-18b":   {"CH4": -1.74, "H2O": -4.0, "CO2": -3.35, "NH3": -5.6},
         "K2-18b-alt": {"CH4": -1.74, "H2O": -4.0, "CO2": -3.35, "NH3": -5.6},
         "TOI-421b": {"CH4": -7.0, "H2O": -3.0, "CO2": -7.0, "NH3": -7.0},     # Davenport+2025: H2O, MMW 2.3-2.7 (solar), no haze
         "HAT-P-18b": {"CH4": -7.0, "H2O": -3.03, "CO2": -7.0, "NH3": -7.0},   # Fu+2022: log H2O -3.03 (+0.31/-0.25), CH4 depleted
         "WASP-107b": {"CH4": -5.8, "H2O": -2.1, "CO2": -3.9, "NH3": -5.0},
         "HD189733b": {"CH4": -7.0, "H2O": -2.9, "CO2": -5.3, "NH3": -7.0},    # Fu+2024: 3-5x solar, CH4 depleted
         "HAT-P-26b": {"CH4": -5.5, "H2O": -2.07, "CO2": -2.91, "NH3": -4.2},
         "GJ3470b":  {"CH4": -4.05, "H2O": -1.08, "CO2": -2.47, "NH3": -7.3},
         "WASP-166b": {"CH4": -7.9, "H2O": -1.30, "CO2": -2.23, "NH3": -4.0},
         "WASP-39b": {"CH4": -7.0, "H2O": -2.3, "CO2": -4.5, "NH3": -7.0}}
ASSUMED = {"TOI-421b", "HAT-P-18b", "HD189733b"}
DECKS = [5e2, 1e2, 3e1, 1e1, 3e0, 1e0]        # Pa; 5e2 is the population's lowest cloud-top pressure

def render(r, cp=None):
    y = FM.spectrum(r, wl, G.GASES, G.FILL_GAS, cloud_pressure=cp)
    return None if y is None else bin_native(np.asarray(y, float)[None, :], wl, edges)[0]

def disp(x, m): return float(np.std(x[m]) * 1e6)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pop", type=int, default=150); a = ap.parse_args()
    L, rows = [], []
    L.append("Muting census: dispersion (std over the measured Tier-3 bins, ppm) of the real spectrum, noise-corrected, against its clear H2/He twin")
    L.append("and the twin under grey decks; muting = clear twin / real; the population's cloud prior stops at 5e2 Pa; 'prior' = clear twin / the most muted")
    L.append("twin the population allows at these bulk parameters (0.7 T_eq or the 5e2 Pa deck, whichever mutes more); the right-hand block repeats it over the AIRS bins (1.95-7.8 um)")
    L.append(f"{'planet':<11} {'bins':>4} {'real':>6} {'noise':>6} {'corr':>6} {'clear':>6} " + " ".join(f"{int(d):>5}Pa" for d in DECKS) + f" {'H2O-2':>6} {'H2O-3':>6} {'muting':>7} {'0.7Teq+500Pa':>7} {'max in prior':>12} {'deck needed':>11} | AIRS only: bins   corr  clear muting  prior")
    for pl, ab in TWINS.items():
        covf = "results/wasp39/coverage_benchmark.csv" if pl == "WASP-39b" else f"results/jwst_panel/{pl}_coverage.csv"
        cov = pd.read_csv(covf); meas = ~cov.filled.to_numpy(bool); real = cov.depth.to_numpy(); err = cov.err.to_numpy()
        s_real = disp(real, meas); noise = float(np.sqrt(np.nanmean(err[meas] ** 2)) * 1e6); s_corr = float(np.sqrt(max(s_real ** 2 - noise ** 2, 0.0)))
        r = M[M.name == pl.replace("-alt", "")].iloc[0].to_dict(); r.update({"atm top_pressure": 1e-4, "atm CO": -15.0, "atm O3": -15.0, **{f"atm {k}": v for k, v in ab.items()}})
        clear = render(r)
        if clear is None: L.append(f"{pl:<11} twin failed"); continue
        s_clear = disp(clear, meas); decks = {}
        for d in DECKS:
            x = render(r, d); decks[d] = disp(x, meas) if x is not None else np.nan
        sens = {}
        for h in (-2.0, -3.0):
            rr = dict(r); rr["atm H2O"] = h; x = render(rr); sens[h] = disp(x, meas) if x is not None else np.nan
        # the population's most muted atmosphere at these bulk parameters: T at the prior floor (0.7 T_eq) and the deck at 5e2 Pa
        rc = dict(r); rc["atm temperature"] = 0.7 * r["atm temperature"]; xc = render(rc, 5e2); s_cold = disp(xc, meas) if xc is not None else np.nan
        muting = s_clear / s_corr if s_corr > 0 else np.inf; max_prior = s_clear / min(decks[5e2], s_cold)
        # the same over the AIRS bins only (1.95-7.8 um), which excludes the optical slope
        ir = meas & (edges[:-1] >= 1.95); s_real_ir = disp(real, ir); noise_ir = float(np.sqrt(np.nanmean(err[ir] ** 2)) * 1e6)
        s_corr_ir = float(np.sqrt(max(s_real_ir ** 2 - noise_ir ** 2, 0.0))); s_clear_ir = disp(clear, ir)
        x5 = render(r, 5e2); s_ir_prior = min(disp(x5, ir) if x5 is not None else np.inf, disp(xc, ir) if xc is not None else np.inf)
        muting_ir = s_clear_ir / s_corr_ir if s_corr_ir > 0 else np.inf; max_prior_ir = s_clear_ir / s_ir_prior
        # deck pressure at which the twin's dispersion falls to the real one (log interpolation on the deck grid)
        dd = np.array([np.inf] + DECKS); ss = np.array([s_clear] + [decks[d] for d in DECKS])
        if s_corr >= s_clear: need = "none"
        elif s_corr < np.nanmin(ss): need = f"<{DECKS[-1]:g}"
        else:
            k = np.where(ss <= s_corr)[0][0]; lo, hi = np.log10(dd[k - 1]) if np.isfinite(dd[k - 1]) else 6.0, np.log10(dd[k])
            need = f"{10 ** (hi + (lo - hi) * (s_corr - ss[k]) / (ss[k - 1] - ss[k])):.3g}"
        L.append(f"{pl:<11} {meas.sum():>4} {s_real:6.0f} {noise:6.0f} {s_corr:6.0f} {s_clear:6.0f} " + " ".join(f"{decks[d]:7.0f}" for d in DECKS) + f" {sens[-2.0]:6.0f} {sens[-3.0]:6.0f} {muting:7.2f} {s_cold:7.0f} {max_prior:12.2f} {need:>11} | {ir.sum():>3} {s_corr_ir:6.0f} {s_clear_ir:6.0f} {muting_ir:6.2f} {max_prior_ir:6.2f}" + ("  (H2O from the literature, rest at floor)" if pl in ASSUMED else ""))
        rows.append(dict(planet=pl, bins=int(meas.sum()), real=s_real, noise=noise, real_corrected=s_corr, clear_twin=s_clear, **{f"deck_{int(d)}Pa": decks[d] for d in DECKS},
                         twin_H2O_m2=sens[-2.0], twin_H2O_m3=sens[-3.0], muting=muting, twin_cold_deck500=s_cold, max_muting_in_prior=max_prior, ir_bins=int(ir.sum()), ir_real_corrected=s_corr_ir, ir_clear_twin=s_clear_ir, ir_muting=muting_ir, ir_max_muting_in_prior=max_prior_ir, deck_needed_Pa=need, abundances_assumed=pl in ASSUMED))
    # the population: rendered (with its own deck) against clear, on a random sample
    src = AF.DIRS["faithful"]; Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet")); Y = np.load(os.path.join(src, "pop1_native.npy"))
    ok = np.all(np.isfinite(Y), axis=1); idx = np.random.default_rng(405).choice(np.where(ok)[0], a.pop, replace=False)
    allb = np.ones(104, bool); mut = []
    for i in idx:
        r = Pte.iloc[i].to_dict(); cp = r.pop("cloud_pressure"); rr = {k: v for k, v in r.items() if k not in ("atm HCN", "atm C2H2")}
        xr = bin_native(Y[i].astype(float)[None, :], wl, edges)[0]; xc = render(rr)
        if xc is None: continue
        irb = edges[:-1] >= 1.95
        mut.append(dict(index=int(i), name=r["name"], cloud_pressure=cp, rendered=disp(xr, allb), clear=disp(xc, allb), muting=disp(xc, allb) / disp(xr, allb), ir_muting=disp(xc, irb) / disp(xr, irb)))
    Pm = pd.DataFrame(mut); Pm.to_csv("results/jwst_panel/muting_population.csv", index=False)
    q = Pm.muting.quantile([0.5, 0.9, 0.99]).to_numpy()
    L.append(""); L.append(f"POP-I sample of {len(Pm)} planets, muting of the rendered spectrum (its own deck, 5e2-1e6 Pa) against its clear twin: median {q[0]:.2f}, 90th {q[1]:.2f}, 99th {q[2]:.2f}, max {Pm.muting.max():.2f}")
    qi = Pm.ir_muting.quantile([0.5, 0.9, 0.99]).to_numpy()
    L.append(f"  the same over the AIRS bins: median {qi[0]:.2f}, 90th {qi[1]:.2f}, 99th {qi[2]:.2f}, max {Pm.ir_muting.max():.2f}")
    L.append("fraction of POP-I planets muted at least as much as each real planet: " + ", ".join(f"{r['planet']} {100*(Pm.muting >= r['muting']).mean():.1f} %" for r in rows))
    pd.DataFrame(rows).to_csv("results/jwst_panel/muting_census.csv", index=False)
    open("results/jwst_panel/muting_census.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
