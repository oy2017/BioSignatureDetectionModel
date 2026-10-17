"""Assemble WASP-39 b's JWST transmission spectrum from the published products and convert it to the
classifier's input (REVISION_PLAN.md, A1-A7). Products (all untrusted downloads, read only):

  benchmark  Carter & May et al. 2024 (Zenodo 10161743): the uniform re-reduction of the four ERS
             modes, R = 100 products with fixed limb darkening (the fitted-limb-darkening set is the
             alternate), PRISM dilution-corrected with the saturated 0.63-2.12 um flagged points dropped;
             0.52-5.34 um
  ers        the four discovery-paper products: NIRISS SOSS orders 1-2 (Feinstein et al. 2023, Zenodo
             7709712; nirHiss R = 300 fiducial, supreme-SPOON alternate), NIRCam F322W2 (Ahrer et al. 2023,
             Zenodo 7101283; Eureka!), NIRSpec G395H (Alderson et al. 2023, Zenodo 7185300; weighted
             average of eleven reductions) and NIRSpec PRISM (Rustamkulov et al. 2023, Zenodo 7388032;
             FIREFLy fiducial, Eureka!, Tiberius, tshirt alternates; saturated region dropped)
  miri       MIRI LRS 5-12 um (Powell et al. 2024, Zenodo 10055845; Eureka! fiducial, SPARTA, Tiberius
             alternates), added to either set; Ariel ends at 7.8 um

Modes are aligned through their overlaps onto NIRISS order 1 (jwst_to_ariel.align_offsets); MIRI, which
shares one bin with the near-infrared modes, is levelled through the benchmark's best-fit model
(align_to_model), and the alternative of leaving it as published is in the sensitivity matrix. Bins
without data are filled from the benchmark model levelled onto the data, or flat at the measured mean.

Usage:
  python wasp39_assemble.py                 # the fiducial (benchmark + MIRI) with full output
  python wasp39_assemble.py --sensitivity   # the matrix of reductions, offsets, fills and absorber masks
  python wasp39_assemble.py --set ers --niriss supreme-SPOON --prism Tiberius --miri SPARTA
Writes results/wasp39/{segments,coverage,converted,verdicts}_<tag>.* and sensitivity.{txt,csv}
"""
import argparse
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)   # numpy on macOS Accelerate reports spurious FP flags from matmul
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import jwst_readers as R  # noqa: E402

OUT = os.path.join(RESULTS, "wasp39"); os.makedirs(OUT, exist_ok=True)
ROOT = os.path.join(HERE, "data", "jwst", "wasp39")
CARTER = "10161743/extracted/ZENODO"
# bands of absorbers the training grid lacks (expectation A7): masked bins are refilled
ABSORBER_BANDS = {"Na": (0.57, 0.61), "K": (0.75, 0.79), "SO2": (3.95, 4.15), "CO": (4.50, 4.90)}


def benchmark_model(root=ROOT):
    return R.carter_model(os.path.join(root, CARTER, "5_MODEL_SPECTRUM", "BestFitModel.txt"))


def reference_model():
    """Fallback gap-filling model when the benchmark archive is absent: the forward model on the MCS
    bulk parameters at 10x-solar-like abundances."""
    import generate_grid as G, alfnoor_faithful as AF
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    M = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet")); r = M[M.name == "WASP-39b"].iloc[0].to_dict()
    r.update({"atm top_pressure": 1e-4, "atm H2O": -2.3, "atm CO2": -4.5, "atm CH4": -7.0, "atm NH3": -7.0, "atm CO": -3.5, "atm O3": -15.0})
    G.FILL_GAS = AF.FILL
    return wl, G.one_native(r, wl).astype(float)


def segments(root=ROOT, set="benchmark", niriss="nirHiss", prism="FIREFLy", miri="Eureka!", ld="Fixed", prism_saturated=False):
    S = {}
    if set == "benchmark":
        d = os.path.join(root, CARTER, "4_TRANSMISSION_SPECTRA", f"{ld}_LimbDarkening")
        S["SOSS-o2"] = R.carter_csv(os.path.join(d, "NIRISS_SOSS_Order2", "R100.csv"))
        S["SOSS-o1"] = R.carter_csv(os.path.join(d, "NIRISS_SOSS_Order1", "R100.csv"))
        S["PRISM"] = R.carter_csv(os.path.join(d, "NIRSpec_PRISM", "bins_scale1+dilution.csv" if ld == "Fixed" else "bins_scale1.csv"), drop_saturated=not prism_saturated)
        S["NIRCam"] = R.carter_csv(os.path.join(d, "NIRCam_F322W2", "R100.csv"))
        S["NRS1"] = R.carter_csv(os.path.join(d, "NIRSpec_G395H_NRS1", "R100.csv"))
        S["NRS2"] = R.carter_csv(os.path.join(d, "NIRSpec_G395H_NRS2", "R100.csv"))
    elif set in ("ers", "prelim"):
        n = os.path.join(root, "7709712/extracted/zenodo_upload/4_TRANSMISSION_SPECTRA")
        S["SOSS-o2"] = R.ers_niriss_h5(os.path.join(n, f"{niriss}_transmission_spectrum_R_300_order_2.h5"), 2)
        S["SOSS-o1"] = R.ers_niriss_h5(os.path.join(n, f"{niriss}_transmission_spectrum_R_300_order_1.h5"), 1)
        if set == "ers":
            S["PRISM"] = R.prism_h5(os.path.join(root, "7388032/extracted/transit_spectra", f"{prism}_transit_spec.h5"), None if prism_saturated else (0.634, 2.115))
        S["NIRCam"] = R.ers_h5(os.path.join(root, "7101283/extracted/2_TRANSMISSION_SPECTRA/LW-transit-spectrum-W39b-NIRCam-15nm_Eureka.h5"))
        S["G395H"] = R.ers_nc(os.path.join(root, "7185300/extracted/G395H_paper_data/3_TRANSMISSION_SPECTRA/transit-spectrum-W39b-G395H-10pix_weighted-average.nc"))
    elif set == "prism":
        S["PRISM"] = R.prism_h5(os.path.join(root, "7388032/extracted/transit_spectra", f"{prism}_transit_spec.h5"), None if prism_saturated else (0.634, 2.115))
    else:
        raise ValueError(set)
    if miri:
        f = {"Eureka!": "Eureka!-wasp-39b-spectrum.h5", "SPARTA": "SPARTA-wasp39b-spectrum.h5", "Tiberius": "Tiberius-wasp-39b-spectrum.h5"}[miri]
        S["MIRI"] = R.miri_h5(os.path.join(root, "10055845/extracted/miri_lrs_zenodo/Figure2_Spectra/spectra", f))
    return S


def convert(S, model, offsets="align", miri_level="model", fill="model", mask=None, reference=None):
    """Segments -> (depth104, err104, coverage table, offsets, model level). offsets: 'align' | 'none'.
    miri_level: 'model' | 'published'. fill: 'model' | 'flat'. mask: list of (lo, hi) um whose bins are
    refilled as if unmeasured."""
    edges, labels = J.tier3_layout(); wl_m, y_m = model
    reference = reference or ("SOSS-o1" if "SOSS-o1" in S else next(iter(S)))
    off = J.align_offsets(S, reference) if offsets == "align" else {k: 0.0 for k in S}
    level = 0.0
    if "MIRI" in S and miri_level == "model" and len(S) > 1:
        off, level = J.align_to_model(S, {k: v for k, v in off.items() if k != "MIRI"}, wl_m, y_m, ["MIRI"])
    d, e, cov = J.combine(S, edges, off)
    if mask:
        cen = 0.5 * (edges[1:] + edges[:-1])
        for lo, hi in mask:
            m = (cen >= lo) & (cen < hi); d[m] = np.nan; e[m] = np.nan; cov.loc[m, "covered"] = 0.0
    d_f, e_f, filled = J.fill_gaps(d, e, wl_m, y_m, edges, coverage=cov)
    if fill == "flat":
        d_f[filled] = np.nanmean(d_f[~filled])
    cov["filled"] = filled; cov["depth"] = d_f; cov["err"] = e_f
    return d_f, e_f, cov, off, level


def score(name, d, edges, n=300, variants=("all", "no_optical", "airs_only"), sig=None):
    import alfnoor_jwst as AJ
    sig = sig or {t: J.ariel_sigma("WASP-39b", edges, c) for t, c in (("tier1", "tier1_transits"), ("tier3", "tier3_transits"))}
    return AJ.summarize(name, d, edges, sig, n=n)


def compact(d, edges, sig, n=300):
    """One line per configuration: all-input verdicts at 1e-4 (KNN MLP RFC SVC), mean Tier-1 frequency, band metrics."""
    import alfnoor_jwst as AJ
    V = AJ.verdict_table(d, variants=("all",)); V = V[V.threshold == "1e-4"]
    F = AJ.tier_frequencies(d, sig, n=n, variants=("all",))
    row = {}
    for m in ("H2O", "CH4", "CO2", "NH3"):
        row[f"{m}_verdicts"] = "".join(str(int(v)) for v in V[V.molecule == m].sort_values("learner").present)
        row[f"{m}_tier1"] = float(F[F.molecule == m].frequency.mean())
    bm = AJ.band_metrics(d, edges); row.update({f"M_{k}": v for k, v in bm.items()})
    return row


def run_one(a):
    tag = a.tag or (f"{a.set}" + (f"_{a.ld}LD" if a.set == "benchmark" and a.ld != "Fixed" else "") + (f"_{a.niriss}" if a.set in ("ers", "prelim") and a.niriss != "nirHiss" else "")
                    + (f"_prism{a.prism}" if a.set in ("ers", "prism") and a.prism != "FIREFLy" else "") + (f"_miri{a.miri}" if a.miri and a.miri != "Eureka!" else "") + ("" if a.miri else "_nomiri"))
    S = segments(a.root, a.set, a.niriss, a.prism, a.miri, a.ld, a.prism_saturated); edges, labels = J.tier3_layout()
    model = benchmark_model(a.root) if os.path.exists(os.path.join(a.root, CARTER, "5_MODEL_SPECTRUM", "BestFitModel.txt")) else reference_model()
    for k, s in S.items():
        print(f"{k:<8} {len(s['wl'])} points, {s['wl'].min():.3f}-{s['wl'].max():.3f} um, median depth {np.median(s['depth'])*1e6:.0f} ppm, median err {np.median(s['err'])*1e6:.0f} ppm")
    np.savez(os.path.join(OUT, f"segments_{tag}.npz"), **{f"{k}_{f}": v for k, s in S.items() for f, v in s.items() if v is not None})
    d_f, e_f, cov, off, level = convert(S, model, "align", "model", "model")
    print("\nmode offsets onto SOSS order 1 (ppm): " + ", ".join(f"{k} {v*1e6:+.0f}" for k, v in off.items()) + f"; model levelled by {level*1e6:+.0f} ppm")
    filled = cov.filled.to_numpy(bool); cov.to_csv(os.path.join(OUT, f"coverage_{tag}.csv"), index=False)
    print(f"\nbins filled from the model: {int(filled.sum())} of {len(d_f)}: " + ", ".join(f"{labels[b]} {edges[b]:.2f}-{edges[b+1]:.2f}" for b in np.where(filled)[0][:8]) + (" ..." if filled.sum() > 8 else ""))
    print(cov.groupby("channel").agg(bins=("bin", "size"), filled=("filled", "sum"), median_err_ppm=("err", lambda s: np.nanmedian(s) * 1e6)).to_string())
    text, V = score(f"WASP-39 b [{tag}: offsets aligned, MIRI levelled through the model, {int(filled.sum())} bins model-filled]", d_f, edges, n=a.draws)
    d_flat = d_f.copy(); d_flat[filled] = np.nanmean(d_f[~filled])
    text2, V2 = score(f"WASP-39 b [{tag}: filled bins replaced by the measured mean]", d_flat, edges, n=a.draws)
    V["fill"] = "model"; V2["fill"] = "flat"; pd.concat([V, V2]).to_csv(os.path.join(OUT, f"verdicts_{tag}.csv"), index=False)
    open(os.path.join(OUT, f"verdicts_{tag}.txt"), "w").write(text + "\n\n" + text2 + "\n")
    print(text); print(); print(text2)


def run_sensitivity(a):
    """Expectations A5 (reductions), A6 (round trip), A7 (absorber masks), plus offsets, fill and MIRI level."""
    edges, labels = J.tier3_layout(); model = benchmark_model(a.root); sig = J.ariel_sigma("WASP-39b", edges)
    rows = []

    def add(name, S, **kw):
        d, e, cov, off, level = convert(S, model, **kw)
        r = dict(configuration=name, n_modes=len(S), filled=int(cov.filled.sum()), **compact(d, edges, sig, a.draws))
        r["offsets_ppm"] = "; ".join(f"{k} {v*1e6:+.0f}" for k, v in off.items() if k != "SOSS-o1"); rows.append(r)
        print(f"{name:<52} filled {r['filled']:>3}  " + "  ".join(f"{m} {r[m+'_verdicts']} {r[m+'_tier1']*100:3.0f}%" for m in ("H2O", "CH4", "CO2", "NH3")) + f"   M_H2O {r['M_H2O']:.2f} M_CH4 {r['M_CH4']:.2f} M_CO2 {r['M_CO2']:.2f}")
        return d, cov

    print("configuration                                        filled   verdicts at 1e-4 (KNN MLP RFC SVC), mean Tier-1 frequency; band metrics")
    B = segments(a.root); d0, cov0 = add("benchmark + MIRI (fiducial)", B)
    add("  MIRI as published (no level)", B, miri_level="published")
    add("  all modes as published (no offsets)", B, offsets="none", miri_level="published")
    add("  fitted limb darkening", segments(a.root, ld="Fit"))
    add("  PRISM saturated region kept", segments(a.root, prism_saturated=True))
    for m in ("SPARTA", "Tiberius"):
        add(f"  MIRI {m}", segments(a.root, miri=m))
    add("  without MIRI, model fill", segments(a.root, miri=None))
    add("  without MIRI, flat fill", segments(a.root, miri=None), fill="flat")
    add("  fiducial, flat fill", B, fill="flat")
    E = segments(a.root, "ers"); add("ERS products + MIRI (nirHiss, FIREFLy, Eureka!)", E)
    add("  NIRISS supreme-SPOON", segments(a.root, "ers", niriss="supreme-SPOON"))
    for p in ("Eureka", "Tiberius", "Tshirt"):
        add(f"  PRISM {p}", segments(a.root, "ers", prism=p))
    add("  ERS without PRISM (the preliminary set) + MIRI", {k: v for k, v in E.items() if k != "PRISM"})
    add("  ERS without PRISM, without MIRI, flat fill", {k: v for k, v in E.items() if k not in ("PRISM", "MIRI")}, fill="flat")
    add("PRISM alone + MIRI (FIREFLy, saturated kept)", segments(a.root, "prism", prism_saturated=True), reference="PRISM")
    add("PRISM alone + MIRI (FIREFLy, saturated dropped)", segments(a.root, "prism"), reference="PRISM")
    for name, bands in (("all four absorbers", list(ABSORBER_BANDS.values())),) + tuple((k, [v]) for k, v in ABSORBER_BANDS.items()):
        add(f"  fiducial, {name} masked, model fill", B, mask=bands)
    add("  fiducial, all four absorbers masked, flat fill", B, mask=list(ABSORBER_BANDS.values()), fill="flat")
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)

    lines = round_trip_lines(B, cov0, model, edges)
    hdr = "configuration                                        filled   verdicts at 1e-4 (KNN MLP RFC SVC), mean Tier-1 frequency; band metrics"
    body = [hdr] + [f"{r['configuration']:<52} filled {r['filled']:>3}  " + "  ".join(f"{m} {r[m+'_verdicts']} {r[m+'_tier1']*100:3.0f}%" for m in ("H2O", "CH4", "CO2", "NH3")) + f"   M_H2O {r['M_H2O']:.2f} M_CH4 {r['M_CH4']:.2f} M_CO2 {r['M_CO2']:.2f}" for r in rows]
    body.append("\noffsets (ppm, onto NIRISS order 1): " + " | ".join(f"{r['configuration'].strip()}: {r['offsets_ppm']}" for r in rows[:3] + rows[10:11]))
    open(os.path.join(OUT, "sensitivity.txt"), "w").write("\n".join(body + lines) + "\n")


def round_trip_lines(B, cov0, model, edges):
    """A6: the benchmark model observed like the benchmark segments and converted, against the model binned
    directly: first noise-free (the conversion's own bias), then scattered with the segments' errors (200
    realizations); verdicts and learner scores at 1e-4, all inputs."""
    import alfnoor_jwst as AJ
    wl_m, y_m = model; L = AJ.learners("all")
    direct, reals = J.round_trip(wl_m, y_m, B, edges, n=200, seed=0)
    clean = J.sample_clean(wl_m, y_m, B); d_c, e_c, cov_c = J.combine(clean, edges); d_c, _, _ = J.fill_gaps(d_c, e_c, wl_m, y_m, edges, coverage=cov_c)
    Zd, Zc, Zr = (AJ.znorm(x, L["cols"]) for x in (direct, d_c, reals))
    lines = ["\n== round trip (A6): the benchmark model sampled at the benchmark grids, converted, against the model binned directly"]
    lines.append(f"noise-free: rms conversion bias over the 104 bins {np.sqrt(np.mean((d_c - direct)**2))*1e6:.0f} ppm, max {np.abs(d_c - direct).max()*1e6:.0f} ppm (bin {int(np.argmax(np.abs(d_c - direct)))}); "
                 f"with the segments' errors, 200 realizations: rms {np.sqrt(np.mean((reals - direct[None, :])**2))*1e6:.0f} ppm (median propagated bin error {np.nanmedian(cov0.err)*1e6:.0f} ppm)")
    for m in ("H2O", "CH4", "CO2", "NH3"):
        cells = []
        for k in ("KNN", "MLP", "RFC", "SVC"):
            mdl = L["models"][("1e-4", m, k)]
            cells.append(f"{k} direct {int(mdl.predict(Zd)[0])} noise-free {int(mdl.predict(Zc)[0])} noised {mdl.predict(Zr).mean()*100:3.0f}% (score {AJ._present_prob(mdl, Zd)[0]:+.2f} -> {AJ._present_prob(mdl, Zc)[0]:+.2f} -> {AJ._present_prob(mdl, Zr).mean():+.2f})")
        lines.append(f"  {m}: " + "; ".join(cells))
    print("\n".join(lines)); return lines


def run_round_trip(a):
    edges, labels = J.tier3_layout(); model = benchmark_model(a.root); B = segments(a.root)
    d, e, cov0, off, level = convert(B, model)
    lines = round_trip_lines(B, cov0, model, edges)
    open(os.path.join(OUT, "round_trip.txt"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT, help="fetch_jwst_data.sh fills it")
    ap.add_argument("--set", default="benchmark", choices=["benchmark", "ers", "prelim", "prism"]); ap.add_argument("--ld", default="Fixed", choices=["Fixed", "Fit"])
    ap.add_argument("--niriss", default="nirHiss"); ap.add_argument("--prism", default="FIREFLy"); ap.add_argument("--miri", default="Eureka!")
    ap.add_argument("--no-miri", dest="miri", action="store_const", const=None); ap.add_argument("--prism-saturated", action="store_true")
    ap.add_argument("--tag", default=None); ap.add_argument("--draws", type=int, default=300); ap.add_argument("--sensitivity", action="store_true"); ap.add_argument("--round-trip", action="store_true")
    a = ap.parse_args()
    run_round_trip(a) if a.round_trip else run_sensitivity(a) if a.sensitivity else run_one(a)


if __name__ == "__main__":
    main()
