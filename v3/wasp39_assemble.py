"""Assemble WASP-39 b's JWST transmission spectrum from the published products and convert it to the
classifier's input (REVISION_PLAN.md, A1-A4). Preliminary until the PRISM, MIRI and Carter benchmark
archives are in: bins without data are filled from a reference model and flagged.

Products (all untrusted downloads, read only): NIRISS SOSS orders 1 and 2 (Feinstein et al. 2023,
Zenodo 7709712; nirHiss at R = 300 as the fiducial, supreme-SPOON as the alternate), NIRCam F322W2
(Ahrer et al. 2023, Zenodo 7101283; Eureka!) and NIRSpec G395H (Alderson et al. 2023, Zenodo
7185300; the weighted average of their eleven reductions).

Usage: python wasp39_assemble.py [--root v3/data/jwst/wasp39] [--niriss supreme-SPOON]
Writes results/wasp39/segments_<tag>.npz, coverage_<tag>.csv, converted_<tag>.csv and verdicts_<tag>.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import jwst_readers as R  # noqa: E402

OUT = os.path.join(RESULTS, "wasp39"); os.makedirs(OUT, exist_ok=True)


def reference_model():
    """A WASP-39 b model at 10x-solar-like abundances for gap filling (the Carter benchmark model
    replaces it once that archive is in): the forward model on the MCS bulk parameters."""
    import generate_grid as G, alfnoor_faithful as AF
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    M = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet")); r = M[M.name == "WASP-39b"].iloc[0].to_dict()
    r.update({"atm top_pressure": 1e-4, "atm H2O": -2.3, "atm CO2": -4.5, "atm CH4": -7.0, "atm NH3": -7.0, "atm CO": -3.5, "atm O3": -15.0})
    G.FILL_GAS = AF.FILL
    return wl, G.one_native(r, wl).astype(float)


def segments(root, niriss="nirHiss"):
    S = {}
    n = os.path.join(root, "7709712/extracted/zenodo_upload/4_TRANSMISSION_SPECTRA")
    S["SOSS-o2"] = R.ers_niriss_h5(os.path.join(n, f"{niriss}_transmission_spectrum_R_300_order_2.h5"), 2)
    S["SOSS-o1"] = R.ers_niriss_h5(os.path.join(n, f"{niriss}_transmission_spectrum_R_300_order_1.h5"), 1)
    S["NIRCam"] = R.ers_h5(os.path.join(root, "7101283/extracted/2_TRANSMISSION_SPECTRA/LW-transit-spectrum-W39b-NIRCam-15nm_Eureka.h5"))
    S["G395H"] = R.ers_nc(os.path.join(root, "7185300/extracted/G395H_paper_data/3_TRANSMISSION_SPECTRA/transit-spectrum-W39b-G395H-10pix_weighted-average.nc"))
    for k in ("7388032", "10055845"):
        pass   # PRISM and MIRI readers are added when their archives arrive
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(HERE, "data", "jwst", "wasp39"), help="fetch_jwst_data.sh fills it")
    ap.add_argument("--niriss", default="nirHiss")
    ap.add_argument("--draws", type=int, default=1000); a = ap.parse_args()
    tag = f"prelim_{a.niriss}"
    S = segments(a.root, a.niriss); edges, labels = J.tier3_layout()
    for k, s in S.items():
        print(f"{k:<8} {len(s['wl'])} points, {s['wl'].min():.3f}-{s['wl'].max():.3f} um, median depth {np.median(s['depth'])*1e6:.0f} ppm, median err {np.median(s['err'])*1e6:.0f} ppm")
    np.savez(os.path.join(OUT, f"segments_{tag}.npz"), **{f"{k}_{f}": v for k, s in S.items() for f, v in s.items() if v is not None})
    off = J.align_offsets(S, "SOSS-o1")
    print("\nmode offsets onto SOSS order 1 (ppm): " + ", ".join(f"{k} {v*1e6:+.0f}" for k, v in off.items()))
    d, e, cov = J.combine(S, edges, off)
    wl_m, y_m = reference_model()
    d_f, e_f, filled = J.fill_gaps(d, e, wl_m, y_m, edges, coverage=cov)
    cov["filled"] = filled; cov["depth"] = d_f; cov["err"] = e_f; cov.to_csv(os.path.join(OUT, f"coverage_{tag}.csv"), index=False)
    print(f"\nbins filled from the model: {int(filled.sum())} of {len(d)}: " + ", ".join(f"{labels[b]} {edges[b]:.2f}-{edges[b+1]:.2f}" for b in np.where(filled)[0][:6]) + (" ..." if filled.sum() > 6 else ""))
    print(cov.groupby("channel").agg(bins=("bin", "size"), filled=("filled", "sum"), median_err_ppm=("err", lambda s: np.nanmedian(s) * 1e6)).to_string())
    import alfnoor_jwst as AJ
    sig = {t: J.ariel_sigma("WASP-39b", edges, c) for t, c in (("tier1", "tier1_transits"), ("tier3", "tier3_transits"))}
    text, V = AJ.summarize(f"WASP-39 b [{tag}: SOSS+NIRCam+G395H, offsets aligned, {int(filled.sum())} bins model-filled]", d_f, edges, sig, n=a.draws)
    # the same without the model fill: measured bins only, the learners retrained on those columns is not
    # available, so instead the filled bins are set to the measured bins' mean (a flat continuation)
    d_flat = d_f.copy(); d_flat[filled] = np.nanmean(d_f[~filled])
    text2, V2 = AJ.summarize(f"WASP-39 b [{tag}: filled bins replaced by the measured mean]", d_flat, edges, sig, n=a.draws)
    V["fill"] = "model"; V2["fill"] = "flat"; pd.concat([V, V2]).to_csv(os.path.join(OUT, f"verdicts_{tag}.csv"), index=False)
    open(os.path.join(OUT, f"verdicts_{tag}.txt"), "w").write(text + "\n\n" + text2 + "\n")
    print(text); print(); print(text2)


if __name__ == "__main__":
    main()
