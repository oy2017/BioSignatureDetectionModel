"""The JWST panel through the same conversion and classifier as WASP-39 b (REVISION_PLAN.md, A6).

Each planet's published, reduced transmission spectrum (untrusted downloads, read only; inventory in
jwst_validation_brief.md) is converted to the 104-bin input: modes aligned by their overlaps, bins
without data filled by a flat continuation at the measured mean ("flat" fill: no model is allowed to
put a feature into a gap) and flagged. Verdicts come from the three input variants at the delivered
precision and under Ariel Tier-1 noise at the planet's MCS transit count. Labels are the consensus
of REVISION_EXPECTATIONS.md; "amb." is not scored.

Usage: python jwst_panel.py --root <panel download dir> [--planets TOI-421b HAT-P-18b ...] [--draws 500]
Writes results/jwst_panel/<planet>_{coverage,verdicts}.csv, <planet>.txt and panel_summary.{csv,txt}
"""
import argparse
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)   # numpy on macOS Accelerate reports spurious FP flags from matmul
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import jwst_readers as R  # noqa: E402

TRAIN = os.environ.get("ALFNOOR_TRAIN", "faithful")      # see alfnoor_jwst.TRAIN / alfnoor_retrain.py
OUT = os.path.join(RESULTS, "jwst_panel" + ("" if TRAIN == "faithful" else f"_{TRAIN}")); os.makedirs(OUT, exist_ok=True)
MOLS = ["CH4", "H2O", "CO2", "NH3"]
P, A = "present", "absent"
# consensus labels at 10^-4 (REVISION_EXPECTATIONS.md); None = ambiguous, not scored
LABELS = {
    "TOI-270d": {"CH4": P, "H2O": None, "CO2": P, "NH3": A},
    "K2-18b": {"CH4": P, "H2O": None, "CO2": None, "NH3": A},   # H2O withdrawn: 95 % limit -3.06 lies above -4
    "K2-18b-alt": {"CH4": P, "H2O": None, "CO2": None, "NH3": A},   # H2O withdrawn: 95 % limit -2.44 lies above -4
    "TOI-421b": {"CH4": A, "H2O": None, "CO2": A, "NH3": None},
    "HAT-P-18b": {"CH4": A, "H2O": None, "CO2": A, "NH3": A},
    "WASP-107b": {"CH4": A, "H2O": P, "CO2": None, "NH3": A},
    "HD189733b": {"CH4": A, "H2O": None, "CO2": A, "NH3": A},
    "HAT-P-26b": {"CH4": A, "H2O": P, "CO2": P, "NH3": A},
    "GJ3470b": {"CH4": None, "H2O": P, "CO2": P, "NH3": A},
    "WASP-166b": {"CH4": A, "H2O": P, "CO2": P, "NH3": None},
    "TOI-1231b": {"CH4": P, "H2O": None, "CO2": P, "NH3": None},       # Sarkar et al. 2026: CH4 -0.85, CO2 -1.90; H2O < -2.24, NH3 < -3.34 (95 %)
    "K2-18b-hu": {"CH4": P, "H2O": A, "CO2": None, "NH3": A},          # Hu et al. 2025 reduction of the same planet (adds G235H)
    "TOI-270d-hm": {"CH4": P, "H2O": None, "CO2": P, "NH3": A},        # Holmberg & Madhusudhan 2024 discovery-team G395H only
    # second batch (free-chemistry constraints at the same rule: P if the 1-sigma lower bound is above -4, A if the upper bound or 95 % limit is below it)
    "GJ9827d": {"CH4": None, "H2O": P, "CO2": None, "NH3": None},     # Piaulet-Ghorayeb et al. 2024: H2O > -1.86 (SCARLET) / > -0.50; CH4 < -3.6, NH3 < -3.55, CO2 < -1.7 (95 %)
    "WASP-52b": {"CH4": A, "H2O": None, "CO2": None, "NH3": None},    # Pan et al. 2026 G395H (POSEIDON, spots): H2O -3.41 +/- 1.04; CO2 -5.09 +1.09/-1.14; CH4 < -7.39 (2 sigma); NH3 no constraint
    "L98-59d": {"CH4": A, "H2O": A, "CO2": None, "NH3": A},           # Gressier et al. 2024 (TauREx free): H2O < -5, CH4 < -6, NH3 < -6, CO2 < -2.5 (level of the limits not stated); H2S/SO2 spectrum
    "GJ1214b": {"CH4": P, "H2O": None, "CO2": P, "NH3": None},        # Schlawin et al. 2024 (Aurora free): CH4 -2.25 +0.72/-1.08 "suggestive" (3.3 sigma), CO2-rich 91 % (+8/-45); others unconstrained
    "LTT3780c": {"CH4": P, "H2O": None, "CO2": None, "NH3": A},       # TOI-732 c, Rigby et al. 2025 (6+2X free): CH4 -2.34 +0.71/-0.80 (3.2 sigma); CO2 < -2.56, H2O < -1.78, NH3 < -4.44 (95 %)
    "WASP-80b": {"CH4": None, "H2O": P, "CO2": None, "NH3": None},    # Bell et al. 2023 (free, transmission): H2O -1.8 +0.6/-1.0; CH4 -4.2 +0.5/-0.7 straddles the threshold; CO2 < -3.09, NH3 < -3.12 (3 sigma)
    "HD209458b": {"CH4": A, "H2O": None, "CO2": None, "NH3": A},      # Xue et al. 2024: equilibrium retrieval for H2O/CO2 (no free value); free 3-sigma limits CH4 < -5.6, NH3 < -4.2
    "WASP-96b": {"CH4": A, "H2O": P, "CO2": A, "NH3": A},             # Radica et al. 2026 (POSEIDON free, SOSS + G395H): H2O -2.61 +0.27/-0.30; CO2 -4.65 +0.34/-0.39; CH4 < -5.91; NH3 < -5.47
    "WASP-17b": {"CH4": A, "H2O": P, "CO2": A, "NH3": None},          # Louie et al. 2025 (POSEIDON free, full data): H2O -2.96 +0.31/-0.24; CO2 -5.80 +0.62/-1.31; CH4 -9.06 +2.04; NH3 not reported
    # third batch, held out: labels fixed from the papers on 2026-10-29 before any verdict was computed (same rule)
    "WASP-94Ab": {"CH4": A, "H2O": P, "CO2": None, "NH3": A},         # Ahrer et al. 2025 (HyDRA free, Eureka! R = 400): H2O -1.60 +0.43/-0.40; CO2 -4.25 +0.74/-0.54 straddles; CH4 < -6.29, NH3 < -5.10 (2 sigma)
    "HIP67522b": {"CH4": None, "H2O": None, "CO2": A, "NH3": None},   # Thao et al. 2024 (free): H2O -3.56 +0.81/-0.96 straddles; CO2 -5.56 +0.81/-0.95; CH4, NH3 not reported
    "HAT-P-12b": {"CH4": A, "H2O": P, "CO2": None, "NH3": A},         # Heinke et al. 2026 (NIRISS + NIRSpec free): H2O -1.82 +0.38/-0.44; CO2 -3.65 +0.50/-0.53 straddles; CH4 < -6.33, NH3 < -6.57 (95 %)
}
TENTATIVE = {("GJ1214b", "CH4"), ("GJ1214b", "CO2"), ("LTT3780c", "CH4")}   # labels resting on 3-sigma-level detections; reported separately
FULL = {"TOI-270d", "K2-18b", "K2-18b-alt", "K2-18b-hu", "TOI-421b", "HAT-P-18b", "WASP-166b", "TOI-1231b", "WASP-96b", "WASP-94Ab", "HAT-P-12b"}        # optical + AIRS covered by data


def numeric_rows(path, delimiter=None, skip=0, lead=0, tolerant=False):
    """Rows of a text table that begin with a number (headers, dashes and comments skipped). `lead` text tokens are
    dropped from the front of each row first; `tolerant` keeps the leading numeric run of a row that ends in text."""
    rows = []
    for k, line in enumerate(open(path, errors="replace")):
        if k < skip:
            continue
        s = line.strip().replace(",", " ") if delimiter == "," else line.strip()
        if not s or s[0] == "#":
            continue
        tok = s.split()[lead:]
        if len(tok) < 2:
            continue
        try:
            rows.append([float(t) for t in tok])
        except ValueError:
            if tolerant:
                num = []
                for t in tok:
                    try: num.append(float(t))
                    except ValueError: break
                if len(num) >= 3: rows.append(num)
            continue
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], float)


def ecsv_rows(path):
    """Columns wave, waveMin, waveMax, yval, mean(yerrLow, yerrUpp) of an ECSV spectrum table (Hu et al. 2025), rows with ignore == 0."""
    import pandas as pd
    t = pd.read_csv(path, comment="#"); t = t[t.get("ignore", 0) == 0]
    return np.column_stack([t.wave, t.waveMin, t.waveMax, t.yval, 0.5 * (t.yerrLow + t.yerrUpp)]).astype(float)


def seg(a, wl, depth, err, unit=1.0, width=None, halfwidth=None, bounds=None, rprs=False, err2=None):
    """Build a segment from table columns (indices). rprs: the depth column is Rp/Rs, not (Rp/Rs)^2;
    err2: a second (asymmetric) error column, symmetrized as the mean of the two."""
    if bounds is not None:
        lo, hi = a[:, bounds[0]], a[:, bounds[1]]; w = hi - lo; x = 0.5 * (lo + hi)
    else:
        x = a[:, wl]; w = a[:, width] if width is not None else (2 * a[:, halfwidth] if halfwidth is not None else None)
    d, e = a[:, depth] * unit, (a[:, err] if err2 is None else 0.5 * (a[:, err] + a[:, err2])) * unit
    if rprs:
        d, e = d ** 2, 2 * d * e
    ok = np.isfinite(d) & np.isfinite(e) & (e > 0)
    o = np.argsort(x[ok])
    return dict(wl=x[ok][o], depth=d[ok][o], err=e[ok][o], width=None if w is None else w[ok][o])


def segments(root, planet):
    r = lambda *p: os.path.join(root, *p)
    if planet == "TOI-270d":
        d = r("TOI-270d", "Felix2025_zenodo15228454", "TOI-270d_transit_spectra")
        return {"SOSS": seg(numeric_rows(f"{d}/native_NIRISS.dat"), 0, 1, 2, 1e-6),
                "NRS1": seg(numeric_rows(f"{d}/native_NRS1.dat"), 0, 1, 2, 1e-6),
                "NRS2": seg(numeric_rows(f"{d}/native_NRS2.dat"), 0, 1, 2, 1e-6)}, "Felix et al. 2025 reanalysis (Zenodo 15228454), native resolution"
    if planet == "K2-18b":
        o = r("K2-18b", "Madhusudhan2023_osf_36djh"); d = r("K2-18b", "Jaziri2025_zenodo16277833", "Spectra")
        return {"SOSS": seg(numeric_rows(f"{o}/K2-18b_niriss_soss_native.txt"), 0, 2, 3, halfwidth=1),
                "G395H": seg(numeric_rows(f"{o}/K2-18b_nirspec_g395h_native.txt"), 0, 2, 3, halfwidth=1),
                "MIRI": seg(numeric_rows(f"{d}/K2-18b_miri_lrs_eureka_taurex.txt"), 0, 1, 2, halfwidth=3)}, "Madhusudhan et al. 2023 native SOSS + G395H (OSF 36djh); MIRI from the Jaziri et al. 2025 copy of Madhusudhan et al. 2025 (Zenodo 16277833)"
    if planet == "K2-18b-hu":
        d = r("K2-18b", "Hu2025_osf_hpu8g", "spectra")
        return {"SOSS-o1": seg(ecsv_rows(f"{d}/niriss_nameless/K2_18b_NIRISS_SOSS_order1_vis1_R300.spec"), 0, 3, 4, 1e-6, bounds=(1, 2)),
                "SOSS-o2": seg(ecsv_rows(f"{d}/niriss_nameless/K2_18b_NIRISS_SOSS_order2_vis1_R100.spec"), 0, 3, 4, 1e-6, bounds=(1, 2)),
                "G235H": seg(numeric_rows(f"{d}/nirspec_combined/K2_G235H_NRS1_NRS2_ave2_direct.dat"), 2, 3, 4, bounds=(0, 1)),
                "G395H": seg(numeric_rows(f"{d}/nirspec_combined/K2_G395H_NRS1_NRS2_ave3_direct.dat"), 2, 3, 4, bounds=(0, 1))}, "Hu et al. 2025 (OSF hpu8g): NIRISS SOSS visit 1, NIRSpec G235H (2 visits) and G395H (3 visits) combined"
    if planet == "TOI-270d-hm":
        d = r("TOI-270d", "Holmberg2024_osf_8fu36")
        return {"G395H": seg(numeric_rows(f"{d}/TOI-270d_nirspec_g395h_native.txt"), 0, 2, 3, halfwidth=1)}, "Holmberg & Madhusudhan 2024 (OSF 8fu36), canonical G395H at native resolution"
    if planet == "TOI-1231b":
        d = r("TOI-1231b", "Sarkar2026_osf_mryd6")
        f = lambda name: seg(numeric_rows(f"{d}/TOI-1231b_{name}_JexoRes.txt"), 0, 2, 3, halfwidth=1)
        return {"SOSS-o1": f("niriss_ord1"), "SOSS-o2": f("niriss_ord2"), "NRS1": f("nirspec_g395h_nrs1"), "NRS2": f("nirspec_g395h_nrs2")}, "Sarkar et al. 2026 (OSF mryd6): JexoRes NIRISS orders 1-2 and NIRSpec G395H NRS1-2"
    if planet == "TOI-421b":
        d = r("TOI-421b", "Davenport2025_zenodo15176438", "TOI421b")
        return {"SOSS": seg(numeric_rows(f"{d}/NIRISS/exoTEDRF/BD-14 1137b_NIRISS_SOSS_transmission_spectrum_Final_50pix.csv", ","), 0, 2, 3, 1e-6, halfwidth=1),
                "G395H": seg(numeric_rows(f"{d}/NIRSpec/Deming_spectrum.txt"), 0, 1, 2)}, "Davenport et al. 2025 (Zenodo 15176438): exoTEDRF SOSS, Deming G395H"
    if planet == "HAT-P-18b":
        return {"SOSS": seg(numeric_rows(r("HAT-P-18b", "Fu2022_vizier_J_ApJ_940_L35", "table1.dat")), 0, 1, 2, 1e-6)}, "Fu et al. 2022 (VizieR J/ApJ/940/L35), NIRISS SOSS"
    if planet == "WASP-107b":
        d = r("WASP-107b", "FinalResults", "FinalSpectra")
        return {"F322W2": seg(numeric_rows(f"{d}/NIRCam_F322W2_Fiducial_Eureka_Spectrum.txt"), 0, 2, 3, halfwidth=1),
                "F444W": seg(numeric_rows(f"{d}/NIRCam_F444W_Fiducial_Eureka_Spectrum.txt"), 0, 2, 3, halfwidth=1),
                "G395H": seg(numeric_rows(r("WASP-107b", "Sing_2024_Fig2_WASP107b_transit_spec_data.csv"), ","), 0, 2, 3, halfwidth=1, rprs=True),
                "MIRI": seg(numeric_rows(f"{d}/MIRI_LRS_Fiducial_Eureka_Spectrum.txt"), 0, 2, 3, halfwidth=1)}, "Welbanks et al. 2024 NIRCam + MIRI (Zenodo 10780448), Sing et al. 2024 G395H (Zenodo 10891400)"
    if planet == "HD189733b":
        return {"NIRCam": seg(numeric_rows(r("HD189733b", "zenodo", "Data", "transit spectrum", "HD189733b_NIRCam_spectrum.txt")), 0, 2, 3, halfwidth=1)}, "Fu et al. 2024 (Zenodo 11459715), NIRCam"
    if planet == "HAT-P-26b":
        return {"G395H": seg(numeric_rows(r("HAT-P-26b", "Gressier2025_IOP_figdata", "f2", "dbf2a.txt")), 0, 1, 2)}, "Gressier et al. 2025 (AJ figure data), G395H ExoTiC-JEDI"
    if planet == "K2-18b-alt":
        d = r("K2-18b", "schmidt", "K2-18_zenodo")
        f = lambda sub, name: seg(numeric_rows(f"{d}/{sub}/exoTEDRF/{name}"), 0, 2, 3, halfwidth=1)
        return {"SOSS-o1": f("NIRISS Spectra/Order 1", "K2-18b_NIRISS_SOSS_Ord1_exoTEDRF_R100.dat"), "SOSS-o2": f("NIRISS Spectra/Order 2", "K2-18b_NIRISS_SOSS_Ord2_exoTEDRF_R100.dat"),
                "NRS1": f("NIRSpec Spectra/NRS1", "K2-18b_NIRSpec_G395H_NRS1_exoTEDRF_R100.dat"), "NRS2": f("NIRSpec Spectra/NRS2", "K2-18b_NIRSpec_G395H_NRS2_exoTEDRF_R100.dat")}, "Schmidt & MacDonald 2025 reanalysis (Zenodo 14735689): exoTEDRF at R = 100"
    if planet == "WASP-166b":
        d = r("WASP-166b", "extracted")
        import glob
        n = glob.glob(f"{d}/**/S6_wasp166_231231_5SigmaCorrNirissO1_ap0_bg0_rp^2_Table_Save.txt", recursive=True)[0]
        g = glob.glob(f"{d}/**/S6_wasp166b_NIRSpec_ap4_bg8_rp^2_Table_Save*v81*.txt", recursive=True)[0]
        # the table's "bin_width" is a half-width: consecutive points step by twice its value
        return {"SOSS": seg(numeric_rows(n), 0, 2, 3, halfwidth=1, err2=4), "G395M": seg(numeric_rows(g), 0, 2, 3, halfwidth=1, err2=4)}, "Mayo et al. 2025 (Zenodo 15122285): Eureka! NIRISS order 1 and NIRSpec G395M, 0.018 um bins"
    if planet == "GJ3470b":
        d = r("GJ3470b")
        return {"F322W2": seg(numeric_rows(f"{d}/Beatty2024_apjlad55e9t3_mrt.txt"), None, 2, 3, 1e-6, bounds=(0, 1)),
                "F444W": seg(numeric_rows(f"{d}/Beatty2024_apjlad55e9t4_mrt.txt"), None, 2, 3, 1e-6, bounds=(0, 1))}, "Beatty et al. 2024 (ApJL machine-readable tables), NIRCam"
    if planet == "GJ9827d":
        rows = [l.split() for l in open(r("GJ9827d", "PiauletGhorayeb2024_apjl_ad6f00_mrt", "apjlad6f00t4_mrt.txt")) if l.startswith("NIRISS_SOSS")]   # Inst Min Max Depth +err -err
        o2 = np.array([x[1:] for x in rows if x[0] == "NIRISS_SOSS2"], float); o1 = np.array([x[1:] for x in rows if x[0] == "NIRISS_SOSS1"], float)
        return {"SOSS-o2": seg(o2, None, 2, 3, 1e-6, bounds=(0, 1), err2=4), "SOSS-o1": seg(o1, None, 2, 3, 1e-6, bounds=(0, 1), err2=4)}, "Piaulet-Ghorayeb et al. 2024 (ApJL machine-readable table A2), supreme-SPOON NIRISS SOSS at R = 100"
    if planet == "WASP-52b":
        import pandas as pd
        t = pd.read_csv(r("WASP52b", "Pan2026_zenodo18734230", "data_behind_figures.csv")); t = t[t.dataset.str.startswith("transitspectroscopy")]
        f = lambda d: seg(t[t.dataset == d][["wavelength", "half_bin_width", "transit_depth", "error"]].to_numpy(float), 0, 2, 3, halfwidth=1)
        return {"NRS1": f("transitspectroscopy_NRS1"), "NRS2": f("transitspectroscopy_NRS2")}, "Pan et al. 2026 (Zenodo 18734230), NIRSpec G395H, transitspectroscopy reduction"
    if planet == "L98-59d":
        return {"G395H": seg(numeric_rows(r("L98-59d", "Gressier2024_apjl_ad73d1_mrt", "apjlad73d1t3_mrt.dat")), 0, 1, 2, halfwidth=3)}, "Gressier et al. 2024 (ApJL machine-readable table 3), NIRSpec G395H, transitspectroscopy at R ~ 100"
    if planet == "GJ1214b":
        return {"G395H": seg(numeric_rows(r("GJ1214b", "Schlawin2024_apjl_ad7fef_fig2data", "fig2_spec_fig2_tshirt.mrt"), tolerant=True), 0, 2, 3, width=1)}, "Schlawin et al. 2024 (ApJL figure 2 data), NIRSpec G395H, tshirt reduction at R = 100"
    if planet == "LTT3780c":
        d = r("TOI-732c", "Rigby2025_osf_wsjmu"); f = lambda name: seg(numeric_rows(f"{d}/TOI-732c_{name}.txt"), 0, 2, 3, halfwidth=1)
        return {"SOSS": f("niriss"), "NRS1": f("nirspec_g395h_nrs1"), "NRS2": f("nirspec_g395h_nrs2"), "MIRI": f("miri_JexoRes")}, "TOI-732 c, Rigby et al. 2025 (OSF wsjmu): JexoRes NIRISS order 1, NIRSpec G395H and MIRI LRS at native resolution"
    if planet == "WASP-80b":
        import pandas as pd
        x = pd.read_excel(r("WASP-80b", "Bell2023_nature_s41586-023-06687-0_sourcedata", "41586_2023_6687_MOESM5_ESM.xlsx"), sheet_name="Transmission_Data").to_numpy(float)
        return {"F322W2": seg(x, 0, 2, 3, halfwidth=1)}, "Bell et al. 2023 (Nature source data, Eureka!), NIRCam F322W2"
    if planet == "HD209458b":
        a = numeric_rows(r("HD209458b", "Xue2024_zenodo10557924", "spectra_final.csv"), ",")   # Eureka! start, end, depth %, err %, best fit; then SPARTA
        return {"NIRCam": seg(a, None, 2, 3, 1e-2, bounds=(0, 1))}, "Xue et al. 2024 (Zenodo 10557924), NIRCam F322W2 + F444W, Eureka! reduction"
    if planet == "WASP-96b":
        d = r("WASP-96b", "Radica2026_zenodo17065171", "WASP-96b_Zenodo", "Transmission_Spectra")
        f = lambda name: seg(numeric_rows(f"{d}/WASP-96b_{name}_transmission_spectrum_PriorLD_Slope_R300.csv", ","), 0, 2, 3, 1e-6, halfwidth=1)
        return {"SOSS": f("NIRISS_SOSS"), "NRS1": f("NIRSpec_G395H_NRS1"), "NRS2": f("NIRSpec_G395H_NRS2")}, "Radica et al. 2026 (Zenodo 17065171): exoTEDRF NIRISS SOSS and NIRSpec G395H at R = 300"
    if planet == "WASP-94Ab":
        d = r("WASP-94Ab", "Zenodo22803020"); f = lambda name: seg(numeric_rows(f"{d}/WASP-94_{name}_Rreference.csv", ",", skip=1), 0, 2, 3, halfwidth=1)
        return {"SOSS": f("NIRISS_SOSS_order1"), "NRS1": f("NIRSPEC_G395H_nrs1"), "NRS2": f("NIRSPEC_G395H_nrs2")}, "Zenodo 22803020 (limb-asymmetry reanalysis of the Ahrer et al. 2025 G395H transit and a NIRISS SOSS transit), limb-averaged reference spectra"
    if planet == "HIP67522b":
        return {"G395H": seg(numeric_rows(r("WASP127b", "Fu2025_zenodo13366851", "Spectra", "HIP 67522b.txt")), 0, 2, 3, width=1)}, "Thao et al. 2024 NIRSpec G395H as compiled by Fu et al. 2025 (Zenodo 13366851)"
    if planet == "HAT-P-12b":
        d = r("HAT-P-12b", "Heinke2026_zenodo19356398", "Retrievals", "data")
        def f(sub):   # ARCiS input: wavelength, depth, error, resolving power (bin width = wavelength / R)
            a = numeric_rows(f"{d}/{sub}/arcis.dat"); a = np.column_stack([a, a[:, 0] / a[:, 3]])
            return seg(a, 0, 1, 2, width=4)
        return {"SOSS": f("NIRISS/ExoTEDRF/Standard"), "G395M": f("NIRSpec/CASCADe"), "MIRI": f("MIRI/CASCADe")}, "Heinke et al. 2026 (Zenodo 19356398): exoTEDRF NIRISS SOSS (standard reduction), CASCADe NIRSpec G395M and MIRI LRS"
    if planet == "WASP-17b":
        import xarray as xr
        d = r("WASP-17b", "Louie2025_zenodo14193061", "Data_reduction_Ahsoka", "Ahsoka_Transmission_Spectra")
        f = lambda o: seg(numeric_rows(f"{d}/wasp17b_visit101_transit_SOSSorder{o}_R100_LDfitAtPixelLevel_Ahsoka.txt"), 0, 2, 3, 1e-6, halfwidth=1)
        m = xr.open_dataset(r("WASP-17b", "Grant2023_zenodo8360121", "ExoTiC-MIRI", "transmission_spectrum_vfinal_bin0.25_utc20230606_125313.nc"))
        mi = np.column_stack([m.central_wavelength.values, m.bin_half_width.values, m.transit_depth.values, m.transit_depth_error.values])
        return {"SOSS-o1": f(1), "SOSS-o2": f(2), "MIRI": seg(mi, 0, 2, 3, halfwidth=1)}, "Louie et al. 2025 (Zenodo 14193061) Ahsoka NIRISS SOSS at R = 100; Grant et al. 2023 (Zenodo 8360121) ExoTiC-MIRI LRS, 0.25 um bins"
    raise KeyError(planet)


# Fourth batch: the independent test of the frozen rule (REVISION_EXPECTATIONS.md, section H; survey in jwst_fourth_batch_survey.md). Labels committed before any verdict on these planets.
# Strict rule: any printed free-chemistry variant on the planet's JWST data that crosses -4 leaves the label unscored. Values are log10 VMR as printed in the source named.
FOURTH = {
    "TrES-4b": {"CH4": A, "H2O": P, "CO2": A, "NH3": None},       # Meech et al. 2025 (POSEIDON free, Tiberius R=100): CH4 < -7.07 (2 sigma); H2O -2.98 +0.68/-0.73; CO2 -6.86 +0.62/-0.65
    "HATS-6b": {"CH4": A, "H2O": A, "CO2": A, "NH3": A},          # Guzman Caloca et al. 2026 (POSEIDON free, M2.5, co-added): CH4 -5.38 +0.18/-0.19; H2O -4.88 +0.25/-0.24; CO2 -7.59 +0.41/-0.44; NH3 -6.03 +0.18/-0.19
    "TOI-5205b": {"CH4": None, "H2O": A, "CO2": A, "NH3": A},     # Canas et al. 2026 (POSEIDON free, M3.1): H2O -10.0 +1.4/-1.3; CO2 -8.6 +0.7/-1.6; NH3 -7.06 +0.44/-1.98; CH4 unscored, one variant reaches -3.74
    "TOI-6894b": {"CH4": P, "H2O": P, "CO2": None, "NH3": P},     # Zhang et al. 2026 (fully free, Appendix B): CH4 -1.67 +0.16/-0.16; H2O -1.40 +0.12/-0.15; NH3 -3.19 +0.24/-0.26; CO2 -4.18 +0.41/-0.48 straddles
    "TOI-199b": {"CH4": P, "H2O": None, "CO2": None, "NH3": None},  # Bello-Arufe et al. 2026 (ExoTR free): CH4 -2.46 +0.85/-1.15; others limits above -4
    "LP791-18c": {"CH4": P, "H2O": None, "CO2": None, "NH3": None},  # Roy et al. 2026 (SCARLET free, published Extended Data Table 3): CH4 -0.86 +0.53/-1.70; others limits above -4
    "NGTS-2b": {"CH4": A, "H2O": None, "CO2": None, "NH3": A},    # Fairman et al. 2026 (POSEIDON free, Model I): CH4 -9.83 +1.58/-1.50; NH3 -9.23 +1.92/-1.86; H2O straddles; CO2 variants cross
    "HAT-P-30b": {"CH4": A, "H2O": None, "CO2": None, "NH3": A},  # Claringbold et al. 2026 (pRT free): CH4 < -6.1 (2 sigma); H2O -3.3 +/-0.8 straddles; CO2 one variant crosses; NH3 -9.35 +1.72/-1.72 and -9.42 +1.78/-1.68 (BeAR, Fig. A4 titles; second reader)
    "HAT-P-1b": {"CH4": A, "H2O": P, "CO2": None, "NH3": None},   # Ashtari et al. 2026 (POSEIDON free): CH4 -9.55 +1.65/-1.59; H2O -1.70 +0.43/-0.71; CO2 -3.99 +0.53/-0.66 straddles; NH3 not in the model
    "TOI-1130b": {"CH4": None, "H2O": P, "CO2": None, "NH3": None},  # Barat et al. 2026 (pRT free, mass fractions): H2O -0.7 +0.3/-1.1 in mass fraction, VMR lower bound -2.58 at mean molecular weight 3
    "Kepler-51d": {"CH4": A, "H2O": None, "CO2": None, "NH3": None},  # Libby-Roberts et al. 2026 (ExoTR free): CH4 66 +72/-45 ppb
    "WASP-121b": {"CH4": None, "H2O": None, "CO2": A, "NH3": None},  # Gapp et al. 2025 (NEMESIS free): CO2 -8.64 +1.82/-1.99 (day), -8.80 +1.64/-1.89 (night); H2O variants disagree
}
FOURTH_TWINS = {"TrES-4b": {"CH4": -8.07, "H2O": -2.98, "CO2": -6.86, "NH3": -7.0}, "HATS-6b": {"CH4": -5.38, "H2O": -4.88, "CO2": -7.59, "NH3": -6.03}, "TOI-5205b": {"CH4": -5.89, "H2O": -10.0, "CO2": -8.6, "NH3": -7.06},
                "TOI-6894b": {"CH4": -1.67, "H2O": -1.40, "CO2": -4.18, "NH3": -3.19}, "TOI-199b": {"CH4": -2.46, "H2O": -3.05, "CO2": -4.47, "NH3": -2.78}, "LP791-18c": {"CH4": -0.86, "H2O": -1.71, "CO2": -2.97, "NH3": -2.09},
                "NGTS-2b": {"CH4": -9.83, "H2O": -3.51, "CO2": -7.38, "NH3": -9.23}, "HAT-P-30b": {"CH4": -7.1, "H2O": -3.3, "CO2": -8.1, "NH3": -7.0}, "HAT-P-1b": {"CH4": -9.55, "H2O": -1.70, "CO2": -3.99, "NH3": -7.0},
                "TOI-1130b": {"CH4": -4.73, "H2O": -1.48, "CO2": -2.37, "NH3": -3.75}, "Kepler-51d": {"CH4": -7.18, "H2O": -7.0, "CO2": -7.0, "NH3": -7.0}, "WASP-121b": {"CH4": -7.0, "H2O": -5.74, "CO2": -8.64, "NH3": -7.0}}
# V1298 Tau b left out on the second reader's finding: Adams 2026 prints CH4 -4.24 against Barat 2025's -6.22, so the literature disagrees near the threshold and no label is fixed.


def fourth_segments(root, planet):
    """The fourth batch (REVISION_EXPECTATIONS.md, section H; parsing recipes from the survey of 2026-10-30). Where a paper states the range its retrieval used, the spectrum is trimmed to it."""
    r = lambda *p: os.path.join(root, *p)
    trim = lambda a, lo, hi: a[(a[:, 0] >= lo) & (a[:, 0] <= hi)]
    if planet == "TrES-4b":
        d = r("TrES-4b", "Meech2025_zenodo15097695", "transmission_spectra", "tiberius")
        return {k: seg(numeric_rows(f"{d}/tiberius_R100_transmission_spectrum_depths_{k}_23Oct24.txt"), 0, 2, 3, width=1) for k in ("NRS1", "NRS2")}, "Meech et al. 2025, principal Tiberius reduction, R = 100 (Zenodo 15097695)"
    if planet == "HATS-6b":
        return {"PRISM": seg(numeric_rows(r("HATS-6b", "GuzmanCaloca2026_zenodo21500595", "data", "BINNED_coadd.txt")), 0, 2, 3, halfwidth=1)}, "Guzman Caloca et al. 2026, Eureka co-added PRISM (Zenodo 21500595)"
    if planet == "TOI-5205b":
        return {"PRISM": seg(trim(numeric_rows(r("TOI-5205b", "Canas2026_aj_ae4976_fig2data", "toi5205_coadd_exotic_pixel.txt")), 0.6, 5.3), 0, 2, 3, halfwidth=1)}, "Canas et al. 2026, ExoTiC-JEDI co-added PRISM, 0.6-5.3 um as retrieved (AJ figure data)"
    if planet == "TOI-6894b":
        return {"PRISM": seg(trim(numeric_rows(r("TOI-6894b", "Zhang2026_apjl_aea4ae_fig2data", "dbf2.txt")), 0.8, 5.2), 0, 1, 2)}, "Zhang et al. 2026, PRISM, 0.8-5.2 um as retrieved (ApJL figure data; the table's unit says percent, but its values, near 0.15, are the fraction: the planet's transit depth is 15 %)"
    if planet == "TOI-199b":
        return {"G395M": seg(numeric_rows(r("TOI-199b", "BelloArufe2026_zenodo17653116", "transmission_spectrum_toi199_eureka_nativespectral.txt")), 0, 2, 3, halfwidth=1)}, "Bello-Arufe et al. 2026, Eureka native G395M NRS1 (Zenodo 17653116)"
    if planet == "LP791-18c":
        return {"PRISM": seg(ecsv_rows(r("LP791-18c", "Roy2026_zenodo17316562", "LP79118c", "LP79118c_fullRes_spectrum.spec")), 0, 3, 4, 1e-6, bounds=(1, 2))}, "Roy et al. 2026, Eureka and ExoTEP full-resolution PRISM (Zenodo 17316562)"
    if planet == "NGTS-2b":
        import netCDF4
        with netCDF4.Dataset(r("NGTS-2b", "Fairman2026_zenodo18864528", "ExoTiC-JEDI_data_products", "ExoTiC-JEDI_R400_transit_spectrum.nc")) as ds:
            a = np.column_stack([np.asarray(ds[k][:], float).ravel() for k in ("central_wavelength", "bin_half_width", "transit_depth", "transit_depth_error")])
        return {"G395H": seg(a, 0, 2, 3, halfwidth=1)}, "Fairman et al. 2026, ExoTiC-JEDI R = 400 G395H (Zenodo 18864528)"
    if planet == "HAT-P-30b":
        return {"G395H": seg(numeric_rows(r("HAT-P-30b", "Claringbold2026_zenodo18682265", "transmission_spectra", "Eureka_R400.dat")), 0, 2, 3, halfwidth=1, err2=4)}, "Claringbold et al. 2026, Eureka R = 400 G395H (Zenodo 18682265)"
    if planet == "HAT-P-1b":
        return {"G395H": seg(numeric_rows(r("HAT-P-1b", "Ashtari2026_aj_aea90a_fig2data", "dbf2.txt")), 0, 2, 3, 1e-6, halfwidth=1, err2=4)}, "Ashtari et al. 2026, primary Eureka G395H (AJ figure data)"
    if planet == "TOI-1130b":
        d = r("TOI-1130b", "Barat2026_apjl_ae5f8b_fig2data"); out = {}
        for k, f in (("SOSS-o1", "toi1130b_niriss_order1_fixed_ld_common_mode_spectrum_final_modelling.mrt"), ("SOSS-o2", "toi1130b_niriss_order2_fixed_ld_common_mode_spectrum_final_modelling.mrt"), ("NRS1", "toi1130b_nrs1_fixed_ld_common_mode_50pixel_spectrum.mrt"), ("NRS2", "toi1130b_nrs2_fixed_ld_common_mode_50pixel_spectrum.mrt")):
            a = numeric_rows(f"{d}/{f}"); a = np.column_stack([a, a[:, 1] + a[:, 2]]); out[k] = seg(a, 0, 3, 4, 1e-6, width=a.shape[1] - 1)
        return out, "Barat et al. 2026, exoTEDRF NIRISS and Eureka NIRSpec (ApJL figure data)"
    if planet == "V1298Taub":
        return {"G395H": seg(numeric_rows(r("V1298Taub", "Barat2025_zenodo16782958", "final spectrum", "v1298b_final_combined_spectrum")), 0, 1, 2, 1e-6)}, "Barat et al. 2025, Eureka G395H (Zenodo 16782958)"
    if planet == "Kepler-51d":
        return {"PRISM": seg(trim(numeric_rows(r("Kepler-51d", "LibbyRoberts2026_aj_ae33c0_mrt", "ajae33c0t2_mrt.txt")), 0.6, 4.5), 0, 1, 2, 1e-2, err2=3)}, "Libby-Roberts et al. 2026, Eureka native PRISM, 0.6-4.5 um as retrieved (AJ Table 2)"
    if planet == "WASP-121b":
        return {"G395H": seg(numeric_rows(r("WASP-121b", "Gapp2025_aj_ad9c6e_fig6data", "dbf6a.txt")), 0, 4, 5, 1e-2, err2=6)}, "Gapp et al. 2025, phase-curve G395H at R of about 600 (AJ figure data)"
    raise ValueError(planet)


def run_planet(root, planet, draws, variants):
    import alfnoor_jwst as AJ
    S, source = segments(root, planet); edges, labels = J.tier3_layout()
    ref = max(S, key=lambda k: len(S[k]["wl"]))
    off = J.align_offsets(S, ref)
    d, e, cov = J.combine(S, edges, off)
    filled = ~np.isfinite(d) | (cov.covered.to_numpy() < 0.5)
    d_f = d.copy(); d_f[filled] = np.nanmean(d[~filled]) if (~filled).any() else np.nan
    cov["filled"] = filled; cov["depth"] = d_f; cov["err"] = np.where(filled, np.nan, e)
    cov.to_csv(os.path.join(OUT, f"{planet}_coverage.csv"), index=False)
    sig = J.ariel_sigma(planet.replace("-alt", "").replace("-hu", "").replace("-hm", ""), edges)
    text, V = AJ.summarize(f"{planet} [{source}; {int(filled.sum())} of 104 bins flat-filled]", d_f, edges, {"tier1": sig}, n=draws)
    V.to_csv(os.path.join(OUT, f"{planet}_verdicts.csv"), index=False)
    head = [f"{planet}: {source}", "segments: " + "; ".join(f"{k} {len(s['wl'])} pts {s['wl'].min():.2f}-{s['wl'].max():.2f} um, offset {off[k]*1e6:+.0f} ppm" for k, s in S.items()),
            "coverage by channel: " + ", ".join(f"{c} {int(g.filled.sum())}/{len(g)} filled" for c, g in cov.groupby("channel"))]
    open(os.path.join(OUT, f"{planet}.txt"), "w").write("\n".join(head) + "\n" + text + "\n")
    F = AJ.tier_frequencies(d_f, sig, n=draws); bm = AJ.band_metrics(d_f, edges)
    rows = []
    for mol in MOLS:
        for v in variants:
            vv = V[(V.variant == v) & (V.threshold == "1e-4") & (V.molecule == mol)]
            n_present = int(vv.present.sum()); verdict = P if n_present >= 3 else (A if n_present <= 1 else "split")
            lab = LABELS.get(planet, {}).get(mol)
            rows.append(dict(planet=planet, molecule=mol, variant=v, label=lab, verdict=verdict, learners_present=n_present,
                             tier1_frequency=float(F[(F.variant == v) & (F.molecule == mol)].frequency.mean()),
                             band_metric=bm.get(mol, np.nan), filled_bins=int(filled.sum()), full_coverage=planet in FULL,
                             correct=None if lab is None or verdict == "split" else (verdict == lab)))
    return rows, head


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(HERE, "data", "jwst", "panel"), help="fetch_jwst_data.sh fills it")
    ap.add_argument("--planets", nargs="*", default=list(LABELS))
    ap.add_argument("--draws", type=int, default=500); a = ap.parse_args()
    variants = ("all", "no_optical", "airs_only"); allrows, notes = [], []
    for pl in a.planets:
        try:
            rows, head = run_planet(a.root, pl, a.draws, variants); allrows += rows; notes += head + [""]
            print("\n".join(head[:2]), flush=True)
        except FileNotFoundError as ex:
            notes += [f"{pl}: data not available ({ex})", ""]; print(f"{pl}: skipped, {ex}", flush=True)
    df = pd.DataFrame(allrows); df.to_csv(os.path.join(OUT, "panel_summary.csv"), index=False)
    L = [f"JWST panel through the consortium classifier (threshold 1e-4; verdict = majority of the four learners; flat fill of uncovered bins; training set {TRAIN})", ""]
    for v in variants:
        g = df[df.variant == v]; s = g[g.correct.notna()]
        L.append(f"== variant {v}: {int(s.correct.sum())}/{len(s)} scored labels correct ({100*s.correct.mean():.0f} %)"
                 + (f"; full-coverage planets only: {int(s[s.full_coverage].correct.sum())}/{int(s.full_coverage.sum())}" if s.full_coverage.any() else ""))
        for pl in sorted(g.planet.unique()):
            gp = g[g.planet == pl].set_index("molecule")
            L.append(f"  {pl:<10} " + "  ".join(f"{m}: {gp.loc[m, 'verdict']:<7}({(LABELS.get(pl, {}).get(m) or 'amb.'):<7})" for m in MOLS))
        L.append("")
    L.append("== Tier-1 detection frequency (mean over learners, all inputs) and band metric")
    g = df[df.variant == "all"]
    for pl, gg in g.groupby("planet"):
        L.append(f"  {pl:<10} " + "  ".join(f"{m}: {gg[gg.molecule == m].tier1_frequency.iloc[0]*100:3.0f}%" for m in MOLS)
                 + "   M_H2O/CH4/CO2 = " + "/".join(f"{gg[gg.molecule == m].band_metric.iloc[0]:.2f}" for m in ("H2O", "CH4", "CO2")))
    L += ["", "== data notes"] + notes
    open(os.path.join(OUT, "panel_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
