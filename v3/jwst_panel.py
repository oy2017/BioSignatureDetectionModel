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

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import jwst_readers as R  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel"); os.makedirs(OUT, exist_ok=True)
MOLS = ["CH4", "H2O", "CO2", "NH3"]
P, A = "present", "absent"
# consensus labels at 10^-4 (REVISION_EXPECTATIONS.md); None = ambiguous, not scored
LABELS = {
    "TOI-270d": {"CH4": P, "H2O": None, "CO2": P, "NH3": A},
    "K2-18b": {"CH4": P, "H2O": A, "CO2": None, "NH3": A},
    "TOI-421b": {"CH4": A, "H2O": None, "CO2": A, "NH3": None},
    "HAT-P-18b": {"CH4": A, "H2O": None, "CO2": A, "NH3": A},
    "WASP-107b": {"CH4": A, "H2O": P, "CO2": None, "NH3": A},
    "HD189733b": {"CH4": A, "H2O": None, "CO2": A, "NH3": A},
    "HAT-P-26b": {"CH4": A, "H2O": P, "CO2": P, "NH3": A},
    "GJ3470b": {"CH4": None, "H2O": P, "CO2": P, "NH3": A},
    "WASP-166b": {"CH4": A, "H2O": P, "CO2": P, "NH3": None},
}
FULL = {"TOI-270d", "K2-18b", "TOI-421b", "HAT-P-18b", "WASP-166b"}        # optical + AIRS covered by data


def numeric_rows(path, delimiter=None, skip=0):
    """Rows of a text table that begin with a number (headers, dashes and comments skipped)."""
    rows = []
    for k, line in enumerate(open(path, errors="replace")):
        if k < skip:
            continue
        s = line.strip().replace(",", " ") if delimiter == "," else line.strip()
        if not s or s[0] == "#":
            continue
        tok = s.split()
        try:
            rows.append([float(t) for t in tok])
        except ValueError:
            continue
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], float)


def seg(a, wl, depth, err, unit=1.0, width=None, halfwidth=None, bounds=None, rprs=False):
    """Build a segment from table columns (indices). rprs: the depth column is Rp/Rs, not (Rp/Rs)^2."""
    if bounds is not None:
        lo, hi = a[:, bounds[0]], a[:, bounds[1]]; w = hi - lo; x = 0.5 * (lo + hi)
    else:
        x = a[:, wl]; w = a[:, width] if width is not None else (2 * a[:, halfwidth] if halfwidth is not None else None)
    d, e = a[:, depth] * unit, a[:, err] * unit
    if rprs:
        d, e = d ** 2, 2 * d * e
    ok = np.isfinite(d) & np.isfinite(e) & (e > 0)
    o = np.argsort(x[ok])
    return dict(wl=x[ok][o], depth=d[ok][o], err=e[ok][o], width=None if w is None else w[ok][o])


def segments(root, planet):
    r = lambda *p: os.path.join(root, *p)
    if planet == "TOI-270d":
        d = r("TOI-270d", "Felix2025_zenodo15228454", "TOI-270d_transit_spectra")
        return {"SOSS": seg(numeric_rows(f"{d}/native_NIRISS.dat"), 0, 1, 2, 1e-6, width=3),
                "NRS1": seg(numeric_rows(f"{d}/native_NRS1.dat"), 0, 1, 2, 1e-6, width=3),
                "NRS2": seg(numeric_rows(f"{d}/native_NRS2.dat"), 0, 1, 2, 1e-6, width=3)}, "Felix et al. 2025 reanalysis (Zenodo 15228454), native resolution"
    if planet == "K2-18b":
        d = r("K2-18b", "Jaziri2025_zenodo16277833", "Spectra")
        return {"NIRISS+NIRSpec": seg(numeric_rows(f"{d}/K2-18b_both_offset1_-41ppm_native.txt"), 0, 1, 2),
                "MIRI": seg(numeric_rows(f"{d}/K2-18b_miri_lrs_eureka_taurex.txt"), 0, 1, 2, halfwidth=3)}, "Jaziri et al. 2025 copies of the Madhusudhan et al. 2023/2025 spectra (Zenodo 16277833)"
    if planet == "TOI-421b":
        d = r("TOI-421b", "Davenport2025_zenodo15176438", "TOI421b")
        return {"SOSS": seg(numeric_rows(f"{d}/NIRISS/exoTEDRF/BD-14 1137b_NIRISS_SOSS_transmission_spectrum_Final_50pix.csv", ","), 0, 2, 3, 1e-6, halfwidth=1),
                "G395H": seg(numeric_rows(f"{d}/NIRSpec/Deming_spectrum.txt"), 0, 1, 2)}, "Davenport et al. 2025 (Zenodo 15176438): exoTEDRF SOSS, Deming G395H"
    if planet == "HAT-P-18b":
        return {"SOSS": seg(numeric_rows(r("HAT-P-18b", "Fu2022_vizier_J_ApJ_940_L35", "table1.dat")), 0, 1, 2, 1e-6)}, "Fu et al. 2022 (VizieR J/ApJ/940/L35), NIRISS SOSS"
    if planet == "WASP-107b":
        d = r("WASP-107b", "FinalResults", "FinalSpectra")
        return {"F322W2": seg(numeric_rows(f"{d}/NIRCam_F322W2_Fiducial_Eureka_Spectrum.txt"), 0, 2, 3, width=1),
                "F444W": seg(numeric_rows(f"{d}/NIRCam_F444W_Fiducial_Eureka_Spectrum.txt"), 0, 2, 3, width=1),
                "G395H": seg(numeric_rows(r("WASP-107b", "Sing_2024_Fig2_WASP107b_transit_spec_data.csv"), ","), 0, 2, 3, halfwidth=1, rprs=True),
                "MIRI": seg(numeric_rows(f"{d}/MIRI_LRS_Fiducial_Eureka_Spectrum.txt"), 0, 2, 3, width=1)}, "Welbanks et al. 2024 NIRCam + MIRI (Zenodo 10780448), Sing et al. 2024 G395H (Zenodo 10891400)"
    if planet == "HD189733b":
        return {"NIRCam": seg(numeric_rows(r("HD189733b", "zenodo", "Data", "transit spectrum", "HD189733b_NIRCam_spectrum.txt")), 0, 2, 3, halfwidth=1)}, "Fu et al. 2024 (Zenodo 11459715), NIRCam"
    if planet == "HAT-P-26b":
        return {"G395H": seg(numeric_rows(r("HAT-P-26b", "Gressier2025_IOP_figdata", "f2", "dbf2a.txt")), 0, 1, 2)}, "Gressier et al. 2025 (AJ figure data), G395H ExoTiC-JEDI"
    if planet == "GJ3470b":
        d = r("GJ3470b")
        return {"F322W2": seg(numeric_rows(f"{d}/Beatty2024_apjlad55e9t3_mrt.txt"), None, 2, 3, 1e-6, bounds=(0, 1)),
                "F444W": seg(numeric_rows(f"{d}/Beatty2024_apjlad55e9t4_mrt.txt"), None, 2, 3, 1e-6, bounds=(0, 1))}, "Beatty et al. 2024 (ApJL machine-readable tables), NIRCam"
    raise KeyError(planet)


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
    sig = J.ariel_sigma(planet, edges)
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
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--planets", nargs="*", default=list(LABELS))
    ap.add_argument("--draws", type=int, default=500); a = ap.parse_args()
    variants = ("all", "no_optical", "airs_only"); allrows, notes = [], []
    for pl in a.planets:
        try:
            rows, head = run_planet(a.root, pl, a.draws, variants); allrows += rows; notes += head + [""]
            print("\n".join(head[:2]), flush=True)
        except FileNotFoundError as ex:
            notes += [f"{pl}: data not available ({ex})", ""]; print(f"{pl}: skipped, {ex}", flush=True)
    df = pd.DataFrame(allrows); df.to_csv(os.path.join(OUT, "panel_summary.csv"), index=False)
    L = ["JWST panel through the consortium classifier (threshold 1e-4; verdict = majority of the four learners; flat fill of uncovered bins)", ""]
    for v in variants:
        g = df[df.variant == v]; s = g[g.correct.notna()]
        L.append(f"== variant {v}: {int(s.correct.sum())}/{len(s)} scored labels correct ({100*s.correct.mean():.0f} %)"
                 + (f"; full-coverage planets only: {int(s[s.full_coverage].correct.sum())}/{int(s.full_coverage.sum())}" if s.full_coverage.any() else ""))
        t = g.pivot_table(index="planet", columns="molecule", values="verdict", aggfunc="first")
        t2 = g.pivot_table(index="planet", columns="molecule", values="label", aggfunc="first")
        for pl in t.index:
            L.append(f"  {pl:<10} " + "  ".join(f"{m}: {t.loc[pl, m]:<7}({(t2.loc[pl, m] or 'amb.'):<7})" for m in MOLS))
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
