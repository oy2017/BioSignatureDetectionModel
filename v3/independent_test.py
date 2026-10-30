"""Real spectra unseen by any design (REVISION_EXPECTATIONS.md, section K). Normalized spectra are copied to results/independent_test/spectra/; the survey and SOURCES.md give the origin.

Writes results/independent_test/{independent_test.csv, independent_test.txt}.
"""
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import amplitude_flag as AF  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402

OUT = os.path.join(RESULTS, "independent_test"); SRC = os.path.join("data", "independent"); P, A = "present", "absent"
# reduction -> (folder, files, MCS name or None, labels)
LABELS = {
    "V1298Tauc-JWST": ("V1298_Tau_c", ["JWST_SOSS_Murphy2026_Case2_R300.csv"], None, dict(CH4=A, H2O=P, CO2=A, NH3=A)),
    "HAT-P-14b-JWST": ("HAT-P-14_b", ["JWST_SOSS_Liu2025.csv", "JWST_G395H_Liu2025.csv"], None, dict(CH4=A, CO2=A)),
    "HAT-P-32b": ("HAT-P-32_b", ["HST_G141_T18_high.csv"], "HAT-P-32b", dict(H2O=P)),
    "HAT-P-41b": ("HAT-P-41_b", ["HST_G141_T18_high.csv"], "HAT-P-41b", dict(H2O=P)),
    "HD106315c": ("HD_106315_c", ["HST_G141_Guilluy2021.csv"], "HD106315c", dict(H2O=P)),
    "V1298Tauc-HST": ("V1298_Tau_c", ["HST_G141_E23.csv"], None, dict(H2O=P)),
    "WASP-62b": ("WASP-62_b", ["HST_G141_Skaf2020.csv"], "WASP-62b", dict(H2O=P)),
    "WASP-79b": ("WASP-79_b", ["HST_G141_Skaf2020.csv"], "WASP-79b", dict(H2O=P)),
    "GJ436b": ("GJ_436_b", ["HST_G141_T18_high.csv"], None, dict(H2O=A)),
    "HAT-P-2b": ("HAT-P-2_b", ["HST_G141_E23.csv"], "HAT-P-2b", dict(H2O=A)),
    "HIP41378f": ("HIP_41378_f", ["HST_G141_E23.csv"], "HIP41378f", dict(H2O=A)),
    "K2-24b": ("K2-24_b", ["HST_G141_E23.csv"], "K2-24b", dict(H2O=A)),
    "WASP-103b": ("WASP-103_b", ["HST_G141_E23.csv"], "WASP-103b", dict(H2O=A)),
}
# planets outside the Ariel sample: mass and radius in Earth units, temperature, stellar radius in solar units (SOURCES.md)
EXTRA = {"V1298Tauc": (4.7, 5.08, 953.0, 1.32), "HAT-P-14b": (3.44 * 317.83, 1.42 * 11.209, 1624.0, 1.82), "GJ436b": (0.08 * 317.83, 0.366 * 11.209, 686.0, 0.455)}
DES = ["published", "AIRS-only", "shape flag", "haze+deck-retrained", "the method"]


def modulation(mass, rad, T, rs):
    Rp = rad * 6.371e6; g = 6.674e-11 * mass * 5.972e24 / Rp ** 2; return 5 * 2 * (1.38e-23 * T / (2.3 * 1.67e-27 * g)) * Rp / (rs * 6.957e8) ** 2


def spectrum(folder, files, edges):
    S = {}
    for f in files:
        d = pd.read_csv(os.path.join(SRC, folder, f)); shutil.copy(os.path.join(SRC, folder, f), os.path.join(OUT, "spectra", f"{folder}_{f}")); w = 2 * d.half_width_um.to_numpy() if d.half_width_um.notna().all() else None
        S[f] = dict(wl=d.wl_um.to_numpy(), depth=d.depth_fraction.to_numpy(), err=d.err_fraction.to_numpy(), width=w)
    ref = max(S, key=lambda k: len(S[k]["wl"])); off = J.align_offsets(S, ref) if len(S) > 1 else {ref: 0.0}; off = {k: off.get(k, 0.0) for k in S}; d, e, cov = J.combine(S, edges, off)
    filled = ~np.isfinite(d) | (cov.covered.to_numpy() < 0.5); x = d.copy(); x[filled] = np.nanmean(d[~filled]); return x, np.where(filled, np.nan, e), filled


def main():
    import panel_tier1 as PT
    Xc, Xo, mod, chk, edges = AF.populations(); ctx = PT.Context(); Xtr, Ptr = ML.training(); rows = []; ratio = lambda X, m: mod / (X[:, m].std(1) * 1e6)
    for i, (name, (folder, files, mcs, labs)) in enumerate(LABELS.items()):
        x, err, filled = spectrum(folder, files, edges); meas = ~filled; base = mcs or name.replace("-JWST", "").replace("-HST", "")
        if mcs is None:
            a = EXTRA[base]; ctx.M.loc[base, ["modulation_5H", "p_mass", "p_radius", "atm temperature", "s radius"]] = [modulation(*a), a[0], a[1], a[2], a[3]]
        md = float(ctx.M.loc[base, "modulation_5H"]); hi = ratio(Xc, meas).max(); lo = np.percentile(ratio(Xo, meas), 1); j2 = float(np.nanmean(err[meas] ** 2)) * 1e12
        res = PT.score(ctx, x[None, :], meas, base, j2, hi, lo)
        for m, lab in labs.items():
            votes, low, up = res[m]
            for d in DES:
                pr = votes[d][0] >= 3; ab = votes[d][0] <= 1
                if d == "the method" and low[0]: pr = ab = False
                sp = not (pr or ab); rows.append(dict(planet=name, unit=base, molecule=m, label=lab, design=d, verdict="P" if pr else "A" if ab else "S", credit=float(pr if lab == P else ab) + 0.5 * sp, bins=int(meas.sum()), up=bool(up[0]), low=bool(low[0])))
        for d in ("matched", "matched+amp"):
            for r in ML.fit_one(name, labs, x, err, filled, np.zeros_like(x), Xtr, Ptr, md, "JWST", d, SEED + 2700 + i):
                rows.append(dict(planet=name, unit=base, molecule=r["molecule"], label=r["label"], design=d, verdict="P" if r["present"] else "A" if r["absent"] else "S", credit=r["credit"], bins=int(meas.sum()), up=False, low=False))
        print(name, int(meas.sum()), flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "independent_test.csv"), index=False); L = []; rng = np.random.default_rng(405); D = DES + ["matched", "matched+amp"]
    for tag, q in (("all 17 labels", R), ("HST water, 11 labels", R[~R.planet.str.endswith("JWST")]), ("JWST, 6 labels", R[R.planet.str.endswith("JWST")])):
        L.append(f"== {tag}")
        for d in D:
            g = q[q.design == d]; pr, ab = g[g.label == P].credit, g[g.label == A].credit; x_ = (g.groupby("unit").credit.sum() - q[q.design == "the method"].groupby("unit").credit.sum()).to_numpy(); b = x_[rng.integers(0, len(x_), (20000, len(x_)))].sum(1)
            L.append(f"  {d:<22} score {g.credit.sum():5.1f}/{len(g)}  present {pr.sum():4.1f}/{len(pr)}  absent {ab.sum():4.1f}/{len(ab)}  balanced {.5 * (pr.mean() + ab.mean()):.3f}  vs method {x_.sum():+4.1f} ({np.percentile(b, 2.5):+.1f},{np.percentile(b, 97.5):+.1f})")
    L += ["", R.pivot_table(index=["planet", "molecule", "label", "bins"], columns="design", values="verdict", aggfunc="first")[D].to_string()]
    t = "\n".join(L); open(os.path.join(OUT, "independent_test.txt"), "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
