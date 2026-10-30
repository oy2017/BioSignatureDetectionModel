"""Numbers quoted in the manuscript that had no result file of their own (audit of 2026-10-30). Writes results/misc_numbers.txt.

1. The haze's departure per block in units of the Tier-1 sigma on the full hazed population (noise-free, offset removed), against the sixty-planet pilot that first gave 18, 4.6 and 1.8.
2. The two transplants of Section 3.3 on simulated spectra: hazed optical points in clean spectra, clean optical points in hazed spectra (all-input design, same noise draw).
3. Mean confidence of each learner on clean and hazed spectra.
4. The payload noise model's calibration scatter against the catalogue's Tier-3 transit counts.
5. The payload noise at the Tier-1 transit count for the three planets of the consortium study's Figure 1 (HD 209458 b, GJ 1214 b and WASP-79 b).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as F  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import pipeline_sim as PS  # noqa: E402

MOLS, LEARNERS = D.MOLS, D.LEARNERS; L = []
cases, Pte, e, sig = D.test_cases(); wl = np.load(os.path.join(DATA, "native_wl.npy")); nf = PS.noise_free_cases(wl, e); cen = 0.5 * (e[1:] + e[:-1])
opt, nir, airs = cen < 1.1, (cen > 1.1) & (cen < 1.95), cen > 1.95
X0, X1 = (nf["clean"] if isinstance(nf, dict) else nf("clean")), (nf["haze3e7"] if isinstance(nf, dict) else nf("haze3e7"))
d = X1 - X0; w = 1.0 / sig ** 2; off = (d * w).sum(1, keepdims=True) / w.sum(1, keepdims=True); z = (d - off) / sig
blk = lambda m: np.sqrt(np.mean(z[:, m] ** 2, axis=1))
L += [f"1. Haze 3e7 minus clean, noise-free, sigma-weighted offset removed, rms per bin in units of the Tier-1 sigma, {len(z)} planets: median (5th to 95th percentile)"]
for nm, m in (("three optical points", opt), ("Ariel NIRSpec channel", nir), ("AIRS bins", airs), ("all 104 bins", np.ones_like(opt))):
    v = blk(m); L.append(f"   {nm:<24} {np.median(v):7.2f} ({np.percentile(v, 5):.2f} to {np.percentile(v, 95):.2f})")
sc = D.Scorer(e); yt = {m: (Pte[f"atm {m}"].to_numpy() > -4.0) for m in MOLS}; Xc, Xh = cases["clean"], cases["haze3e7"]
T1 = Xc.copy(); T1[:, opt] = Xh[:, opt]; T2 = Xh.copy(); T2[:, opt] = Xc[:, opt]
rate = lambda X, m, ks: np.mean([np.mean((sc.probs(X)["all"][(m, k)] >= 0.5)[yt[m]]) for k in ks]) * 100
L += ["", "2. Transplants, detection rate of CH4 on planets carrying it, all-input design (%)"]
Pc, Ph, P1, P2 = sc.probs(Xc), sc.probs(Xh), sc.probs(T1), sc.probs(T2)
for tag, P_ in (("clean", Pc), ("hazed", Ph), ("hazed optical points in clean spectra", P1), ("clean optical points in hazed spectra", P2)):
    r = {k: 100 * np.mean((P_["all"][("CH4", k)] >= 0.5)[yt["CH4"]]) for k in LEARNERS}; L.append(f"   {tag:<40} mean {np.mean(list(r.values())):5.1f}; " + ", ".join(f"{k} {v:.1f}" for k, v in r.items()))
    r2 = {m: np.mean([100 * np.mean((P_["all"][(m, k)] >= 0.5)[yt[m]]) for k in LEARNERS]) for m in MOLS}; L.append(f"   {'':<40} per molecule, mean of learners: " + ", ".join(f"{m} {v:.1f}" for m, v in r2.items()))
L += ["", "3. Mean confidence max(p, 1-p) over molecules"]
for k in ("KNN", "MLP", "RFC"):
    f = lambda P_: np.mean([np.mean(np.maximum(P_["all"][(m, k)], 1 - P_["all"][(m, k)])) for m in MOLS]); L.append(f"   {k}: clean {f(Pc):.3f}, hazed {f(Ph):.3f}")
    L.append(f"   {k}, CH4 only: clean {np.mean(np.maximum(Pc['all'][('CH4', k)], 1 - Pc['all'][('CH4', k)])):.3f}, hazed {np.mean(np.maximum(Ph['all'][('CH4', k)], 1 - Ph['all'][('CH4', k)])):.3f}")
try:
    import noise as N
    M = N._mcs_hosts(); ee = F.layout_edges("tier3_r20"); teffs, absn = N.nsr_binned(np.asarray(ee, float), N.EXOSIM_NPZ); node = np.argmin(np.abs(teffs[None, :] - M["s temperature"].to_numpy()[:, None]), axis=1); out = np.vstack([absn[int(teffs[i])] for i in node])
    s = out * ((M["Star Distance [pc]"].to_numpy() / 47.5) * (1.18 / M["s radius"].to_numpy()) * np.sqrt(2.0 / (M.t14_s.to_numpy() / 3600)))[:, None]; N3 = M.tier3_transits.to_numpy(float); ok = N3 >= 5; est = np.log10(N3[ok] / (7 * np.median(s[ok], 1) / M.modulation_5H.to_numpy()[ok]) ** 2); k2 = np.median(est); res = est - k2
    L += ["", f"4. Payload noise model: targets with at least five Tier-3 transits {ok.sum()}; residual of log10 transit count about the calibration: 1.4826 x MAD {1.4826 * np.median(np.abs(res)):.3f} dex, standard deviation {res.std():.3f} dex, 16th-84th half-width {0.5 * (np.percentile(res, 84) - np.percentile(res, 16)):.3f} dex"]
except Exception as ex:
    L += ["", f"4. Payload noise model check failed: {ex}"]
import jwst_to_ariel as J  # noqa: E402
edges3, _ = J.tier3_layout(); c3 = 0.5 * (edges3[1:] + edges3[:-1]); L += ["", "5. Payload noise at the Tier-1 transit count, the three planets of Figure 1 of the consortium study, per-bin sigma in ppm: median over the AIRS bins and over Ariel's NIRSpec channel, and each optical point"]
for pl in ("HD209458b", "GJ1214b", "WASP-79b"):
    sg = J.ariel_sigma(pl, edges3) * 1e6; L.append(f"   {pl:<10} AIRS {np.median(sg[c3 > 1.95]):7.0f}   NIRSpec channel {np.median(sg[(c3 > 1.1) & (c3 < 1.95)]):7.0f}   optical " + ", ".join(f"{v:.0f}" for v in sg[c3 < 1.1]))
txt = "\n".join(L); open(os.path.join(RESULTS, "misc_numbers.txt"), "w").write(txt + "\n"); print(txt)
