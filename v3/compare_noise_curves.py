"""Compare the ExoSim2 per-star noise curves with the ExoRad curves v2 used.

For every host T_eff on the shared grid: Spearman of the per-bin noise-to-signal shape,
the median level ratio, and the ratio's spread across bins (a constant ratio means a
normalisation difference; a varying one means a shape difference the SNR convention
cannot absorb). Also checks whether the level ratio itself trends with T_eff.

Usage: python compare_noise_curves.py
Reads  results/exosim_nsr_curves.npz and ../ariel_noise_model/ariel_nsr_curves.npz
Writes results/noise_curves_comparison.txt
"""
import os, numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
ES = np.load(os.path.join(HERE, "results", "exosim_nsr_curves.npz"))
ER = np.load(os.path.join(os.path.dirname(HERE), "ariel_noise_model", "ariel_nsr_curves.npz"))

L = ["ExoSim2 (reconstructed Ariel-like payload) vs ExoRad curves used in v2, per host T_eff", "",
     f"{'T_eff':>6} {'bins':>5} {'Spearman':>9} {'ratio med':>10} {'ratio 5-95%':>14}"]
rows = []
for T in ES["teffs"].astype(int):
    k = f"nsr_{T}"
    if k not in ES or k not in ER: continue
    wl, nsr = ES[f"wl_{T}"], ES[k]; ok = np.isfinite(nsr) & (nsr > 0)
    ref = np.interp(wl[ok], ER[f"wl_{T}"], ER[k]); r = nsr[ok] / ref
    rho = spearmanr(ref, nsr[ok]).statistic
    rows.append((T, ok.sum(), rho, np.median(r), np.percentile(r, 5), np.percentile(r, 95)))
    L.append(f"{T:>6} {ok.sum():>5} {rho:>9.3f} {np.median(r):>10.2f} {np.percentile(r,5):>6.2f}-{np.percentile(r,95):<6.2f}")
rho = np.array([x[2] for x in rows]); med = np.array([x[3] for x in rows]); Ts = np.array([x[0] for x in rows])
L += ["", f"shape agreement: Spearman min {rho.min():.3f}, median {np.median(rho):.3f}, max {rho.max():.3f} over {len(rows)} stars",
      f"level ratio ExoSim2/ExoRad: median {np.median(med):.2f}, range {med.min():.2f}-{med.max():.2f}",
      f"does the level ratio trend with T_eff?  Spearman(T_eff, ratio) = {spearmanr(Ts, med).statistic:+.3f}",
      "", "Reading: a high per-star Spearman with a ratio that is nearly constant across bins AND across",
      "T_eff means the two noise models differ by one normalisation, which the study's SNR convention",
      "sets anyway. A ratio that varies across bins is a shape difference and is what the ExoSim2 noise",
      "axis will measure the classifier's sensitivity to."]
txt = "\n".join(L); print(txt)
open(os.path.join(HERE, "results", "noise_curves_comparison.txt"), "w").write(txt + "\n")
