"""The power-law repair of hazed spectra (identifiability_calibrate.py --departure haze3e7 --aerosol powerlaw), on the planets it completed, beside the clear-model and free-haze retrievals of the same spectra. Writes results/powerlaw_summary.txt."""
import os, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); MOLS = ["CH4", "H2O", "CO2", "NH3"]
load = lambda f: pd.read_csv(os.path.join(R, f)).drop_duplicates("name").set_index("name")
P = load("identifiability_calibration_haze3e7_powerlaw.csv"); P = P[P["error"].isna()] if "error" in P else P
F = load("identifiability_calibration_haze3e7_freehaze.csv").reindex(P.index); C = load("identifiability_calibration_haze3e7.csv").reindex(P.index); Cl = load("identifiability_calibration.csv").reindex(P.index)
L = [f"Hazed spectra (3e7 m^-3, Lee-Mie, column-filling), {len(P)} planets of the fixed random 200 completed; the retrieval's aerosol is a power-law opacity with two free parameters, a family that does not contain the truth", ""]
L.append(f"{'':<34}" + "".join(f"{m:>18}" for m in MOLS))
def row(lab, f):
    L.append(f"{lab:<34}" + "".join(f"{f(m):>18}" for m in MOLS))
for name, D in (("clean spectra, clear model", Cl), ("hazed, clear model", C), ("hazed, power law", P), ("hazed, haze density free", F)):
    L.append(name)
    row("  accuracy (p_true > 0.5) %", lambda m: f"{100 * np.nanmean(D[f'{m}_p_true'] > .5):.1f}")
    row("  recall of present labels %", lambda m: f"{100 * np.nanmean((D[f'{m}_p_true'] > .5)[D[f'{m}_label'] == 1]):.1f}")
    row("  decidable (p_true > 0.95) %", lambda m: f"{100 * np.nanmean(D[f'{m}_p_true'] > .95):.1f}")
    row("  p_true < 0.05 %", lambda m: f"{100 * np.nanmean(D[f'{m}_p_true'] < .05):.1f}")
    L.append(f"  median log Z {np.nanmedian(D.logz):.1f}; median minutes {np.nanmedian(D.minutes):.0f}; median calls {np.nanmedian(D.ncall):.0f}")
for m in ("CH4", "H2O"):
    pres = Cl[f"{m}_label"] == 1; dec = Cl[f"{m}_p_true"] > .95; idx = pres & dec
    L.append(f"{m} present and decidable on the clean spectrum ({int(idx.sum())}): found (p_true > 0.5) clear model {int((C[f'{m}_p_true'] > .5)[idx].sum())}, power law {int((P[f'{m}_p_true'] > .5)[idx].sum())}, free haze {int((F[f'{m}_p_true'] > .5)[idx].sum())}; still decidable power law {int((P[f'{m}_p_true'] > .95)[idx].sum())}, free haze {int((F[f'{m}_p_true'] > .95)[idx].sum())}")
if "pl_alpha_med" in P: L.append(f"power-law slope median {P.pl_alpha_med.median():.2f} (prior limit 10; truth is not a power law), log sigma0 at 1 um median {P.pl_logsigma0_med.median():.2f}; share with slope above 9: {100 * (P.pl_alpha_med > 9).mean():.0f} %")
txt = "\n".join(L); open(os.path.join(R, "powerlaw_summary.txt"), "w").write(txt + "\n"); print(txt)
