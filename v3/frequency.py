"""Is the repair rule confounded with the frequency content of the shift?

The rule says what augmentation can absorb is set by how many numbers the shift
draws per spectrum. An alternative reading, from the Fourier view of robustness,
is that noise augmentation helps against high-frequency corruption and not low.
If every repairable axis is also low-frequency and every unrepairable one high,
the two explanations are indistinguishable in this design.

Measures the perturbation (shifted minus clean, noise-free where possible) in the
bin-index domain and reports where its power sits.
"""
import os
import sys

import numpy as np
import pandas as pd

V2 = os.path.dirname(os.path.abspath(__file__))   # resolve beside this file, like the other scripts
sys.path.insert(0, V2)
os.chdir(V2)
from bin_spectra import bin_native  # noqa: E402
from common import DATA, TESTS, centres, configs, load_split  # noqa: E402
from noise import sigma_matrix  # noqa: E402

CFG = "ariel"
cen = centres(CFG)
edges = np.array(configs()[CFG]["edges"])
wl = np.load(os.path.join(DATA, "native_wl.npy"))

Xc = np.vstack([load_split(t, CFG)[0] for t in TESTS])
Xnf = np.vstack([np.load(os.path.join(DATA, f"{t}_{CFG}.npy")) for t in TESTS]).astype(float)
P = pd.concat([load_split(t, CFG, noisy=False)[2] for t in TESTS], ignore_index=True)
sig = np.median(sigma_matrix(Xnf, P["s temperature"].to_numpy(), cen), axis=1, keepdims=True)
rng = np.random.default_rng(0)


def rerender(case):
    Xn = np.vstack([np.load(os.path.join(DATA, f"{t}_native_{case}.npy")).astype(float) for t in TESTS])
    Xb = bin_native(Xn, wl, edges)
    return np.where(np.isfinite(Xb), Xb, Xnf) - Xnf          # noise-free perturbation


def white(snr=5):
    m = np.sqrt((15.0 / snr) ** 2 - 1.0)
    return rng.normal(0, 1, Xnf.shape) * sig * m


def correlated(snr=5, s=3.0):
    from scipy.ndimage import gaussian_filter1d
    m = np.sqrt((15.0 / snr) ** 2 - 1.0)
    z = gaussian_filter1d(rng.normal(0, 1, Xnf.shape), sigma=s, axis=1)
    z /= z.std(axis=1, keepdims=True) + 1e-12
    return z * sig * m


def ramp(s=2.0):
    amp = s * sig / np.abs(Xnf).mean(axis=1, keepdims=True)
    tilt = np.linspace(-1, 1, Xnf.shape[1])
    signs = rng.choice([-1.0, 1.0], size=(len(Xnf), 1))
    return Xnf * (signs * amp * tilt)


def recovered(fn, case):
    """The measured recovered fraction for one case, read from the CSV the repair script wrote
    (never typed in: the v2 numbers do not carry over to this grid)."""
    p = os.path.join("results", f"{CFG}_{fn}.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        key = d.case.astype(str).str.lower().str.replace(" ", "")
        r = d[key == case.lower().replace(" ", "")]
        if len(r) and np.isfinite(r.pct_of_gap.iloc[0]):
            return float(r.pct_of_gap.iloc[0])
    return np.nan


# (name, draws per spectrum, measured recovery, perturbation)
CASES = [("stellar spots 20%", 0,   recovered("augment", "tlse_spots20"), lambda: rerender("tlse_spots20")),
         ("haze 3e7",          0,   recovered("augment", "haze_3p0e7"),   lambda: rerender("haze_3p0e7")),
         ("gain ramp 2x",      1,   recovered("augment_ramp", "x2.0"),    ramp),
         ("correlated noise",  102, recovered("augment", "snr5"),         correlated),
         ("white noise",       102, recovered("augment_white", "snr5"),   white)]

L = ["Is the repair rule confounded with the frequency content of the shift?", "",
     "A Fourier account of robustness says noise augmentation helps against",
     "high-frequency corruption and hurts against low. If every repairable axis were",
     "low-frequency and every unrepairable one high, that account and the draw-count",
     "rule would be indistinguishable here. Power is measured on the perturbation",
     "(shifted minus clean) across the 102 bins.", "",
     f"{'axis':<20}{'repair':<17}{'centroid':>10}{'power>1/8':>12}"]
print(L[0])
rows = []
for name, draws, rec, fn in CASES:
    rep = f"recovers {rec:.0f}%" if np.isfinite(rec) else "recovers n/a"
    D = fn()
    D = D[np.all(np.isfinite(D), axis=1)]
    D = D - D.mean(axis=1, keepdims=True)
    F = np.abs(np.fft.rfft(D, axis=1)) ** 2
    f = np.fft.rfftfreq(D.shape[1])                       # cycles per bin
    Pw = F.mean(axis=0); Pw /= Pw.sum()
    centroid = float((Pw * f).sum())
    hi = float(Pw[f > 0.125].sum())
    # participation ratio of the mean power spectrum: how many modes carry the shift
    dof = float((Pw.sum() ** 2) / (Pw ** 2).sum())
    L.append(f"{name:<20}{rep:<17}{centroid:>10.3f}{hi:>11.0%}")
    rows.append((name, draws, rec, centroid, hi, dof))

from scipy.stats import spearmanr
ok = [r for r in rows if np.isfinite(r[2])]
rho = spearmanr([r[4] for r in ok], [r[2] for r in ok]).correlation if len(ok) >= 3 else np.nan
smooth, rough = min(ok, key=lambda r: r[4]), max(ok, key=lambda r: r[4])
few = [r[2] for r in ok if r[1] <= 1]; many = [r[2] for r in ok if r[1] > 1]
L += ["",
      "If the Fourier account held, recovery would rise with high-frequency power.",
      f"Measured Spearman(power above 1/8 cycle per bin, recovered) = {rho:+.2f} over {len(ok)} cases.",
      f"Smoothest perturbation: {smooth[0]} ({smooth[4]:.0%} high-frequency power) recovers {smooth[2]:.0f}%;",
      f"roughest: {rough[0]} ({rough[4]:.0%}) recovers {rough[2]:.0f}%.",
      f"Grouped by draw count instead: <=1 draw recovers {min(few):.0f}-{max(few):.0f}%, per-bin draws {min(many):.0f}-{max(many):.0f}%"
      + (" (non-overlapping)." if max(many) < min(few) else " (overlapping)."),
      ""]
import os as _os
_p = _os.path.join("results", "ariel_frequency.txt")
open(_p, "w").write("\n".join(L) + "\n")
print("\n".join(L)); print("wrote", _p)
