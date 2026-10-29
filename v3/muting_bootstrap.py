"""Bootstrap intervals for the muting factors of the real JWST spectra (jwst_muting_census.py).

The muting is the clear twin's dispersion over the noise-corrected dispersion of the real spectrum,
sqrt(std^2 - mean err^2); for a low-S/N spectrum the correction removes most of the variance and the ratio is
unstable. Here the real depths are resampled with their per-bin errors (Gaussian), the corrected dispersion is
recomputed on each resample with the same twin, and the 16th-84th percentiles of the muting are reported, with
the fraction of resamples in which the muting exceeds the population's maximum at those bulk parameters. A
resample carries the original noise and the added draw, so its correction subtracts twice the noise variance.

Usage: python muting_bootstrap.py [--boot 2000]   -> results/jwst_panel/muting_bootstrap.{txt,csv}
"""
import argparse, os, sys
import numpy as np, pandas as pd
V3 = os.path.dirname(os.path.abspath(__file__)); os.chdir(V3); sys.path.insert(0, V3)
from common import SEED  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
    C = pd.read_csv("results/jwst_panel/muting_census.csv"); rng = np.random.default_rng(SEED)
    rows, L = [], [f"Bootstrap of the muting factors ({a.boot} resamples of the real depths with their per-bin errors); muting = clear twin / noise-corrected real dispersion",
                   f"{'planet':<11} {'bins':>4} {'muting':>7} {'16th':>6} {'84th':>6} {'max in prior':>12} {'P(muting > max in prior)':>24}"]
    for r in C.itertuples():
        pl = r.planet; covf = "results/wasp39/coverage_benchmark.csv" if pl == "WASP-39b" else f"results/jwst_panel/{pl}_coverage.csv"
        cov = pd.read_csv(covf); meas = ~cov.filled.to_numpy(bool); real = cov.depth.to_numpy()[meas]; err = cov.err.to_numpy()[meas]
        noise = np.sqrt(np.nanmean(err ** 2)) * 1e6
        def muting(x, clear, k=2.0):
            s = np.std(x) * 1e6; sc = np.sqrt(max(s ** 2 - k * noise ** 2, 0.0)); return clear / sc if sc > 0 else np.inf
        B = np.array([muting(real + rng.normal(0, 1, len(real)) * err, r.clear_twin) for _ in range(a.boot)])
        lo, hi = np.nanpercentile(B[np.isfinite(B)], [16, 84]) if np.isfinite(B).any() else (np.nan, np.nan)
        p_beyond = float(np.mean(B > r.max_muting_in_prior))
        rows.append(dict(planet=pl, bins=int(meas.sum()), muting=r.muting, lo=lo, hi=hi, max_in_prior=r.max_muting_in_prior, p_beyond_prior=p_beyond, n_inf=int((~np.isfinite(B)).sum())))
        L.append(f"{pl:<11} {int(meas.sum()):>4} {r.muting:7.2f} {lo:6.2f} {hi:6.2f} {r.max_muting_in_prior:12.2f} {p_beyond:24.2f}" + (f"   ({rows[-1]['n_inf']} resamples with no excess over noise)" if rows[-1]["n_inf"] else ""))
    pd.DataFrame(rows).to_csv("results/jwst_panel/muting_bootstrap.csv", index=False)
    open("results/jwst_panel/muting_bootstrap.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
