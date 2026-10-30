"""POP-I under physically confined hazes (review of 2026-10-30, REVISION_FINDINGS F23; expectations F/G in REVISION_EXPECTATIONS.md).

Every earlier haze filled the column at one number density up to the model's top boundary. Rendered here with the same planets,
decks and gases: a Lee-Mie haze of the same particles and density below 1 Pa only, and a power-law haze at constant mixing ratio
of slope 4 and log10 sigma_0(1 um) -29 and -28, about 10^3 and 10^4 times the H2 Rayleigh opacity. Scored by pipeline_sim.py.

Writes data/alfnoor_faithful/pop1_native_{hazetop1,plhaze29,plhaze28}.npy and results/haze_confined.txt.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import alfnoor_faithful as AF  # noqa: E402
from bin_spectra import bin_native  # noqa: E402

HAZES = {"hazetop1": dict(radius=0.1, q=40, mix_ratio=3e7, topP=1.0), "plhaze29": dict(log_sigma0=-29.0, alpha=4.0), "plhaze28": dict(log_sigma0=-28.0, alpha=4.0)}


def _work(rows, wl, haze):
    import warnings; warnings.filterwarnings("ignore")
    import forward_model as FM, shift_aerosol as A
    return [FM.spectrum(r, wl, A.GASES, AF.FILL, cloud_pressure=float(r["cloud_pressure"]), haze=haze) for r in rows]


def main(jobs=2):
    src = AF.DIRS["faithful"]; wl = np.load("data/native_wl.npy"); e = AF.layout_edges("tier3_r20"); cen = 0.5 * (e[1:] + e[:-1])
    P = pd.read_parquet(os.path.join(src, "pop1_params.parquet")); X0 = np.load(os.path.join(src, "pop1_native.npy")).astype(float); ok = np.isfinite(X0).all(1)
    rows = [P.iloc[i].to_dict() for i in range(len(P))]; chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]; L = []
    Bc = bin_native(X0[ok], wl, e); opt, nir, air = cen < 1.1, (cen > 1.1) & (cen < 1.95), cen > 1.95
    for name, haze in HAZES.items():
        f = os.path.join(src, f"pop1_native_{name}.npy"); t0 = time.time()
        if not os.path.exists(f):
            res = Parallel(n_jobs=jobs)(delayed(_work)(c, wl, haze) for c in chunks); X = np.full((len(rows), len(wl)), np.nan, dtype=np.float32)
            for i, s in enumerate(s for c in res for s in c):
                if s is not None: X[i] = s
            np.save(f, X)
        X = np.load(f).astype(float); good = np.isfinite(X).all(1) & ok; d = (bin_native(X[good], wl, e) - bin_native(X0[good], wl, e)) * 1e6
        q = lambda m: " / ".join(f"{np.percentile(d[:, m].mean(1), p):.0f}" for p in (10, 50, 90))
        L.append(f"{name:<9} {haze}: {int(good.sum())} of {int(ok.sum())} rendered, {time.time() - t0:.0f} s; hazed minus own-deck depth, ppm, 10th / median / 90th percentile of planets: optical {q(opt)}; NIRSpec {q(nir)}; AIRS {q(air)}")
        print(L[-1], flush=True)
    open("results/haze_confined.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
