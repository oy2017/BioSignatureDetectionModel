"""The randomized training grid (Test 1 of the trust programme): every ingredient drawn per planet.

Two aligned renders of the training planets with per-planet CONTINUOUS aerosol draws, one on the
equilibrium abundances and one on the quenched abundances of the same planet:
    data/train_native_rand_aero_eq.npy     data/train_native_rand_aero_q.npy
plus the draws themselves (data/train_rand_draws.parquet). Everything downstream is a per-planet
selection among {clean_eq, clean_q, aero_eq, aero_q}, a multiplicative spot factor and a noise
level, so every leave-one-axis-out variant is built without re-rendering.

Draws (seeded):
  haze_density   with prob 0.6: log-uniform 1e5 .. 3e8 m^-3, else none
  cloud_pressure with prob 0.5: log-uniform 1e3 .. 1e5 Pa (cloud top), else none
  spot_frac      with prob 0.7: uniform 0 .. 0.20, else 0          (applied at training time)
  quenched       prob 0.5                                          (selects the chemistry render)
  snr            uniform 5 .. 15                                   (applied at training time)
Ranges are continuous and deliberately wider than the discrete test levels except at the
extremes (haze 1e10, cloud 1e1/1e2 Pa stay outside: the envelope must show its edge).

Usage: python randomize_train.py --jobs 8
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, SEED  # noqa: E402
import shift_aerosol as A  # noqa: E402


def _worker(rows, wl):
    out = []
    for r in rows:
        hz, cp = r.pop("_haze"), r.pop("_cloud")
        out.append(A.one_native(r, wl, cloud_pressure=None if np.isnan(cp) else cp,
                                haze_density=None if np.isnan(hz) else hz))
    return out


def render(P, draws, wl, jobs, chunk=25):
    rows = []
    for i in range(len(P)):
        r = P.iloc[i].to_dict(); r["_haze"] = draws.haze_density.iloc[i]; r["_cloud"] = draws.cloud_pressure.iloc[i]
        rows.append(r)
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    out = Parallel(n_jobs=jobs)(delayed(_worker)(c, wl) for c in chunks)
    X = np.full((len(rows), len(wl)), np.nan, dtype=np.float32)
    for i, s in enumerate(s for c in out for s in c):
        if s is not None:
            X[i] = s
    return X


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    Peq = pd.read_parquet(os.path.join(DATA, "train_params.parquet"))
    Pq = pd.read_parquet(os.path.join(DATA, "train_params_quenched.parquet"))
    assert len(Peq) == len(Pq)
    n = len(Peq); rng = np.random.default_rng(SEED + 101)
    dp = os.path.join(DATA, "train_rand_draws.parquet")
    if os.path.exists(dp):
        draws = pd.read_parquet(dp)
    else:
        draws = pd.DataFrame(dict(
            haze_density=np.where(rng.random(n) < 0.6, 10 ** rng.uniform(np.log10(1e5), np.log10(3e8), n), np.nan),
            cloud_pressure=np.where(rng.random(n) < 0.5, 10 ** rng.uniform(3, 5, n), np.nan),
            spot_frac=np.where(rng.random(n) < 0.7, rng.uniform(0.0, 0.20, n), 0.0),
            quenched=rng.random(n) < 0.5,
            snr=rng.uniform(5.0, 15.0, n)))
        draws.to_parquet(dp)
    print(f"{n} planets: haze on {np.isfinite(draws.haze_density).mean():.0%}, cloud on {np.isfinite(draws.cloud_pressure).mean():.0%},"
          f" spots on {(draws.spot_frac > 0).mean():.0%}, quenched {draws.quenched.mean():.0%}", flush=True)
    for tag, P in (("eq", Peq), ("q", Pq)):
        out = os.path.join(DATA, f"train_native_rand_aero_{tag}.npy")
        if os.path.exists(out):
            print(f"  {tag}: exists"); continue
        t0 = time.time(); X = render(P, draws, wl, a.jobs)
        bad = int((~np.all(np.isfinite(X), axis=1)).sum())
        np.save(out, X); print(f"  {tag}: {X.shape}, {bad} failures, {time.time()-t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
