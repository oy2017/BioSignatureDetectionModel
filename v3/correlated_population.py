"""Render the chemically correlated training population C (REVISION_EXPECTATIONS.md, section M). Writes data/correlated/{correlated_native.npy, correlated_params.parquet}.

Usage: python correlated_population.py [jobs] [n]
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import DATA, SEED  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import generate_grid as G  # noqa: E402

N = 16000; OUT = os.path.join(DATA, "correlated"); os.makedirs(OUT, exist_ok=True); GAS = ["H2O", "CO2", "CH4", "NH3", "CO"]


def chem_chunk(c):
    ch = G.Chemistry(); return [ch.composition(*x[:3], mode=x[3], g_cgs=x[4]) for x in c]


def draw(P, n, rng, jobs):
    Q = P.iloc[rng.integers(0, len(P), n)].reset_index(drop=True); T = 10 ** rng.uniform(np.log10(250.0), np.log10(2500.0), n)
    Q["modulation_5H"] = Q["modulation_5H"].to_numpy() * T / Q["atm temperature"].to_numpy(); Q["atm temperature"] = T   # scale height follows temperature
    Q["co_ratio"] = rng.uniform(0.2, 1.5, n); Q["mh"] = rng.uniform(-1.0, 3.0, n); Q["mode"] = np.where(rng.uniform(size=n) < 0.5, "equilibrium", "quenched")
    g = 6.674e-8 * (Q.p_mass.to_numpy() * 5.972e27) / (Q.p_radius.to_numpy() * 6.371e8) ** 2; a = list(zip(T, Q.co_ratio, Q.mh, Q["mode"], g))
    C = [x for c in Parallel(n_jobs=jobs)(delayed(chem_chunk)(a[k:k + 200]) for k in range(0, n, 200)) for x in c]; A = np.array([[c[h] for h in GAS] for c in C])
    Q[[f"eq {h}" for h in GAS]] = A; s = rng.uniform(0, 1.5, n)[:, None]; A = A + rng.normal(0, 1, A.shape) * s; A = np.where(rng.uniform(size=A.shape) < 0.15, rng.uniform(-9, -1, A.shape), A); A = np.clip(A, -15, -0.3)
    t = (10 ** A).sum(1, keepdims=True); A = A - np.log10(np.maximum(t / 0.9, 1.0))     # room for the H2/He fill
    Q[[f"atm {h}" for h in GAS]] = A; u = rng.uniform(size=n); Q["group"] = np.where(u < 1 / 3, "clear", np.where(u < 2 / 3, "haze", "deck")); Q["haze_density"] = 10.0 ** rng.uniform(6.0, np.log10(3e8), n); Q["deck_pa"] = 10.0 ** rng.uniform(np.log10(5.0), np.log10(500.0), n)
    return Q


def main(jobs=4, n=N):
    wl = np.load(os.path.join(DATA, "native_wl.npy")); P = pd.read_parquet(os.path.join(AF.DIRS["faithful"], "pop3_params.parquet")); rng = np.random.default_rng(SEED + 2345); Q = draw(P, n, rng, jobs); rows = []
    for _, q in Q.iterrows():
        r = {k: q[k] for k in P.columns}
        if q.group == "haze": r["_haze"] = float(q.haze_density)
        if q.group == "deck": r["cloud_pressure"] = float(q.deck_pa)
        rows.append(r)
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(AF._worker)(rows[k:k + 25], wl, None, {}) for k in range(0, n, 25)); X = np.full((n, len(wl)), np.nan, np.float32); j = 0
    for c in res:
        for s in c:
            if s is not None: X[j] = s
            j += 1
    tag = "" if n == N else f"_test{n}"; np.save(os.path.join(OUT, f"correlated_native{tag}.npy"), X); Q.to_parquet(os.path.join(OUT, f"correlated_params{tag}.parquet")); print("finite", int(np.isfinite(X).all(1).sum()), "of", n, flush=True)
    print(Q[[f"atm {h}" for h in GAS]].describe().round(2)); print({h: round(float((Q[f"atm {h}"] > -4).mean()), 3) for h in GAS})


if __name__ == "__main__":
    a = sys.argv[1:]; main(int(a[0]) if a else 4, int(a[1]) if len(a) > 1 else N)
