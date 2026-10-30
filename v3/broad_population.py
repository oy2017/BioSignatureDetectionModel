"""Render the broad training population B (REVISION_EXPECTATIONS.md, section L). Writes data/broad/{broad_native.npy, broad_params.parquet}."""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import DATA, SEED  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402

N = 8000; OUT = os.path.join(DATA, "broad"); os.makedirs(OUT, exist_ok=True)


def main(jobs=4, n=N):
    wl = np.load(os.path.join(DATA, "native_wl.npy")); P = pd.read_parquet(os.path.join(AF.DIRS["faithful"], "pop3_params.parquet")); rng = np.random.default_rng(SEED + 1234)
    Q = P.iloc[rng.integers(0, len(P), n)].reset_index(drop=True)
    for g in ("H2O", "CO2", "CH4", "NH3", "CO"): Q[f"atm {g}"] = rng.uniform(-9, -1, n)
    u = rng.uniform(size=n); Q["group"] = np.where(u < 1 / 3, "clear", np.where(u < 2 / 3, "haze", "deck")); Q["haze_density"] = 10.0 ** rng.uniform(6.0, np.log10(3e8), n); Q["deck_pa"] = 10.0 ** rng.uniform(np.log10(5.0), np.log10(500.0), n)
    rows = []
    for i, q in Q.iterrows():
        r = {k: q[k] for k in P.columns}
        if q.group == "haze": r["_haze"] = float(q.haze_density)
        if q.group == "deck": r["cloud_pressure"] = float(q.deck_pa)
        rows.append(r)
    chunks = [rows[k:k + 25] for k in range(0, n, 25)]; res = Parallel(n_jobs=jobs, verbose=5)(delayed(AF._worker)(c, wl, None, {}) for c in chunks)
    X = np.full((n, len(wl)), np.nan, np.float32); j = 0
    for c in res:
        for s in c:
            if s is not None: X[j] = s
            j += 1
    np.save(os.path.join(OUT, "broad_native.npy"), X); Q.to_parquet(os.path.join(OUT, "broad_params.parquet")); print("finite", int(np.isfinite(X).all(1).sum()), "of", n)


if __name__ == "__main__":
    a = sys.argv[1:]; main(int(a[0]) if a else 4, int(a[1]) if len(a) > 1 else N)
