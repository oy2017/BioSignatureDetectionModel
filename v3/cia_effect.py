"""How much does collision-induced absorption change the test spectra? (review of 2026-10-30)

The draft stated "at most 3 ppm", a number whose script is not in the repository. Here a random 200 POP-I planets are rendered with and
without H2-H2 and H2-He CIA (HITRAN 2011 tables, https://hitran.org/data/CIA/main/H2-H2_2011.cia and H2-He_2011.cia) with their own decks, and the difference is binned to the 104 bins.

Usage: OMP_NUM_THREADS=1 python cia_effect.py <directory with H2-H2_2011.cia and H2-He_2011.cia>   -> results/cia_effect.txt
       python cia_effect.py <directory> --population   renders POP-I and POP-III with CIA: data/alfnoor_faithful/pop1_native_cia.npy (the test-only
       case scored by pipeline_sim.py) and data/alfnoor_faithful_cia/ (the refit: alfnoor_faithful.py --fit --natives faithful_cia --layout tier3_r20 --noise radiometric)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import alfnoor_faithful as AF  # noqa: E402
import forward_model as FM  # noqa: E402
import shift_aerosol as A  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
from taurex.cache import CIACache  # noqa: E402
from taurex.contributions import CIAContribution  # noqa: E402


def depth(tm, wl):
    wn, rp = tm.model()[:2]; w = 1e4 / np.asarray(wn, float); o = np.argsort(w); return np.interp(wl, w[o], np.asarray(rp, float)[o])


def main(path, n=200):
    CIACache().set_cia_path(path)
    src = AF.DIRS["faithful"]; wl = np.load("data/native_wl.npy"); e = AF.layout_edges("tier3_r20"); cen = 0.5 * (e[1:] + e[:-1])
    P = pd.read_parquet(os.path.join(src, "pop1_params.parquet")); ok = np.where(np.isfinite(np.load(os.path.join(src, "pop1_native.npy")).astype(float)).all(1))[0]
    idx = np.random.default_rng(405).choice(ok, n, replace=False); rows = []
    for i in idx:
        r = P.iloc[int(i)].to_dict(); cp = float(r["cloud_pressure"]); base = FM.build_model(r, A.GASES, AF.FILL, cp); y0 = depth(base, wl)
        tm = FM.build_model(r, A.GASES, AF.FILL, cp); tm.add_contribution(CIAContribution(cia_pairs=["H2-H2", "H2-He"])); tm.build(); y1 = depth(tm, wl)
        d = bin_native((y1 - y0)[None, :], wl, e)[0] * 1e6; b0 = bin_native(y0[None, :], wl, e)[0] * 1e6
        rows.append(dict(name=r["name"], cloud_pressure=cp, mod5H=float(r["modulation_5H"]) * 1e6, max_abs=np.abs(d).max(), rms=d.std(), mean=d.mean(), disp=b0.std(), peak_um=cen[np.argmax(np.abs(d))], low_abs=max(r["atm H2O"], r["atm CH4"], r["atm CO2"], r["atm NH3"])))
    D = pd.DataFrame(rows); D.to_csv("results/cia_effect.csv", index=False); q = lambda s: " / ".join(f"{np.percentile(s, p):.1f}" for p in (50, 90, 99, 100))
    L = [f"Collision-induced absorption, {n} random POP-I planets with their own decks, binned difference with minus without (ppm); median / 90th / 99th / max over planets",
         f"largest change in any bin: {q(D.max_abs)}", f"spread of the change across the bins (what survives the per-spectrum normalisation): {q(D.rms)}", f"mean change: {q(D['mean'].abs())}",
         f"spread of the change as % of the spectrum's dispersion: {q(100 * D.rms / D.disp)}", f"planets with a change above 3 ppm in some bin: {100 * np.mean(D.max_abs > 3):.0f} %; above 30 ppm: {100 * np.mean(D.max_abs > 30):.0f} %",
         f"wavelength of the largest change, median {D.peak_um.median():.2f} um"]
    open("results/cia_effect.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


def _work(rows, wl, path):
    import warnings as w; w.filterwarnings("ignore")
    CIACache().set_cia_path(path); out = []
    for r in rows:
        try:
            tm = FM.build_model(r, A.GASES, AF.FILL, float(r["cloud_pressure"])); tm.add_contribution(CIAContribution(cia_pairs=["H2-H2", "H2-He"])); tm.build(); y = depth(tm, wl)
            out.append(y.astype(np.float32) if np.all(np.isfinite(y)) and np.all((y > 0) & (y <= 1)) else None)
        except Exception:
            out.append(None)
    return out


def population(path, jobs=2):
    import shutil
    from joblib import Parallel, delayed
    src = AF.DIRS["faithful"]; dst = os.path.join(os.path.dirname(src), "alfnoor_faithful_cia"); os.makedirs(dst, exist_ok=True); wl = np.load("data/native_wl.npy")
    for split, f in (("pop1", os.path.join(src, "pop1_native_cia.npy")), ("pop3", os.path.join(dst, "pop3_native.npy"))):
        if os.path.exists(f):
            continue
        P = pd.read_parquet(os.path.join(src, f"{split}_params.parquet")); rows = [P.iloc[i].to_dict() for i in range(len(P))]; chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
        res = Parallel(n_jobs=jobs)(delayed(_work)(c, wl, path) for c in chunks); X = np.full((len(rows), len(wl)), np.nan, dtype=np.float32)
        for i, y in enumerate(y for c in res for y in c):
            if y is not None: X[i] = y
        np.save(f, X); print(split, int(np.isfinite(X).all(1).sum()), "of", len(rows), "rendered", flush=True)
    np.save(os.path.join(dst, "pop1_native.npy"), np.load(os.path.join(src, "pop1_native_cia.npy")))
    for n in ("pop1_params.parquet", "pop3_params.parquet"):
        shutil.copy(os.path.join(src, n), os.path.join(dst, n))


if __name__ == "__main__":
    population(sys.argv[1]) if "--population" in sys.argv else main(sys.argv[1])
