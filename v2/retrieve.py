"""Does the agreement between a cheap classifier and expensive Bayesian retrieval
survive distribution shift?

Amortized inference is sold on the premise that you pay once and apply everywhere.
That premise is never tested under shift. Here the SAME planets are retrieved twice
--- once from their clean spectra, once from spectra re-rendered under a haze ---
and the frozen classifier scores the identical spectra. Three quantities are then
comparable within planet:

  retrieval vs truth        does the expensive method itself degrade?
  classifier vs truth       does the cheap method degrade? (already known from the budget)
  classifier vs retrieval   does the PROXY RELATIONSHIP degrade, i.e. is amortization
                            still valid where it matters?

Retrieval: nested sampling (nestle, multi-ellipsoid) over seven parameters --- the
four gas abundances with opacity data, isothermal temperature, planet radius, and a
grey cloud-top pressure so the retrieval is not handed the absence of aerosols free. The
forward model is MultiREx at the Ariel configuration, the same one that generated
the data, so model mismatch is excluded by construction and the measured
disagreement is a lower bound. A hard cap on likelihood calls bounds the tail: the
earlier pilot had one planet run eight hours.

Usage:
  python retrieve.py --n 60 --jobs 20 --case clean
  python retrieve.py --n 60 --jobs 20 --case haze_3p0e7
Writes results/retrieval_{case}.csv (one row per planet, resumable).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bin_spectra import bin_native  # noqa: E402
from common import DATA, RESULTS, SEED, SNR, TESTS, centres, configs, load_split  # noqa: E402
from noise import sigma_matrix  # noqa: E402

GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]
BIO_CH4, BIO_O3 = -6.0, -7.0
# Only the four gases with opacity tables are retrieved. CO and NH3 have no cross
# sections in this compilation, so they act solely through mean molecular weight and
# are unconstrained by the spectrum; sampling them spends likelihood calls on the
# prior. Free parameters: H2O, CO2, CH4, O3, isothermal T, radius, grey cloud top.
RETRIEVED = ["H2O", "CO2", "CH4", "O3"]
NDIM = 7
NPOINTS = 80
MAXCALL = 15000          # hard cap; bounds the tail the earlier pilot suffered


def pooled(cfg):
    Xs, ys, Ps, Nf = [], [], [], []
    for t in TESTS:
        X, y, P = load_split(t, cfg)
        Xs.append(X); ys.append(y); Ps.append(P)
        Nf.append(np.load(os.path.join(DATA, f"{t}_{cfg}.npy")).astype(float))
    return np.vstack(Xs), np.concatenate(ys), pd.concat(Ps, ignore_index=True), np.vstack(Nf)


def shifted_spectra(case, cfg, P, Xnf):
    """Binned + noised spectra for a re-rendered axis, clean-level sigma, paired seeds."""
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    edges = np.array(configs()[cfg]["edges"]); cen = centres(cfg)
    Xn = np.vstack([np.load(os.path.join(DATA, f"{t}_native_{case}.npy")).astype(float) for t in TESTS])
    Xb = bin_native(Xn, wl, edges)
    Xb = np.where(np.isfinite(Xb), Xb, Xnf)
    sig = sigma_matrix(Xnf, P["s temperature"].to_numpy(), cen, SNR, "ariel", None)
    out = np.empty_like(Xb); i = 0
    for k, t in enumerate(TESTS):
        n = len(pd.read_parquet(os.path.join(DATA, f"{t}_params.parquet")))
        rng = np.random.default_rng(2000 + k + 1)
        out[i:i + n] = Xb[i:i + n] + rng.normal(0, 1, sig[i:i + n].shape) * sig[i:i + n]
        i += n
    return out


def select(y, P, n_per_class, seed=SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    return np.sort(np.concatenate([rng.choice(pos, n_per_class, replace=False),
                                   rng.choice(neg, n_per_class, replace=False)]))


def build(row, cloud=None):
    from multirex import Atmosphere, Planet, Star, System
    kw = dict(temperature=float(row["atm temperature"]),
              base_pressure=float(row["atm base_pressure"]),
              top_pressure=float(row["atm top_pressure"]),
              composition={g: float(row[f"atm {g}"]) for g in GASES}, fill_gas="H2")
    if cloud is not None:
        kw["cloud_pressure"] = cloud
    atm = Atmosphere(**kw)
    planet = Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]), atmosphere=atm)
    star = Star(temperature=float(row["s temperature"]), radius=float(row["s radius"]),
                mass=float(row["s mass"]))
    s = System(planet=planet, star=star, sma=float(row["sma"])); s.make_tm()
    return s


def retrieve_one(args):
    i, row, obs, sig, wn = args
    import warnings; warnings.filterwarnings("ignore")
    import nestle
    t0 = time.time()
    try:
        system = build(row)
        tm = system.transmission
        r0 = tm["planet_radius"]
        # 7 free parameters: four gases with opacity, T, radius, grey cloud top (log Pa)
        lo = np.array([-12, -12, -12, -12, 200.0, 0.6 * r0, 0.0])
        hi = np.array([-1, -1, -1, -1, 3000.0, 1.4 * r0, 6.0])

        def prior(u):
            return lo + u * (hi - lo)

        def loglike(x):
            for j, g in enumerate(RETRIEVED):
                tm[g] = 10 ** x[j]
            tm["T"] = x[4]; tm["planet_radius"] = x[5]
            try:
                model = system.generate_spectrum(wn)[1]
            except Exception:
                return -1e10
            if not np.all(np.isfinite(model)):
                return -1e10
            ll = -0.5 * np.sum(((obs - model) / sig) ** 2)
            return ll if np.isfinite(ll) else -1e10

        res = nestle.sample(loglike, prior, NDIM, method="multi", npoints=NPOINTS, maxcall=MAXCALL)
        w = res.weights / res.weights.sum()
        ch4 = res.samples[:, RETRIEVED.index("CH4")]; o3 = res.samples[:, RETRIEVED.index("O3")]
        p_pos = float(w[(ch4 > BIO_CH4) & (o3 > BIO_O3)].sum())
        med = lambda v: float(np.interp(0.5, np.cumsum(w[np.argsort(v)]), np.sort(v)))
        return dict(idx=int(i), ch4_med=med(ch4), o3_med=med(o3), p_pos=p_pos,
                    ret_label=int(med(ch4) > BIO_CH4 and med(o3) > BIO_O3),
                    ncall=int(res.ncall), minutes=(time.time() - t0) / 60)
    except Exception as e:
        return dict(idx=int(i), ch4_med=np.nan, o3_med=np.nan, p_pos=np.nan,
                    ret_label=-1, ncall=0, minutes=(time.time() - t0) / 60, error=str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ariel"); ap.add_argument("--case", default="clean")
    ap.add_argument("--n", type=int, default=60, help="planets PER CLASS")
    ap.add_argument("--jobs", type=int, default=20)
    a = ap.parse_args(); cfg = a.config
    X, y, P, Xnf = pooled(cfg)
    cen = centres(cfg)
    Xobs = X if a.case == "clean" else shifted_spectra(a.case, cfg, P, Xnf)
    sig = sigma_matrix(Xnf, P["s temperature"].to_numpy(), cen, SNR, "ariel", None)
    sel = select(y, P, a.n // 2)
    wn = 1e4 / cen[::-1]
    out_path = os.path.join(RESULTS, f"retrieval_{a.case}.csv")
    done = set()
    if os.path.exists(out_path):
        done = set(pd.read_csv(out_path)["idx"].tolist())
        print(f"resuming: {len(done)} planets already retrieved", flush=True)
    todo = [i for i in sel if int(i) not in done]
    print(f"{a.case}: {len(todo)} planets to retrieve, {a.jobs} workers, cap {MAXCALL} calls", flush=True)
    tasks = [(i, P.iloc[i].to_dict(), Xobs[i][::-1], sig[i][::-1], wn) for i in todo]
    from joblib import Parallel, delayed
    t0 = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(retrieve_one)(t) for t in tasks)
    df = pd.DataFrame(rows)
    df["true_label"] = [int(y[i]) for i in df["idx"]]
    if os.path.exists(out_path):
        df = pd.concat([pd.read_csv(out_path), df], ignore_index=True)
    df.to_csv(out_path, index=False)
    ok = df[df.ret_label >= 0]
    print(f"\n{a.case}: {len(ok)}/{len(df)} succeeded in {(time.time()-t0)/60:.0f} min "
          f"(median {ok.minutes.median():.1f} min/planet)")
    print(f"  retrieval label vs truth: {100*(ok.ret_label==ok.true_label).mean():.0f}%")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
