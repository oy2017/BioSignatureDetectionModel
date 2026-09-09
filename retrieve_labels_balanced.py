"""Retrieval-derived (and probabilistic) labels on a defensible, stratified sample.

For a stratified random sample of 25 positive + 25 negative held-out planets
(seed 42, drawn from the five pooled test sets), run a nested-sampling
atmospheric retrieval and form (a) a retrieval-derived biosignature label from
the posterior medians and (b) a posterior label probability P+ (the probabilistic
label). The identical planets are re-scored by the frozen classifier in
eval_retrieval_vs_classifier.py, so the two are compared planet-for-planet.

DISCLOSED IDEALISATION (best case, so the measured disagreement is a lower bound):
the retrieval uses MultiREx's own forward model and opacity (no model mismatch),
fixes every non-fitted parameter at its true value, sees clear-sky SNR-15 spectra,
and uses npoints=100. It therefore captures two of the four uncertainty sources the
reviewer named -- observational noise and parameter degeneracy -- but not stellar
context, competing abiotic explanations, or forward-model mismatch. A realistic
retrieval-based relabelling that adds those is named as the next step.

Usage: python retrieve_labels_balanced.py [n_per_class] [npoints]
"""
import sys
import time
import re

import numpy as np
import pandas as pd
import nestle
from multirex import Atmosphere, Planet, Star, System

GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]
CH4_THR, O3_THR = -6.0, -7.0          # log10 label thresholds
ERR = 1.0e-4                          # transit-depth error (SNR-15 noise floor)
SEED = 42
N_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
NPOINTS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT = "final_results/H2_retrieval_balanced.csv"

fp = re.compile(r"^-?\d+\.\d+$")


def spectral_cols(df):
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))], key=float)


def load_pooled():
    dfs = [pd.read_parquet(f"multirex_spectra_H2_test_set_{i}.parquet") for i in range(1, 6)]
    return pd.concat(dfs, ignore_index=True)


def select(df):
    """Stratified random sample: N_PER_CLASS positives + N_PER_CLASS negatives, fixed seed."""
    lab = ((df["atm CH4"].astype(float) >= CH4_THR)
           & (df["atm O3"].astype(float) >= O3_THR)).astype(int).values
    rng = np.random.default_rng(SEED)
    pos = np.where(lab == 1)[0]
    neg = np.where(lab == 0)[0]
    sp = rng.choice(pos, N_PER_CLASS, replace=False)
    sn = rng.choice(neg, N_PER_CLASS, replace=False)
    idx = np.concatenate([np.sort(sp), np.sort(sn)])
    return idx, lab


def retrieve(system, wn, obs, npoints):
    tm = system.transmission
    r0 = tm["planet_radius"]
    lo = np.array([-12, -12, -12, -12, 200.0, 0.6 * r0])
    hi = np.array([-1, -1, -1, -1, 3000.0, 1.4 * r0])

    def prior(u):
        return lo + u * (hi - lo)

    def loglike(x):
        tm["H2O"] = 10 ** x[0]; tm["CH4"] = 10 ** x[1]
        tm["CO2"] = 10 ** x[2]; tm["O3"] = 10 ** x[3]
        tm["T"] = x[4]; tm["planet_radius"] = x[5]
        try:
            model = system.generate_spectrum(wn)[1]
        except Exception:
            return -1e10
        if not np.all(np.isfinite(model)):
            return -1e10
        ll = -0.5 * np.sum(((obs - model) / ERR) ** 2)
        return ll if np.isfinite(ll) else -1e10

    res = nestle.sample(loglike, prior, 6, method="multi", npoints=npoints, maxcall=60000)
    w = res.weights / res.weights.sum()
    return res.samples, w


def wq(samples, w, q):
    order = np.argsort(samples)
    cw = np.cumsum(w[order])
    return np.interp(q, cw, samples[order])


def main():
    df = load_pooled()
    idx, lab = select(df)
    cols = spectral_cols(df)
    wl = np.array([float(c) for c in cols])
    wn = 1e4 / wl[::-1]
    spec = df[cols].values[:, ::-1]                      # ordered to match wn ascending

    print(f"pooled test planets: {len(df)};  sample: {N_PER_CLASS} pos + {N_PER_CLASS} neg "
          f"(seed {SEED});  npoints={NPOINTS};  ERR={ERR:.1e}", flush=True)

    rows = []
    t_start = time.time()
    for k, pi in enumerate(idx):
        row = df.iloc[pi]
        comp = {g: float(row[f"atm {g}"]) for g in GASES}
        atm = Atmosphere(temperature=float(row["atm temperature"]),
                         base_pressure=float(row["atm base_pressure"]),
                         top_pressure=float(row["atm top_pressure"]),
                         composition=comp, fill_gas="H2")
        system = System(planet=Planet(radius=float(row["p_radius"]),
                        mass=float(row["p_mass"]), atmosphere=atm),
                        star=Star(temperature=float(row["s temperature"]),
                        radius=float(row["s radius"]), mass=float(row["s mass"])),
                        sma=float(row["sma"]))
        system.make_tm()
        t0 = time.time()
        samples, w = retrieve(system, wn, spec[pi], NPOINTS)
        dt = time.time() - t0

        ch4_inj, o3_inj = float(row["atm CH4"]), float(row["atm O3"])
        ch4_med = wq(samples[:, 1], w, 0.5); ch4_lo = wq(samples[:, 1], w, 0.16); ch4_hi = wq(samples[:, 1], w, 0.84)
        o3_med = wq(samples[:, 3], w, 0.5); o3_lo = wq(samples[:, 3], w, 0.16); o3_hi = wq(samples[:, 3], w, 0.84)
        p_pos = float(w[(samples[:, 1] >= CH4_THR) & (samples[:, 3] >= O3_THR)].sum())
        true_label = int(lab[pi])
        ret_label = int(ch4_med >= CH4_THR and o3_med >= O3_THR)
        rows.append(dict(pool_idx=int(pi), true_label=true_label,
                         ch4_inj=ch4_inj, ch4_med=ch4_med, ch4_lo=ch4_lo, ch4_hi=ch4_hi,
                         o3_inj=o3_inj, o3_med=o3_med, o3_lo=o3_lo, o3_hi=o3_hi,
                         p_pos=p_pos, ret_label=ret_label,
                         flip=int(true_label != ret_label), sec=dt))
        pd.DataFrame(rows).to_csv(OUT, index=False)         # incremental checkpoint
        print(f"[{k+1}/{len(idx)}] pool#{pi} true={true_label} {dt/60:.1f}min | "
              f"CH4 {ch4_inj:+.2f}->{ch4_med:+.2f} | O3 {o3_inj:+.2f}->{o3_med:+.2f} | "
              f"ret={ret_label} P+={p_pos:.2f}" + ("  FLIP" if true_label != ret_label else ""),
              flush=True)

    R = pd.DataFrame(rows)
    print(f"\n=== {len(R)} planets, npoints={NPOINTS}, total {(time.time()-t_start)/60:.0f} min ===")
    print(f"retrieval label vs truth: {100*(R.true_label==R.ret_label).mean():.0f}% agree "
          f"({int(R.flip.sum())} flips)")
    print(f"  positives (recall):    {100*(R[R.true_label==1].ret_label==1).mean():.0f}%")
    print(f"  negatives (specificity): {100*(R[R.true_label==0].ret_label==0).mean():.0f}%")
    R.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
