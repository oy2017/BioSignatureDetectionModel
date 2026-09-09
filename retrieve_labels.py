"""
Retrieval-derived labels (R1-2 / R1-4), trial version.

For each committed test planet: run a nested-sampling atmospheric retrieval that
fits the four spectrally active molecules (H2O, CH4, CO2, O3) plus temperature
and planet radius against the spectrum, then reads off the CH4 and O3 posteriors
and forms a *retrieval-derived* label (does the retrieval place CH4 >= 1e-6 AND
O3 >= 1e-7?). Compared against the injected-abundance label.

The forward model is MultiREx's own TransmissionModel, driven in place (only the
fitted parameters change between likelihood calls), so the physics matches the
generator exactly. Errors: a uniform 1e-4 in transit depth, the median noise
floor measured in the R1-3 domain-shift analysis (~SNR 15). The committed
spectrum is used as the observation as-is (no separate noise draw), so the
posterior width reflects parameter degeneracy at that error level.

Usage: python retrieve_labels.py <n_planets> <npoints>
"""
import sys, time, re
import numpy as np
import pandas as pd
import nestle
from multirex import Atmosphere, Planet, Star, System

GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]
FIT = ["H2O", "CH4", "CO2", "O3"]           # spectrally active -> fitted
FIX = ["CO", "NH3"]                          # no opacity -> fixed at injected (MMW only)
TEST = "multirex_spectra_H2_test_set_1.parquet"
CH4_THR, O3_THR = -6.0, -7.0                 # log10 label thresholds
ERR = 1.0e-4                                 # transit-depth error (noise floor)

fp = re.compile(r"^-?\d+\.\d+$")


def spectral_cols(df):
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))], key=float)


def retrieve(system, wn, obs, npoints):
    tm = system.transmission
    r0 = tm["planet_radius"]
    # priors: log10 mixing ratios in [-12,-1]; T in [200,3000] K; radius +/-40%
    lo = np.array([-12, -12, -12, -12, 200.0, 0.6 * r0])
    hi = np.array([-1,  -1,  -1,  -1, 3000.0, 1.4 * r0])
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
    res = nestle.sample(loglike, prior, 6, method="multi", npoints=npoints,
                        maxcall=60000)
    w = res.weights / res.weights.sum()
    return res.samples, w


def wq(samples, w, q):
    order = np.argsort(samples)
    cw = np.cumsum(w[order])
    return np.interp(q, cw, samples[order])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    npoints = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    global ERR
    if len(sys.argv) > 3:
        ERR = float(sys.argv[3])       # e.g. 1e-6 for the noise-free limit
    print(f"error bar (transit depth): {ERR:.1e}")
    df = pd.read_parquet(TEST)
    cols = spectral_cols(df)
    wl = np.array([float(c) for c in cols])
    wn = 1e4 / wl[::-1]
    obs_all = df[cols].values[:, ::-1]           # order to match wn ascending

    rows = []
    t_start = time.time()
    for i in range(n):
        row = df.iloc[i]
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
        samples, w = retrieve(system, wn, obs_all[i], npoints)
        dt = time.time() - t0

        ch4_inj, o3_inj = float(row["atm CH4"]), float(row["atm O3"])
        ch4_med = wq(samples[:, 1], w, 0.5); ch4_lo = wq(samples[:, 1], w, 0.16); ch4_hi = wq(samples[:, 1], w, 0.84)
        o3_med = wq(samples[:, 3], w, 0.5); o3_lo = wq(samples[:, 3], w, 0.16); o3_hi = wq(samples[:, 3], w, 0.84)
        # posterior probability the label is positive
        pos = w[(samples[:, 1] >= CH4_THR) & (samples[:, 3] >= O3_THR)].sum()
        inj_label = int(ch4_inj >= CH4_THR and o3_inj >= O3_THR)
        ret_label = int(ch4_med >= CH4_THR and o3_med >= O3_THR)
        rows.append(dict(i=i, ch4_inj=ch4_inj, ch4_med=ch4_med, ch4_lo=ch4_lo, ch4_hi=ch4_hi,
                         o3_inj=o3_inj, o3_med=o3_med, o3_lo=o3_lo, o3_hi=o3_hi,
                         p_pos=pos, inj_label=inj_label, ret_label=ret_label,
                         flip=int(inj_label != ret_label), sec=dt))
        print(f"[{i+1}/{n}] {dt/60:.1f} min | "
              f"CH4 inj {ch4_inj:+.2f} -> ret {ch4_med:+.2f} [{ch4_lo:+.2f},{ch4_hi:+.2f}] | "
              f"O3 inj {o3_inj:+.2f} -> ret {o3_med:+.2f} [{o3_lo:+.2f},{o3_hi:+.2f}] | "
              f"label inj {inj_label} ret {ret_label} P+={pos:.2f}"
              + ("  FLIP" if inj_label != ret_label else ""), flush=True)
        pd.DataFrame(rows).to_csv("final_results/H2_retrieval_trial.csv", index=False)

    R = pd.DataFrame(rows)
    flips = int(R["flip"].sum())
    total_min = (time.time() - t_start) / 60
    print(f"\n=== {n} planets, npoints={npoints} ===")
    print(f"median retrieval time: {R['sec'].median()/60:.1f} min/planet;  total {total_min:.1f} min")
    print(f"label flips (retrieval vs injected): {flips}/{n}")
    print(f"CH4 recovered within 0.5 dex: {(abs(R['ch4_med']-R['ch4_inj'])<0.5).sum()}/{n}; "
          f"O3 within 0.5 dex: {(abs(R['o3_med']-R['o3_inj'])<0.5).sum()}/{n}")
    R.to_csv("final_results/H2_retrieval_trial.csv", index=False)
    print("Wrote final_results/H2_retrieval_trial.csv")


if __name__ == "__main__":
    main()
