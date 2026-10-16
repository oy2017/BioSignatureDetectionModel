"""Is the label decidable from the spectrum at all? (REVISION_PLAN.md, B1-B3.)

For each test planet and each molecule X, the classifier's label is "log10 abundance > t"
(t = -4 by default). The label is IDENTIFIABLE at the planet's Tier-1 precision if no atmosphere of
the opposite class fits the planet's noise-free spectrum within the noise. The test:

    dchi2 = min over theta' in the opposite class of  sum_b [ (S_b(theta) - S_b(theta') - o) / sigma_b ]^2

over the 104 Tier-3 bins with the planet's radiometric per-bin sigma, where o is an additive offset
minimized analytically (the planet-radius degeneracy to first order; the classifier removes offsets
by normalization anyway). X is held at the class boundary t, which is the opposite class's closest
member (band depth is monotone in abundance), and the nuisance parameters are searched: isothermal
temperature within the population prior (0.7-1.05 T_eq), the gray cloud-top pressure (5e2-1e6 Pa),
and the other three abundances (1e-9-1e-2). Nelder-Mead from the true parameters plus random
restarts drawn from the prior; the smallest dchi2 is kept. The label is decidable at significance
s if dchi2 > s^2 (one-parameter nested-model test); the threshold is calibrated against
nested-sampling posteriors in identifiability_calibrate.py.

Departures (B3.3): the truth spectrum is the departed one and the counterfactual model carries the
same departure (the haze density, the deck, the spot factor, the extra absorbers, the noise level),
so dchi2 asks whether the label is decidable given a correct model of the departure. A classifier
error on a planet whose label is decidable is a recoverable error; on one whose label is not, it is
irrecoverable.

Usage:
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python identifiability.py --departure none --jobs 8   # POP-I at Tier-1 noise
  (pin the BLAS threads: NumPy on macOS uses Accelerate, which ignores OMP_NUM_THREADS; 8 unpinned
   workers spun 80 threads and ran ten times slower)
  python identifiability.py --departure haze3e7 --jobs 8
  python identifiability.py --noise tier2 --jobs 8
  python identifiability.py --limit 8 --jobs 4 --starts 2             # timing run
Writes results/identifiability_<departure>_<noise>.csv (one row per planet x molecule, resumable
by chunk) and the summary results/identifiability_<departure>_<noise>.txt.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import alfnoor_screen as AS  # noqa: E402
import forward_model as FM  # noqa: E402
import generate_grid as G  # noqa: E402

MOLS, THRESH = AS.MOLS, -4.0
T_RANGE, LOGP_RANGE, AB_RANGE = (0.7, 1.05), (np.log10(5e2), 6.0), (-9.0, -2.0)
HAZE = {"haze3e7": 3e7, "haze2e6": 2e6, "haze2p4e8": 2.4e8}
NOISE_X = {"noise_x2": 2.0, "noise_x3": 3.0}
SPOTS = {"spots10": 0.10, "spots20": 0.20}
TIERS = {"tier1": "tier1_transits", "tier2": "tier2_transits", "tier3": "tier3_transits"}


def chi2_offset(d, sigma):
    """chi^2 of the residual d after removing the best additive offset."""
    w = 1.0 / sigma ** 2
    o = np.sum(w * d) / np.sum(w)
    return float(np.sum(w * (d - o) ** 2))


class Planet:
    """One test planet: its truth spectrum (binned), noise, and a forward model for counterfactuals."""

    def __init__(self, row, truth_native, wl, edges, sigma, departure, gases, spot_factor=None):
        self.row, self.wl, self.edges, self.sigma = dict(row), wl, edges, sigma
        self.departure, self.gases = departure, gases
        self.haze = None if departure not in HAZE else {"radius": 0.1, "q": 40, "mix_ratio": HAZE[departure]}
        self.fixed_cloud = 1e2 if departure == "cloud1e2" else None
        self.spot = spot_factor                     # multiplicative contamination on the native grid, or None
        self.T_p = float(row["T_p"])                 # the MCS transit temperature; the drawn T is U(0.7, 1.05) x T_p
        self.truth = self._bin(truth_native)
        self.calls = 0

    def _bin(self, y):
        if self.spot is not None:
            y = y * self.spot
        return bin_native(np.asarray(y, float)[None, :], self.wl, self.edges)[0]

    def render(self, T, logp, abund):
        r = dict(self.row); r["atm temperature"] = float(T)
        for g, v in abund.items():
            r[f"atm {g}"] = float(v)
        cp = self.fixed_cloud if self.fixed_cloud is not None else 10.0 ** logp
        self.calls += 1
        y = FM.spectrum(r, self.wl, self.gases, AF.FILL, cloud_pressure=cp, haze=self.haze)
        return None if y is None else self._bin(y.astype(float))

    def dchi2(self, mol, starts, rng):
        """Minimum chi^2 (offset-free) of the opposite class, with mol pinned at the boundary."""
        others = [m for m in MOLS if m != mol]
        t0 = float(self.row["atm temperature"]); lp0 = np.log10(float(self.row["cloud_pressure"]))
        x_true = np.array([t0] + [lp0] + [float(self.row[f"atm {m}"]) for m in others])
        lo = np.array([T_RANGE[0] * self.T_p, LOGP_RANGE[0]] + [AB_RANGE[0]] * 3)
        hi = np.array([T_RANGE[1] * self.T_p, LOGP_RANGE[1]] + [AB_RANGE[1]] * 3)
        scale = hi - lo

        def f(u):                                  # u in [0, 1]^5 (clipped), mapped to the box
            x = lo + np.clip(u, 0, 1) * scale
            ab = {m: x[2 + k] for k, m in enumerate(others)}; ab[mol] = THRESH
            y = self.render(x[0], x[1], ab)
            if y is None:
                return 1e12
            return chi2_offset(self.truth - y, self.sigma)
        best, bx = np.inf, None
        u0s = [(x_true - lo) / scale] + [rng.uniform(0, 1, 5) for _ in range(starts - 1)]
        for u0 in u0s:
            res = minimize(f, np.clip(u0, 0, 1), method="Nelder-Mead",
                           options=dict(xatol=1e-3, fatol=1e-3, maxfev=600, initial_simplex=self._simplex(np.clip(u0, 0, 1))))
            if res.fun < best:
                best, bx = res.fun, lo + np.clip(res.x, 0, 1) * scale
        return best, bx

    @staticmethod
    def _simplex(u0, step=0.15):
        S = [u0]
        for k in range(len(u0)):
            v = u0.copy(); v[k] = v[k] + step if v[k] + step <= 1 else v[k] - step; S.append(v)
        return np.array(S)


def _spot_factors(Pte, wl, f):
    import shift_tlse as T
    T.set_grid(wl); Tst = Pte["s temperature"].to_numpy(); logg = np.log10(T.G_SUN * Pte["s mass"].to_numpy() / Pte["s radius"].to_numpy() ** 2 * 100)
    return T.contamination(Tst, logg, f, 0.0)


def _worker(rows, truths, sigmas, spots, wl, edges, departure, gases, starts, seed):
    import warnings; warnings.filterwarnings("ignore")
    G.FILL_GAS = AF.FILL
    rng = np.random.default_rng(seed); out = []
    for k, (r, y) in enumerate(zip(rows, truths)):
        pl = Planet(r, y, wl, edges, sigmas[k], departure, gases, None if spots is None else spots[k])
        for mol in MOLS:
            t0 = time.time(); d, bx = pl.dchi2(mol, starts, rng)
            out.append(dict(index=r["_index"], name=r["name"], molecule=mol, log_true=float(r[f"atm {mol}"]),
                            label=int(r[f"atm {mol}"] > THRESH), dchi2=d, calls=pl.calls, seconds=time.time() - t0,
                            cf_T=bx[0] if bx is not None else np.nan, cf_logP=bx[1] if bx is not None else np.nan))
            pl.calls = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--departure", default="none", choices=["none", "cloud1e2", "absorbers"] + list(HAZE) + list(SPOTS) + list(NOISE_X))
    ap.add_argument("--noise", default="tier1", choices=list(TIERS)); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--starts", type=int, default=3); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--chunk", type=int, default=2)
    a = ap.parse_args()
    src = AF.DIRS["faithful"]; wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = AF.layout_edges("tier3_r20")
    Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet"))
    nat = {"none": "pop1_native", "cloud1e2": "pop1_native_cloud1e2", "absorbers": "pop1_native_absorbers"}
    nat.update({k: "pop1_native_haze3e7" for k in HAZE if k == "haze3e7"}); nat.update({k: "pop1_native" for k in list(SPOTS) + list(NOISE_X)})
    if a.departure in HAZE and a.departure != "haze3e7":
        sys.exit("only haze3e7 has a rendered POP-I set; render the others first")
    Y = np.load(os.path.join(src, f"{nat[a.departure]}.npy")).astype(np.float64)
    ok = np.all(np.isfinite(Y), axis=1) & np.all(np.isfinite(np.load(os.path.join(src, "pop1_native.npy"))), axis=1)
    Pte = Pte[ok].reset_index(drop=False).rename(columns={"index": "_index"}); Y = Y[ok]
    mcs = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet"))[["name", "atm temperature"]].rename(columns={"atm temperature": "T_p"})
    Pte = Pte.merge(mcs.drop_duplicates("name"), on="name", how="left"); assert Pte.T_p.notna().all()
    gases = list(G.GASES) + (["HCN", "C2H2"] if a.departure == "absorbers" else [])
    sig = AF.sigma(Pte, edges, "radiometric") if a.noise == "tier1" else AF.NZ.radiometric_sigma(Pte, edges, TIERS[a.noise])[0]
    sig = sig * NOISE_X.get(a.departure, 1.0)
    spots = _spot_factors(Pte, wl, SPOTS[a.departure]) if a.departure in SPOTS else None
    if a.limit:
        Pte, Y, sig = Pte.iloc[:a.limit], Y[:a.limit], sig[:a.limit]; spots = None if spots is None else spots[:a.limit]
    tag = f"{a.departure}_{a.noise}"; out_csv = os.path.join(RESULTS, f"identifiability_{tag}.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) and not a.limit else pd.DataFrame()
    todo = [i for i in range(len(Pte)) if done.empty or int(Pte.iloc[i]["_index"]) not in set(done["index"])]
    print(f"{tag}: {len(Pte)} planets, {len(todo)} to do, {a.starts} starts, {a.jobs} jobs", flush=True)
    rows = [Pte.iloc[i].to_dict() for i in todo]
    chunks = [list(range(i, min(i + a.chunk, len(todo)))) for i in range(0, len(todo), a.chunk)]
    t0 = time.time(); n_done = 0
    for batch in [chunks[i:i + a.jobs] for i in range(0, len(chunks), a.jobs)]:
        res = Parallel(n_jobs=a.jobs)(delayed(_worker)([rows[j] for j in c], [Y[todo[j]] for j in c], [sig[todo[j]] for j in c],
                                                      None if spots is None else [spots[todo[j]] for j in c], wl, edges, a.departure, gases, a.starts, SEED + c[0])
                                      for c in batch)
        new = pd.DataFrame([r for c in res for r in c]); done = pd.concat([done, new], ignore_index=True)
        if not a.limit:
            done.to_csv(out_csv, index=False)
        n_done += sum(len(c) for c in batch)
        print(f"  {n_done}/{len(todo)} planets, {time.time()-t0:.0f} s, median {new.seconds.median():.1f} s and {new.calls.median():.0f} calls per planet-molecule", flush=True)
    if a.limit:
        print(done.to_string()); return
    summarize(done, tag)


def summarize(df, tag):
    L = [f"Identifiability of the consortium labels (log10 abundance > {THRESH:g}) [{tag}]: {df['index'].nunique()} planets",
         "dchi2 = min chi^2 of the opposite class (offset-free, 104 Tier-3 bins, the planet's per-bin sigma)", ""]
    for s, lab in ((1.96, "95 % one-sided (dchi2 < 3.84)"), (3.0, "3 sigma (dchi2 < 9)"), (5.0, "5 sigma (dchi2 < 25)")):
        L.append(f"== undecidable fraction at {lab}")
        for mol in MOLS:
            g = df[df.molecule == mol]; u = g.dchi2 < s ** 2
            L.append(f"  {mol:<4} all {u.mean()*100:5.1f} %   present {u[g.label == 1].mean()*100:5.1f} %   absent {u[g.label == 0].mean()*100:5.1f} %   "
                     f"within 0.5 dex of the threshold {u[np.abs(g.log_true - THRESH) < 0.5].mean()*100:5.1f} %")
        L.append("")
    L.append("== Bayes-optimal accuracy ceiling implied at 3 sigma: 1 - undecidable/2 (an undecidable label is a coin toss)")
    for mol in MOLS:
        g = df[df.molecule == mol]; u = (g.dchi2 < 9).mean(); L.append(f"  {mol:<4} ceiling {100*(1-u/2):5.1f} %")
    L.append(""); L.append(f"median calls per planet-molecule {df.calls.median():.0f}, median {df.seconds.median():.1f} s")
    open(os.path.join(RESULTS, f"identifiability_{tag}.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
