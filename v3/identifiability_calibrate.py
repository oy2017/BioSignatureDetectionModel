"""Calibrate the dchi2 identifiability test against nested-sampling posteriors (REVISION_PLAN.md, B1).

On a subset of POP-I planets, a full Bayesian retrieval of the noise-free Tier-3 spectrum with the
planet's Tier-1 per-bin sigma as the likelihood width gives, for each molecule X, the posterior
probability of the TRUE class, p_true = P(log X > t | data) for a present molecule and
1 - P(...) for an absent one. This is the expected posterior, the information-content measure
used by tier studies; --noisy adds one noise draw instead. The retrieval frees the same quantities
the dchi2 test searches (T, gray cloud-top pressure, the four abundances) plus an additive depth
offset, with uniform priors on the population's ranges, nestle's multi-ellipsoid sampler, and a
hard cap on likelihood calls. The forward model is built once per planet and its fitting
parameters updated per call (TauREx's tm[name] = value), which is several times faster than
rebuilding.

After the subset: the Spearman correlation between dchi2 (from identifiability.py) and
logit(p_true) per molecule, and the dchi2 at which a logistic fit of p_true crosses 0.95 and
0.84: the calibrated thresholds for "decidable".

Departure spot checks (--departure haze3e7 | cloud1e2 | noise_x2 | noise_x3): the observed spectrum carries the
departure while the retrieval's forward model stays the clear-atmosphere one the classifier was trained on
(so the retrieval is tested under the same model error), or, with --model-departure, includes the haze
(the identifiability question: is the label decidable given a correct model). The sample (--n) is half
planets whose CH4 label is present and decidable on the clean retrieval, half random, so the CH4 recall
under the departure is measurable. --summarize for a departure compares its posteriors with the clean
run's on the same planets.

Usage:
  python identifiability_calibrate.py --n 955 --jobs 8            # retrievals (resumable)
  python identifiability_calibrate.py --n 1 --jobs 1 --npoints 50  # timing
  python identifiability_calibrate.py --summarize                  # the mapping, once both CSVs exist
  python identifiability_calibrate.py --departure haze3e7 --n 60 --jobs 2 [--model-departure]
  python identifiability_calibrate.py --departure haze3e7 --summarize [--model-departure]
Writes results/identifiability_calibration[_<departure>[_modelled]].csv / .txt
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import alfnoor_screen as AS  # noqa: E402
import forward_model as FM  # noqa: E402
import generate_grid as G  # noqa: E402
from identifiability import THRESH, T_RANGE, LOGP_RANGE, AB_RANGE, HAZE, NOISE_X  # noqa: E402

MOLS = AS.MOLS
OFFSET = 5e-3            # +- fractional depth, uniform prior on the additive offset


def select(Pte, n, seed=SEED):
    """Half random, half among planets with at least one label within 0.7 dex of the threshold."""
    rng = np.random.default_rng(seed)
    near = np.where(np.min(np.abs(Pte[[f"atm {m}" for m in MOLS]].to_numpy() - THRESH), axis=1) < 0.7)[0]
    far = np.setdiff1d(np.arange(len(Pte)), near)
    a = rng.choice(near, min(n // 2, len(near)), replace=False); b = rng.choice(far, n - len(a), replace=False)
    return np.sort(np.concatenate([a, b]))


def select_departure(Pte, n, seed=SEED):
    """Half planets with CH4 present and decidable (p_true > 0.95) on the clean retrieval, half random others."""
    rng = np.random.default_rng(seed + 1)
    C = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv")).set_index("index")
    idx = Pte["_index"].to_numpy(); have = np.isin(idx, C.index)
    dec = np.zeros(len(Pte), bool); dec[have] = (C.loc[idx[have], "CH4_label"].to_numpy() == 1) & (C.loc[idx[have], "CH4_p_true"].to_numpy() > 0.95)
    a = rng.choice(np.where(dec)[0], min(n // 2, int(dec.sum())), replace=False)
    b = rng.choice(np.setdiff1d(np.arange(len(Pte)), a), n - len(a), replace=False)
    return np.sort(np.concatenate([a, b]))


def retrieve_one(i, row, obs, sig, wl, edges, npoints, maxcall, seed, haze=None):
    import warnings; warnings.filterwarnings("ignore")
    import nestle
    G.FILL_GAS = AF.FILL
    t0 = time.time()
    try:
        r = dict(row); tm = FM.build_model(r, list(G.GASES), AF.FILL, cloud_pressure=float(r["cloud_pressure"]), haze=haze)
        T_p = float(r["T_p"])
        lo = np.array([T_RANGE[0] * T_p, LOGP_RANGE[0]] + [AB_RANGE[0]] * 4 + [-OFFSET])
        hi = np.array([T_RANGE[1] * T_p, LOGP_RANGE[1]] + [AB_RANGE[1]] * 4 + [OFFSET])

        def prior(u):
            return lo + u * (hi - lo)

        def loglike(x):
            tm["T"] = x[0]; tm["clouds_pressure"] = 10.0 ** x[1]
            for k, m in enumerate(MOLS):
                tm[m] = 10.0 ** x[2 + k]
            try:
                wn, d = tm.model()[:2]
            except Exception:
                return -1e10
            w = 1e4 / np.asarray(wn, float); o = np.argsort(w)
            d = np.asarray(d, float)[o]; w = w[o]; m = (w >= FM.WL_MIN) & (w <= FM.WL_MAX)
            y = bin_native(d[m][None, :], w[m], edges)[0] + x[6]
            if not np.all(np.isfinite(y)):
                return -1e10
            ll = -0.5 * np.sum(((obs - y) / sig) ** 2)
            return ll if np.isfinite(ll) else -1e10

        rng = np.random.RandomState(seed)
        res = nestle.sample(loglike, prior, 7, method="multi", npoints=npoints, maxcall=maxcall, rstate=rng)
        w = res.weights / res.weights.sum(); out = dict(index=int(i), name=r["name"], ncall=int(res.ncall), minutes=(time.time() - t0) / 60, logz=float(res.logz))
        for k, m in enumerate(MOLS):
            s = res.samples[:, 2 + k]; p = float(w[s > THRESH].sum()); lab = int(r[f"atm {m}"] > THRESH)
            q = lambda f: float(np.interp(f, np.cumsum(w[np.argsort(s)]), np.sort(s)))
            out.update({f"{m}_label": lab, f"{m}_p_present": p, f"{m}_p_true": p if lab else 1 - p, f"{m}_med": q(0.5), f"{m}_lo": q(0.16), f"{m}_hi": q(0.84), f"{m}_true": float(r[f"atm {m}"])})
        return out
    except Exception as e:
        return dict(index=int(i), name=row.get("name", "?"), ncall=0, minutes=(time.time() - t0) / 60, error=str(e)[:120])


def summarize_departure(tag):
    """The departure run's posteriors against the clean run's on the same planets: accuracy, CH4 recall, p_true shift."""
    cal = pd.read_csv(os.path.join(RESULTS, f"identifiability_calibration_{tag}.csv"))
    cal = (cal[cal["error"].isna()] if "error" in cal else cal).reset_index(drop=True)
    ref = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv")).set_index("index").loc[cal["index"]].reset_index()
    L = [f"retrieval of {len(cal)} planets under {tag} (observed spectrum with the departure; forward model "
         f"{'with the haze' if tag.endswith('_modelled') else 'clear, as trained'}), against the clean retrieval of the same planets", ""]
    for m in MOLS:
        y = cal[f"{m}_label"].to_numpy(); pd_ = cal[f"{m}_p_present"].to_numpy(); pc = ref[f"{m}_p_present"].to_numpy()
        td = cal[f"{m}_p_true"].to_numpy(); tc = ref[f"{m}_p_true"].to_numpy()
        acc = lambda p: 100 * ((p >= 0.5) == (y == 1)).mean(); rec = lambda p: 100 * (p[y == 1] >= 0.5).mean() if (y == 1).any() else np.nan
        dec = lambda t: 100 * (t > 0.95).mean(); mis = lambda t: 100 * (t < 0.05).mean()
        L.append(f"{m:<4} n={len(y)} present {int((y==1).sum())}:  accuracy clean {acc(pc):5.1f} -> {acc(pd_):5.1f}   recall of present {rec(pc):5.1f} -> {rec(pd_):5.1f}   "
                 f"decidable (p_true > 0.95) {dec(tc):5.1f} -> {dec(td):5.1f} %   confidently wrong (p_true < 0.05) {mis(tc):4.1f} -> {mis(td):4.1f} %   median p_true {np.median(tc):.3f} -> {np.median(td):.3f}   "
                 f"median posterior shift of log X {np.median(cal[f'{m}_med'].to_numpy() - ref[f'{m}_med'].to_numpy()):+.2f} dex")
    cd = (cal["CH4_label"] == 1) & (ref["CH4_p_true"] > 0.95)
    if cd.any():
        L.append(f"CH4 present and decidable on clean (n={int(cd.sum())}): recall under {tag} {100 * (cal.loc[cd, 'CH4_p_present'] >= 0.5).mean():.1f} %, "
                 f"still decidable {100 * (cal.loc[cd, 'CH4_p_true'] > 0.95).mean():.1f} %, median p_true {cal.loc[cd, 'CH4_p_true'].median():.3f}")
    L.append(f"log evidence (median, min): clean {ref.logz.median():.1f}, {ref.logz.min():.1f} -> {tag} {cal.logz.median():.1f}, {cal.logz.min():.1f}; the retrieval's own goodness of fit flags the departure")
    L.append(""); L.append(f"sample: {cal.minutes.median():.1f} min and {cal.ncall.median():.0f} calls per planet (median)")
    open(os.path.join(RESULTS, f"identifiability_calibration_{tag}.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def summarize():
    """Spearman per molecule; calibrated thresholds (the dchi2 at which the logistic fit of 1[p_true > q] crosses 1/2, q = 0.95
    and 0.84); a monotone (isotonic) map dchi2 -> p_true applied to every planet of the identifiability run, giving the
    population's undecidable fractions and Bayes-accuracy ceiling. Writes identifiability_calibration.txt and
    identifiability_calibration_thresholds.csv (read by alfnoor_diagnostic.py)."""
    from scipy.stats import spearmanr
    from scipy.optimize import brentq
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    cal = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv"))
    idf = pd.read_csv(os.path.join(RESULTS, "identifiability_none_tier1.csv")); idf = idf[np.isfinite(idf.dchi2)]
    full = cal["index"].nunique() >= idf["index"].nunique()
    L = [f"dchi2 test against nested-sampling posteriors on {cal['index'].nunique()} POP-I planets (noise-free spectra, Tier-1 sigma)",
         "p_true = posterior probability of the true class; decidable at level q if p_true > q",
         ("the retrieval covers every planet of the identifiability run, so the 'sample' fractions and ceiling are the population's; "
          "the isotonic column is what the dchi2 test alone would give (the conditional mean of p_true at each dchi2, thresholded)") if full else
         "the 'population' fractions and ceiling map the sample's dchi2 -> p_true relation onto every planet of the identifiability run", ""]
    rows = []
    for m in MOLS:
        a = idf[idf.molecule == m][["index", "dchi2", "label"]].merge(cal[["index", f"{m}_p_true"]], on="index")
        if len(a) < 10:
            L.append(f"{m}: only {len(a)} planets in both files"); continue
        p = np.clip(a[f"{m}_p_true"].to_numpy(), 1e-4, 1 - 1e-4); d = a.dchi2.to_numpy(); x = np.log10(np.clip(d, 1e-3, None))
        rho = spearmanr(d, np.log(p / (1 - p))).correlation
        thr = {}
        for q in (0.95, 0.84):
            yq = (a[f"{m}_p_true"] > q).astype(int).to_numpy()
            if yq.min() == yq.max():
                thr[q] = np.nan; continue
            lr = LogisticRegression(C=1e6).fit(x[:, None], yq)
            f = lambda t: lr.predict_proba([[np.log10(t)]])[0, 1] - 0.5
            try:
                thr[q] = brentq(f, 1e-3, 1e4)
            except ValueError:
                thr[q] = np.nan
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, a[f"{m}_p_true"].to_numpy())
        allm = idf[idf.molecule == m]; pp = iso.predict(np.log10(np.clip(allm.dchi2.to_numpy(), 1e-3, None)))
        ceiling_sample = np.maximum(p, 1 - p).mean(); ceiling_pop = np.maximum(pp, 1 - pp).mean()
        L.append(f"{m:<4} n={len(a)}  Spearman(dchi2, logit p_true) = {rho:.2f};  calibrated threshold: p_true > 0.95 at dchi2 = {thr[0.95]:.2f}, "
                 f"p_true > 0.84 at dchi2 = {thr[0.84]:.2f}  (nested-model 95 %: 3.84; 3 sigma: 9)")
        L.append(f"     undecidable (p_true < 0.95 / < 0.84): sample {100*(p < 0.95).mean():.0f} / {100*(p < 0.84).mean():.0f} %;  population via the isotonic map "
                 f"{100*(pp < 0.95).mean():.0f} / {100*(pp < 0.84).mean():.0f} %;  Bayes-accuracy ceiling sample {100*ceiling_sample:.1f} %, population {100*ceiling_pop:.1f} %")
        rows.append(dict(molecule=m, n=len(a), spearman=rho, dchi2_at_95=thr[0.95], dchi2_at_84=thr[0.84], undecidable_95_pop=(pp < 0.95).mean(),
                         undecidable_84_pop=(pp < 0.84).mean(), ceiling_pop=ceiling_pop, ceiling_sample=ceiling_sample,
                         undecidable_95_sample=(p < 0.95).mean(), undecidable_84_sample=(p < 0.84).mean(), full_population=full))
    L.append(""); L.append(f"sample: {cal.minutes.median():.1f} min and {cal.ncall.median():.0f} calls per planet (median)")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "identifiability_calibration_thresholds.csv"), index=False)
    open(os.path.join(RESULTS, "identifiability_calibration.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--npoints", type=int, default=100); ap.add_argument("--maxcall", type=int, default=20000)
    ap.add_argument("--noisy", action="store_true"); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--departure", default="none", choices=["none", "haze3e7", "cloud1e2"] + list(NOISE_X)); ap.add_argument("--model-departure", action="store_true")
    a = ap.parse_args()
    tag = "" if a.departure == "none" else f"_{a.departure}" + ("_modelled" if a.model_departure else "")
    if a.summarize:
        (summarize_departure(tag[1:]) if tag else summarize()); return
    src = AF.DIRS["faithful"]; wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = AF.layout_edges("tier3_r20")
    nat = {"haze3e7": "pop1_native_haze3e7", "cloud1e2": "pop1_native_cloud1e2"}.get(a.departure, "pop1_native")
    Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet")); Y = np.load(os.path.join(src, f"{nat}.npy")).astype(np.float64)
    ok = np.all(np.isfinite(Y), axis=1) & np.all(np.isfinite(np.load(os.path.join(src, "pop1_native.npy"))), axis=1)
    Pte = Pte[ok].reset_index(drop=False).rename(columns={"index": "_index"}); Y = Y[ok]
    haze = {"radius": 0.1, "q": 40, "mix_ratio": HAZE[a.departure]} if (a.model_departure and a.departure in HAZE) else None
    mcs = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet"))[["name", "atm temperature"]].rename(columns={"atm temperature": "T_p"})
    Pte = Pte.merge(mcs.drop_duplicates("name"), on="name", how="left")
    sig = AF.sigma(Pte, edges, "radiometric") * NOISE_X.get(a.departure, 1.0); Xb = bin_native(Y, wl, edges)
    if a.noisy:
        Xb = Xb + np.random.default_rng(SEED + 7).normal(0, 1, Xb.shape) * sig
    sel = np.arange(len(Pte)) if a.n >= len(Pte) else (select_departure(Pte, a.n) if tag else select(Pte, a.n))
    out = os.path.join(RESULTS, f"identifiability_calibration{tag}.csv")
    done = pd.read_csv(out) if os.path.exists(out) and a.n > 1 else pd.DataFrame()
    todo = [i for i in sel if done.empty or int(Pte.iloc[i]["_index"]) not in set(done["index"])]
    print(f"{len(sel)} planets selected, {len(todo)} to retrieve, {a.jobs} jobs, {a.npoints} live points, cap {a.maxcall} calls", flush=True)
    from joblib import Parallel, delayed
    t0 = time.time()
    # one pool over every planet, results streamed as they finish (no batch barrier, so a slow planet holds no one up);
    # the CSV is rewritten every `a.jobs` results and the run resumes from it
    gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(
        delayed(retrieve_one)(int(Pte.iloc[i]["_index"]), Pte.iloc[i].to_dict(), Xb[i], sig[i], wl, edges, a.npoints, a.maxcall, SEED + int(i), haze) for i in todo)
    pending = []
    for k, row in enumerate(gen, 1):
        pending.append(row)
        if k % a.jobs == 0 or k == len(todo):
            new = pd.DataFrame(pending); done = pd.concat([done, new], ignore_index=True); pending = []
            if a.n > 1:
                done.to_csv(out, index=False)
            print(f"  {k}/{len(todo)}  {time.time()-t0:.0f} s  median {new.minutes.median():.1f} min, {new.ncall.median():.0f} calls per planet", flush=True)
            if a.n == 1:
                print(new.T.to_string())


if __name__ == "__main__":
    main()
