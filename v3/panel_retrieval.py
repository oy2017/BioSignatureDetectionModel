"""Retrieve the real panel, so that the paper's posterior definition of the error classes can be applied to real spectra (review of 2026-10-30).

For each reduction with a catalogue entry, the same nested sampling as identifiability_calibrate.py on the converted 104-bin spectrum, with the filled bins given no weight, at two
precisions (the JWST per-bin errors, and Ariel's Tier-1 sigma) and with two forward models (clear, and haze density free). The posterior probability of each label's true class then says
whether the spectrum decides it under this model. The radius is fixed at the catalogue's and the depth offset is free over a wider range than in the simulated study (REAL_OFFSET). Writes results/jwst_panel/panel_retrieval_<precision>_<model>.csv and, with --summarize, panel_retrieval.txt.

Usage: OMP_NUM_THREADS=1 python panel_retrieval.py --jobs 9 ; python panel_retrieval.py --summarize
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED  # noqa: E402
import identifiability_calibrate as IC  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import router_real as RR  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel"); BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", ""); MOLS = ["CH4", "H2O", "CO2", "NH3"]
HAZE = {"radius": 0.1, "q": 40, "mix_ratio": 1e3}
REAL_OFFSET = 0.02   # +- fractional depth: the real spectra sit up to 1.1 % of depth from the catalogue radius's model (WASP-17 b), beyond the simulated study's 0.5 %; the input of the classifier is normalised, so this freedom is what it already has


def _retrieve(*a, **k):
    IC.OFFSET = REAL_OFFSET
    return IC.retrieve_one(*a, **k)


def run(prec, model, jobs):
    out = os.path.join(OUT, f"panel_retrieval_{prec}_{model}.csv"); done = set(pd.read_csv(out).planet) if os.path.exists(out) else set()
    M = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet")).drop_duplicates("name").set_index("name"); wl = np.load(os.path.join(DATA, "native_wl.npy")); edges, _ = J.tier3_layout(); jobs_ = []
    for k, (pl, d) in enumerate(RR.PANEL.items()):
        if pl in done: continue
        cov = pd.read_csv(d["coverage"]); x = cov.depth.to_numpy().copy(); filled = cov.filled.to_numpy(bool); x[filled] = np.nanmean(x[~filled]); err = cov.err.to_numpy()
        sig = np.where(filled, 1.0, np.nan_to_num(err, nan=1.0)) if prec == "jwst" else np.where(filled, 1.0, J.ariel_sigma(BASE(pl), edges))
        r = M.loc[BASE(pl)].to_dict(); r["name"] = pl; r["T_p"] = float(r["atm temperature"]); r["cloud_pressure"] = 1e6; r["atm top_pressure"] = 1e-4; r["atm CO"] = -15.0; r["atm O3"] = -15.0
        for m in MOLS:
            lab = d["labels"].get(m); r[f"atm {m}"] = -3.0 if lab == "present" else -6.0
        jobs_.append((k, r, x, sig))
    print(prec, model, len(jobs_), "to retrieve", flush=True)
    for res in Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(_retrieve)(k, r, x, s, wl, edges, 100, 20000, SEED + k, HAZE if model == "freehaze" else None, 10800, model == "freehaze") for k, r, x, s in jobs_):
        res["planet"] = res["name"]
        old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(); pd.concat([old, pd.DataFrame([res])], ignore_index=True).to_csv(out, index=False); print(res["planet"], round(res.get("minutes", 0), 1), res.get("logz"), flush=True)


def summarize():
    L = ["The real panel retrieved (panel_retrieval.py): for the labels the literature decides, the share the retrieval decides right (p_true > 0.95), leaves undecided, and decides wrong (p_true < 0.05); score counts p_true > 0.5 as right, on the labels with a posterior, and calling every such label absent is the baseline", ""]
    V = pd.read_csv(os.path.join(OUT, "amplitude_flag_router.csv")).set_index(["planet", "molecule"]); allrows = []
    rng = np.random.default_rng(405); C = pd.read_csv(os.path.join(OUT, "panel_control.csv")).set_index(["planet", "molecule"]).control
    T = pd.read_csv(os.path.join(OUT, "panel_tier1.csv")); RULE = "mask, clear maximum (the rule) -> haze, two-sided"
    for prec in ("jwst", "tier1"):
        for model in ("clear", "freehaze"):
            f = os.path.join(OUT, f"panel_retrieval_{prec}_{model}.csv")
            if not os.path.exists(f): continue
            R = pd.read_csv(f).drop_duplicates("planet").set_index("planet"); rows = []
            for (pl, m), lab in V.label.dropna().items():
                if pl not in R.index or not np.isfinite(R.loc[pl].get(f"{m}_p_present", np.nan)): continue
                p = R.loc[pl, f"{m}_p_present"]; pt = p if lab == "present" else 1 - p; v = V.loc[(pl, m), "published"]; wrong = v != lab and v != "split"
                rows.append(dict(precision=prec, model=model, planet=pl, molecule=m, label=lab, p_true=pt, published_wrong=wrong, method=V.loc[(pl, m), RULE], control=C.get((pl, m), "no twin")))
            if not rows: continue
            D_ = pd.DataFrame(rows); allrows.append(D_); ok, un, ng = (D_.p_true > .95).sum(), ((D_.p_true >= .05) & (D_.p_true <= .95)).sum(), (D_.p_true < .05).sum(); wr = D_[D_.published_wrong]
            pres = D_[D_.label == "present"]; ab = D_[D_.label == "absent"]; score = float((D_.p_true > 0.5).sum()); tag = "JWST" if prec == "jwst" else "Tier-1"
            Q = T[(T.precision == tag) & T.label.notna()].set_index(["planet", "molecule"]); idx = list(zip(D_.planet, D_.molecule)); cred = lambda d: Q[Q.design == d].credit.reindex(pd.MultiIndex.from_tuples(idx)).to_numpy()
            cp, cm, ca = cred("published"), cred("the method"), cred("AIRS-only"); rt = (D_.p_true > .5).to_numpy(float); plan = D_.planet.map(BASE).to_numpy(); units = sorted(set(plan)); ix = rng.integers(0, len(units), (20000, len(units))); cnt = pd.Series(plan).value_counts().loc[units].to_numpy(); den = cnt[ix].sum(1)
            per = lambda c: pd.Series(c).groupby(plan).sum().loc[units].to_numpy(); dif = lambda a, b: (per(a)[ix].sum(1) - per(b)[ix].sum(1)) / den * len(D_)
            ci = lambda a, b: np.percentile(dif(a, b), [2.5, 97.5]).round(1)
            done = R[R["error"].isna()] if "error" in R else R; capped = int((done.ncall >= 20000).sum()) if "ncall" in done else 0
            L.append(f"{prec:<6} {model:<9} {len(D_)} labels, {R.shape[0]} reductions ({len(done)} completed, {capped} at the call cap, median {done.minutes.median():.0f} min): decided right {ok}, undecided {un}, decided wrong {ng}; present labels right/undecided/wrong {(pres.p_true > .95).sum()}/{((pres.p_true >= .05) & (pres.p_true <= .95)).sum()}/{(pres.p_true < .05).sum()} of {len(pres)}; absent {(ab.p_true > .95).sum()}/{((ab.p_true >= .05) & (ab.p_true <= .95)).sum()}/{(ab.p_true < .05).sum()} of {len(ab)}; median log Z {R.logz.median():.1f}")
            L.append(f"       score (p_true > 0.5) {score:.0f} of {len(D_)}, every label absent {len(ab)}; on these labels re-implemented classifier {np.nansum(cp):.1f}, AIRS-only {np.nansum(ca):.1f}, method {np.nansum(cm):.1f}; retrieval minus re-implemented {score - np.nansum(cp):+.1f} {ci(rt, cp)}, retrieval minus method {score - np.nansum(cm):+.1f} {ci(rt, cm)}")
            L.append(f"       of the re-implemented classifier's {len(wr)} wrong labels: recoverable {(wr.p_true > .95).sum()}, undecidable {((wr.p_true >= .05) & (wr.p_true <= .95)).sum()}, misled {(wr.p_true < .05).sum()}; of those passing the mask control ({(wr.control == 'passes').sum()}): recoverable {((wr.p_true > .95) & (wr.control == 'passes')).sum()}, undecidable {(((wr.p_true >= .05) & (wr.p_true <= .95)) & (wr.control == 'passes')).sum()}, misled {((wr.p_true < .05) & (wr.control == 'passes')).sum()}")
    if allrows:
        A = pd.concat(allrows); A.to_csv(os.path.join(OUT, "panel_retrieval_labels.csv"), index=False)
        L += ["", "methane labels, p_true per reduction"]; L.append(A[(A.molecule == "CH4") & (A.label == "present")].pivot_table(index="planet", columns=["precision", "model"], values="p_true").round(2).to_string())
        L += ["", "all present labels, p_true"]; L.append(A[A.label == "present"].pivot_table(index=["planet", "molecule"], columns=["precision", "model"], values="p_true").round(2).to_string())
    txt = "\n".join(L); open(os.path.join(OUT, "panel_retrieval.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=9); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--prec", nargs="*", default=["tier1", "jwst"]); ap.add_argument("--model", nargs="*", default=["clear", "freehaze"]); a = ap.parse_args()
    if a.summarize: summarize()
    else:
        for prec in a.prec:
            for model in a.model:
                run(prec, model, a.jobs)
