"""Every design on the real panel at Ariel Tier-1 precision, so that the gain and the simulated cost are stated at the same noise (review of 2026-10-30).

Each reduction's converted spectrum (flat fill as in jwst_panel.py) receives the payload model's Tier-1 noise at the planet's transit count in 200 draws, on top of
the JWST noise already in the points. In each draw the flags are recomputed: the shape flag at its clean thresholds, the amplitude flag with the JWST and Tier-1 noise
removed in quadrature, against the per-reduction thresholds of amplitude_flag.py. Verdict: majority of four learners; credit: right 1, split or withheld one half, wrong 0,
averaged over the draws. The same code with no added noise reproduces the JWST-precision scores of Table 8, which is the check printed first.

Writes results/jwst_panel/panel_tier1.{txt,csv}.
"""
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import DATA, RESULTS  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import alfnoor_jwst as AJ  # noqa: E402
import alfnoor_trust as AT  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import router_real as RR  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel"); MOLS, LEARNERS = D.MOLS, D.LEARNERS; BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
RULES_G, RULES_C = ("knn", "mahalanobis", "pca_recon"), ("margin", "ensemble")   # the existing rules as routers to the AIRS-only variant, added after the referee pass of 2026-10-30
DESIGNS = ["published", "no-optical", "AIRS-only", "shape flag", "haze-retrained", "haze+deck-retrained", "the method", "census, same haze design", "knn -> AIRS-only", "mahalanobis -> AIRS-only", "pca_recon -> AIRS-only", "margin -> AIRS-only", "ensemble -> AIRS-only", "amplitude flag, AIRS-only elsewhere"]


class Context:
    """The frozen rule: learners, clean thresholds and retrained designs; nothing in it depends on the real panel."""

    def __init__(self):
        cases, Pte, e, sig = D.test_cases(); self.sc = D.Scorer(e); S0, _, _ = self.sc.scores(cases["clean"]); q = lambda s: np.quantile(s[np.isfinite(s)], 1 - D.DECLINE)
        self.thr_opt = q(S0[("optical", None)]); self.thr_dis = {(m, k): q(S0[("disagree", (m, k))]) for m in MOLS for k in LEARNERS}
        self.thr_g = {r: q(S0[(r, None)]) for r in RULES_G}; self.thr_c = {(r, m, k): q(S0[(r, (m, k))]) for r in RULES_C for m in MOLS for k in LEARNERS}
        self.ret = {s: joblib.load(os.path.join(DATA, "alfnoor_jwst", f"learners_all_{s}.joblib")) for s in ("haze", "hazedeck")}
        self.M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name"); self.edges, _ = J.tier3_layout()


def score(ctx, X, meas, base_name, noise2, hi, lo, census=False, extra_design=True):
    """Votes of every design on a batch of 104-bin spectra X, with the flags recomputed from each spectrum. hi, lo: the amplitude thresholds through this reduction's measured bins."""
    sc = ctx.sc; Z = AT.norm(X); opt = np.abs(sc._optical(Z) - sc.opt_mu) / sc.opt_sd > ctx.thr_opt; P = sc.probs(X); PR = {s: {(m, k): AT.prob(L["models"][("1e-4", m, k)], AJ.znorm(X, L["cols"])) for m in MOLS for k in LEARNERS} for s, L in ctx.ret.items()}
    S_, _, _ = sc.scores(X)
    md = float(ctx.M.loc[base_name, "modulation_5H"]) * 1e6
    v = np.var(X[:, meas] * 1e6, axis=1) - noise2; ratio = np.full(len(X), np.inf); ratio[v > 0] = md / np.sqrt(v[v > 0]); up = np.isfinite(ratio) & (ratio > hi); low = ratio < lo
    out = {}
    for m in MOLS:
        votes = {d: np.zeros(len(X)) for d in DESIGNS}
        for k in LEARNERS:
            pa, pn, ps = P["all"][(m, k)], P["no_optical"][(m, k)], P["airs_only"][(m, k)]; dis = np.maximum(np.abs(pa - pn), np.abs(pa - ps)) > ctx.thr_dis[(m, k)]; shape = np.where(opt | dis, ps, pa)
            for d, p in (("published", pa), ("no-optical", pn), ("AIRS-only", ps), ("shape flag", shape), ("haze-retrained", PR["haze"][(m, k)]), ("haze+deck-retrained", PR["hazedeck"][(m, k)]), ("the method", np.where(up, PR["haze"][(m, k)], shape)), ("census, same haze design", PR["haze"][(m, k)] if census else shape), ("amplitude flag, AIRS-only elsewhere", np.where(up, PR["haze"][(m, k)], ps))):
                votes[d] += p >= 0.5
            for r in RULES_G: votes[f"{r} -> AIRS-only"] += np.where(S_[(r, None)] > ctx.thr_g[r], ps, pa) >= 0.5
            for r in RULES_C: votes[f"{r} -> AIRS-only"] += np.where(S_[(r, (m, k))] > ctx.thr_c[(r, m, k)], ps, pa) >= 0.5
        out[m] = (votes, low, up)
    return out


def main(n=500):
    ctx = Context(); M = ctx.M; edges = ctx.edges; rows = []
    F = pd.read_csv(os.path.join(OUT, "amplitude_flag.csv")).set_index("planet")
    MR_ = pd.read_csv(os.path.join(OUT, "muting_router.csv")); census_up = set(MR_[MR_.muting_flag == True].planet)  # the muting census: beyond the prior, read against a clear twin at the published composition

    for pl, d in RR.PANEL.items():
        cov = pd.read_csv(d["coverage"]); x = cov.depth.to_numpy().copy(); filled = cov.filled.to_numpy(bool); x[filled] = np.nanmean(x[~filled]); err = cov.err.to_numpy()
        s = J.ariel_sigma(BASE(pl), edges); j2 = float(np.nanmean(err[~filled] ** 2)) * 1e12; t2 = float(np.mean(s[~filled] ** 2)) * 1e12
        for prec, X, noise2 in (("JWST", x[None, :], j2), ("Tier-1", J.noise_draws(x, s, n, seed=405), j2 + t2)):
            res = score(ctx, X, ~filled, BASE(pl), noise2, F.hi_clear_max[pl], F.lo_own_p01[pl], census=pl in census_up)
            for m in MOLS:
                votes, low, up = res[m]; lab = d["labels"].get(m)
                for name in DESIGNS:
                    pr = votes[name] >= 3; ab = votes[name] <= 1; sp = ~(pr | ab)
                    if name in ("the method", "amplitude flag, AIRS-only elsewhere"): sp = sp | low; pr = pr & ~low; ab = ab & ~low
                    if name.startswith("census") and pl == "HIP67522b": sp = np.ones_like(sp); pr = np.zeros_like(pr); ab = np.zeros_like(ab)   # the census's own reading: nine times more featured than its twin, withheld as the lower cut does
                    cr = np.nan if lab is None else float(np.mean((pr if lab == "present" else ab) + 0.5 * sp))
                    rows.append(dict(precision=prec, planet=pl, molecule=m, design=name, label=lab, credit=cr, present=float(pr.mean()), split=float(sp.mean()), absent=float(ab.mean()), up=float(up.mean()), low=float(low.mean())))
        print(pl, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "panel_tier1.csv"), index=False)
    L = [f"Designs on the real panel at JWST precision and at Tier-1 noise ({n} draws); credit right 1, split or withheld 0.5, wrong 0; 95 % bootstrap over planets of the difference from the published classifier", ""]
    rng = np.random.default_rng(405)
    for prec in ("JWST", "Tier-1"):
        q2 = df[(df.precision == prec) & df.label.notna()]; piv = q2.pivot_table(index=["planet", "molecule"], columns="design", values="credit"); plan = np.array([BASE(p) for p in piv.index.get_level_values(0)]); units = sorted(set(plan))
        per = piv.groupby(plan).sum().loc[units]; cnt = pd.Series(plan).value_counts().loc[units].to_numpy(); idx = rng.integers(0, len(units), (20000, len(units))); den = cnt[idx].sum(1)
        L.append(f"== {prec}: {len(piv)} labels"); L.append(f"{'design':<24} {'score':>6} {'vs published':>24} {'vs AIRS-only':>24} {'CH4 present on the 8 (mean)':>28} {'present labels':>15} {'absent labels':>14}")
        for name in DESIGNS:
            dif = lambda ref: (per[name].to_numpy()[idx].sum(1) - per[ref].to_numpy()[idx].sum(1)) / den * len(piv); a, b = dif("published"), dif("AIRS-only")
            ch4 = df[(df.precision == prec) & (df.design == name) & (df.molecule == "CH4") & df.planet.isin(RR.METHANE)].present.sum(); lp = q2[(q2.design == name) & (q2.label == "present")].credit.sum(); la = q2[(q2.design == name) & (q2.label == "absent")].credit.sum()
            L.append(f"{name:<24} {piv[name].sum():6.1f} {piv[name].sum() - piv['published'].sum():+6.1f} ({np.percentile(a, 2.5):+5.1f},{np.percentile(a, 97.5):+5.1f}) {piv[name].sum() - piv['AIRS-only'].sum():+6.1f} ({np.percentile(b, 2.5):+5.1f},{np.percentile(b, 97.5):+5.1f}) {ch4:28.2f} {lp:15.1f} {la:14.1f}")
        L.append("")
    tq = df[(df.precision == "Tier-1") & (df.molecule == "CH4") & df.planet.isin(RR.METHANE)].pivot_table(index="planet", columns="design", values="present")[DESIGNS]
    L += ["CH4 detection frequency at Tier-1 on the methane reductions", tq.round(2).to_string()]
    txt = "\n".join(L); open(os.path.join(OUT, "panel_tier1.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
