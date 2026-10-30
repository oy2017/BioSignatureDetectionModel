"""The full departure-aware pipeline measured in simulation, instead of copied from the shape-flag row (review of 2026-10-30).

Same test cases, noise draw and learners as alfnoor_diagnostic.py's remedy block. For each planet, molecule and learner:
  shape flag     optical-excess score or input disagreement above its clean 90th percentile -> AIRS-only variant
  amplitude flag catalogue five-scale-height modulation / spectrum dispersion above T_HI (muting_router.scale_height_proxy,
                 largest value of any clear POP-I planet) -> the retrained design on all inputs; takes precedence
  withheld       the same ratio below T_LO (first percentile of the population with its own decks); unscored
The ratio is computed as a pipeline would at Tier-1: the noisy spectrum's dispersion over the 104 bins with the nominal
Tier-1 noise removed in quadrature; a spectrum with no dispersion above its noise has no ratio and takes no amplitude flag,
as on the real panel. It is also computed on the noise-free spectrum, which bounds what the flag does at high precision.

Writes results/pipeline_sim.{txt,csv}.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as F  # noqa: E402
import alfnoor_jwst as AJ  # noqa: E402
import alfnoor_trust as AT  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import muting_router as MR  # noqa: E402

MOLS, LEARNERS = D.MOLS, D.LEARNERS
CASES = ["clean", "haze2e6", "haze3e7", "cloud1e2", "spots20", "compound", "white_x2", "white_x3"]
CONFINED = ["hazetop1", "plhaze29", "plhaze28"]   # haze_confined.py; used when rendered
CASES += [c for c in CONFINED if os.path.exists(os.path.join(F.DIRS["faithful"], f"pop1_native_{c}.npy"))]


def noise_free_cases(wl, e):
    import shift_tlse as T
    src = F.DIRS["faithful"]; Xte = np.load(os.path.join(src, "pop1_native.npy")).astype(float); ok = np.isfinite(Xte).all(1); clean_n = Xte[ok]
    Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet"))[ok].reset_index(drop=True)
    T.set_grid(wl); logg = np.log10(T.G_SUN * Pte["s mass"].to_numpy() / Pte["s radius"].to_numpy() ** 2 * 100)
    contam = lambda f: T.contamination(Pte["s temperature"].to_numpy(), logg, f, 0.0)

    def shifted(name):
        Y = np.load(os.path.join(src, f"pop1_native_{name}.npy")).astype(float)[ok]; bad = ~np.isfinite(Y).all(1); Y[bad] = clean_n[bad]; return Y
    out = {}
    for c in CASES:
        if c == "clean" or c.startswith("white_x"): x = clean_n
        elif c.startswith("spots"): x = clean_n * contam(int(c[5:]) / 100)
        elif c == "compound": x = shifted("haze3e7") * contam(0.20)
        else: x = shifted(c)
        out[c] = bin_native(x, wl, e)
    return out


def main():
    _, PQ = MR.scale_height_proxy(); T_HI, T_LO = PQ["clear_max"], PQ["pop_p01"]
    cases, Pte, e, sig = D.test_cases(extra=tuple(c for c in CASES if c not in D.CASES)); wl = np.load(os.path.join(DATA, "native_wl.npy")); nf = noise_free_cases(wl, e)
    sc = D.Scorer(e); mod = Pte["modulation_5H"].to_numpy(float) * 1e6
    y = {m: (Pte[f"atm {m}"] > -4).to_numpy() for m in MOLS}
    ret = {s: joblib.load(os.path.join(DATA, "alfnoor_jwst", f"learners_all_{s}.joblib")) for s in ("haze", "hazedeck")}
    noise2 = np.mean((sig * 1e6) ** 2, axis=1)

    def ratio_observed(X):
        v = np.var(X * 1e6, axis=1) - noise2; r = np.full(len(X), np.inf); pos = v > 0; r[pos] = mod[pos] / np.sqrt(v[pos]); return r
    ratio_nf = lambda c: mod / (nf[c].std(1) * 1e6)
    opt = lambda X: np.abs(sc._optical(AT.norm(X)) - sc.opt_mu) / sc.opt_sd

    def dis(P, m, k):
        p = P["all"][(m, k)]; return np.maximum(np.abs(p - P["no_optical"][(m, k)]), np.abs(p - P["airs_only"][(m, k)]))
    P0 = sc.probs(cases["clean"]); thr_opt = np.quantile(opt(cases["clean"]), 0.9); thr_dis = {(m, k): np.quantile(dis(P0, m, k), 0.9) for m in MOLS for k in LEARNERS}
    rows, rates = [], []
    for c in CASES:
        X = cases[c]; P = P0 if c == "clean" else sc.probs(X); o = opt(X)
        PR = {s: {(m, k): AT.prob(L["models"][("1e-4", m, k)], AJ.znorm(X, L["cols"])) for m in MOLS for k in LEARNERS} for s, L in ret.items()}
        amp = {"observed": ratio_observed(X), "noise-free": ratio_nf(c)}
        rates.append(dict(case=c, **{f"{a} up %": 100 * np.mean(np.isfinite(r) & (r > T_HI)) for a, r in amp.items()}, **{f"{a} low %": 100 * np.mean(r < T_LO) for a, r in amp.items()}, undefined_pct=100 * np.mean(~np.isfinite(amp["observed"])), optical_flag_pct=100 * np.mean(o > thr_opt)))
        for m in MOLS:
            for k in LEARNERS:
                p, pa = P["all"][(m, k)], P["airs_only"][(m, k)]; shape = (o > thr_opt) | (dis(P, m, k) > thr_dis[(m, k)]); ps = np.where(shape, pa, p); t = y[m]
                def rec(name, pp, keep=None):
                    keep = np.ones(len(pp), bool) if keep is None else keep; right = (pp >= 0.5) == t
                    rows.append(dict(case=c, molecule=m, learner=k, pipeline=name, accuracy=100 * right[keep].mean(), coverage=100 * keep.mean(), half_credit=100 * (right[keep].sum() + 0.5 * (~keep).sum()) / len(t), recall=100 * (pp >= 0.5)[keep & t].mean(), shape_flagged=100 * shape.mean()))
                rec("published, all inputs", p); rec("no-optical everywhere", P["no_optical"][(m, k)]); rec("AIRS-only everywhere", pa); rec("shape flag -> AIRS-only", ps)
                for s in ret:
                    pr = PR[s][(m, k)]; rec(f"retrained ({s}) everywhere", pr)
                    for a, r in amp.items():
                        up = np.isfinite(r) & (r > T_HI); low = r < T_LO; pp = np.where(up, pr, ps)
                        rec(f"pipeline ({s}, flag {a})", pp); rec(f"pipeline ({s}, flag {a}), two-sided", pp, ~low)
        print("done", c, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "pipeline_sim.csv"), index=False)
    L = [f"Full pipeline in simulation (pipeline_sim.py): accuracy %, mean over four learners and four molecules, POP-I ({len(Pte)} planets), thresholds T_HI {T_HI:.2f}, T_LO {T_LO:.2f}", ""]
    piv = df.groupby(["pipeline", "case"]).accuracy.mean().unstack("case")[CASES]; order = [p for p in dict.fromkeys(df.pipeline)]
    L.append(f"{'pipeline':<48} " + " ".join(f"{c:>9}" for c in CASES))
    for p_ in order: L.append(f"{p_:<48} " + " ".join(f"{piv.loc[p_, c]:9.1f}" for c in CASES))
    L += ["", "recall of present labels %"]; piv = df.groupby(["pipeline", "case"]).recall.mean().unstack("case")[CASES]
    for p_ in order: L.append(f"{p_:<48} " + " ".join(f"{piv.loc[p_, c]:9.1f}" for c in CASES))
    L += ["", "two-sided rows, coverage % and half-credit accuracy % (a withheld label counts one half)"]
    for p_ in [q for q in order if "two-sided" in q]:
        g = df[df.pipeline == p_].groupby("case")[["coverage", "half_credit"]].mean().loc[CASES]; L.append(f"{p_:<48} " + " ".join(f"{g.coverage[c]:4.0f}/{g.half_credit[c]:4.1f}" for c in CASES))
    g = df[df.pipeline == "shape flag -> AIRS-only"].groupby("case").shape_flagged.mean().loc[CASES]; L += ["", "shape flag fires, % of cells: " + ", ".join(f"{c} {g[c]:.1f}" for c in CASES)]
    L += ["", "amplitude flag and optical flag rates, % of planets"]; L.append(pd.DataFrame(rates).round(1).to_string(index=False))
    txt = "\n".join(L); open(os.path.join(RESULTS, "pipeline_sim.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
