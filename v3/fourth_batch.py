"""The independent test of the frozen routing method (REVISION_EXPECTATIONS.md, section H; rule frozen before any of these labels or spectra entered the code).

The planets and labels are jwst_panel.FOURTH, committed before this script was first run on them. Each reduction goes through the conversion of jwst_panel.run_planet, then through
panel_tier1.score, the function that reproduces the 66-label panel exactly: same learners, same clean thresholds, same retrained designs, and the amplitude thresholds read from POP-I through
the reduction's own mask as in amplitude_flag.py. Nothing is retrained or re-thresholded. Scores: right 1, split or withheld 0.5, wrong 0, at JWST precision and at Tier-1 noise (500 draws).

Writes results/jwst_panel_fourth/{fourth_batch.txt, fourth_batch.csv, <planet>_coverage.csv}.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import amplitude_flag as AF  # noqa: E402
import jwst_panel as JP  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import panel_tier1 as PT  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel_fourth"); os.makedirs(OUT, exist_ok=True); MOLS = PT.MOLS; DESIGNS = [d for d in PT.DESIGNS if not d.startswith("census")]
BASE = PT.BASE; DRAWS = 500; ROOT = os.path.join(HERE, "data", "jwst", "panel")


def convert(planet):
    S, source = JP.segments(ROOT, planet); edges, _ = J.tier3_layout(); ref = max(S, key=lambda k: len(S[k]["wl"])); off = J.align_offsets(S, ref); d, e, cov = J.combine(S, edges, off)
    filled = ~np.isfinite(d) | (cov.covered.to_numpy() < 0.5); d_f = d.copy(); d_f[filled] = np.nanmean(d[~filled]) if (~filled).any() else np.nan
    cov["filled"] = filled; cov["depth"] = d_f; cov["err"] = np.where(filled, np.nan, e); cov.to_csv(os.path.join(OUT, f"{planet}_coverage.csv"), index=False)
    return d_f, filled, np.where(filled, np.nan, e), source, edges


def main():
    Xc, Xo, mod, chk, edges = AF.populations(); ctx = PT.Context(); rows, flags = [], []
    ratio = lambda X, m: mod / (X[:, m].std(1) * 1e6)
    for pl, labs in JP.FOURTH.items():
        x, filled, err, source, _ = convert(pl); meas = ~filled; hi = ratio(Xc, meas).max(); lo = np.percentile(ratio(Xo, meas), 1); s = J.ariel_sigma(BASE(pl), edges)
        j2 = float(np.nanmean(err[meas] ** 2)) * 1e12; t2 = float(np.mean(s[meas] ** 2)) * 1e12
        for prec, X, noise2 in (("JWST", x[None, :], j2), ("Tier-1", J.noise_draws(x, s, DRAWS, seed=405), j2 + t2)):
            res = PT.score(ctx, X, meas, BASE(pl), noise2, hi, lo)
            for m in MOLS:
                votes, low, up = res[m]; lab = labs.get(m)
                for name in DESIGNS:
                    pr = votes[name] >= 3; ab = votes[name] <= 1; sp = ~(pr | ab)
                    if name == "the method": sp = sp | low; pr = pr & ~low; ab = ab & ~low
                    cr = np.nan if lab is None else float(np.mean((pr if lab == "present" else ab) + 0.5 * sp))
                    rows.append(dict(precision=prec, planet=pl, molecule=m, design=name, label=lab, credit=cr, present=float(pr.mean()), split=float(sp.mean()), absent=float(ab.mean()), up=float(up.mean()), low=float(low.mean())))
            if prec == "JWST":
                v = np.var(x[meas] * 1e6) - j2; md = float(ctx.M.loc[BASE(pl), "modulation_5H"]) * 1e6
                flags.append(dict(planet=pl, source=source, filled=int(filled.sum()), measured=int(meas.sum()), measured_optical=int((meas & (0.5 * (edges[1:] + edges[:-1]) < 1.1)).sum()), ratio=(md / np.sqrt(v)) if v > 0 else np.inf, hi=hi, lo=lo, mass_flag=None))
        print(pl, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "fourth_batch.csv"), index=False); Fl = pd.DataFrame(flags).set_index("planet"); Fl.to_csv(os.path.join(OUT, "fourth_batch_flags.csv"))
    L = [f"The frozen routing method on the fourth batch: {len(JP.FOURTH)} reductions; labels committed before this run; right 1, split or withheld 0.5, wrong 0", ""]
    L.append(Fl.drop(columns=["mass_flag", "source"]).round(2).to_string()); L.append("")
    rng = np.random.default_rng(405)
    for prec in ("JWST", "Tier-1"):
        q = df[(df.precision == prec) & df.label.notna()]; piv = q.pivot_table(index=["planet", "molecule"], columns="design", values="credit")[DESIGNS]; plan = np.array([BASE(p) for p in piv.index.get_level_values(0)]); units = sorted(set(plan))
        per = piv.groupby(plan).sum().loc[units]; cnt = pd.Series(plan).value_counts().loc[units].to_numpy(); ix = rng.integers(0, len(units), (20000, len(units))); den = cnt[ix].sum(1); nabs = int((q[q.design == "published"].label == "absent").sum())
        L.append(f"== {prec}: {len(piv)} labels on {len(units)} planets ({nabs} absent, {len(piv) - nabs} present; calling every label absent scores {nabs})")
        for name in DESIGNS:
            a = (per[name].to_numpy()[ix].sum(1) - per["published"].to_numpy()[ix].sum(1)) / den * len(piv); b = (per["the method"].to_numpy()[ix].sum(1) - per[name].to_numpy()[ix].sum(1)) / den * len(piv)
            ps = q[(q.design == name) & (q.label == "present")].credit.sum(); ab = q[(q.design == name) & (q.label == "absent")].credit.sum()
            L.append(f"  {name:<24} {piv[name].sum():6.1f}   vs re-implemented {piv[name].sum() - piv['published'].sum():+5.1f} ({np.percentile(a, 2.5):+5.1f},{np.percentile(a, 97.5):+5.1f})   method minus this {piv['the method'].sum() - piv[name].sum():+5.1f} ({np.percentile(b, 2.5):+5.1f},{np.percentile(b, 97.5):+5.1f})   present {ps:5.1f}  absent {ab:5.1f}")
        L.append("")
    # label by label at JWST precision, and the expectations of section H
    J0 = df[df.precision == "JWST"]; V = J0.assign(v=np.where(J0.low > 0, "withheld", np.where(J0.present == 1, "present", np.where(J0.absent == 1, "absent", "split")))).pivot_table(index=["planet", "molecule", "label"], columns="design", values="v", aggfunc="first")[DESIGNS]
    L.append(V.to_string()); L.append("")
    P_ = V.reset_index(); pub = P_["published"]; met = P_["the method"]; lab = P_["label"]
    ch = P_[pub != met][["planet", "molecule", "label", "published", "the method"]]; L.append("the method against the re-implemented classifier, changed labels:"); L.append(ch.to_string(index=False)); rw = int(((pub == lab) & (met != lab) & (met != "split") & (met != "withheld")).sum())
    tj, tt = [df[(df.precision == p) & df.label.notna()].pivot_table(index=["planet", "molecule"], columns="design", values="credit")[DESIGNS].sum() for p in ("JWST", "Tier-1")]
    L += ["", "Expectations of section H:", f"H1 method above the re-implemented classifier at JWST precision: {tj['the method'] - tj['published']:+.1f}", f"H2 method within two labels of AIRS-only ({tj['the method'] - tj['AIRS-only']:+.1f}) and of haze+deck-retrained ({tj['the method'] - tj['haze+deck-retrained']:+.1f})", f"H3 gain at Tier-1 ({tt['the method'] - tt['published']:+.1f}) smaller than at JWST precision ({tj['the method'] - tj['published']:+.1f})", f"H4 labels from right to wrong: {rw}", "H5, H6: read the tables above"]
    L.append(""); up = df[(df.precision == "Tier-1") & (df.design == "the method") & (df.molecule == "CH4")].set_index("planet"); L.append("amplitude flag, share of Tier-1 draws above the threshold: " + ", ".join(f"{p} {up.up[p]:.2f}" for p in up.index))
    txt = "\n".join(L); open(os.path.join(OUT, "fourth_batch.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
