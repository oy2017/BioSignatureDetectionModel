"""What makes the classifier miss on real spectra? Two tests of statements the draft made without testing them (review of 2026-10-30).

1. Transplants on the real reductions. In each reduction with a clear twin, the all-input learners read the real spectrum with one block of Tier-3 bins replaced by the twin's
   (shifted to the real mean over the measured AIRS bins): the three optical points, the eleven bins of Ariel's NIRSpec channel, both, or the AIRS bins. If the miss enters through the
   optical points, replacing them repairs it.
2. Scaled twins. The clear twin is scaled about its mean so that its dispersion over the measured bins equals the real noise-corrected dispersion, filled flat through the same mask, and
   scattered 200 times with the real per-bin errors. If the miss comes from amplitude alone, the scaled twin is missed too; per-spectrum normalisation removes pure scaling, so only the noise can do it.

Writes results/jwst_panel/panel_mechanism.{txt,csv}. Verdict: majority of the four learners at 1e-4; learners calling the label present are listed.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import jwst_muting_census as C  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import router_real as RR  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel"); MOLS, LEARNERS = D.MOLS, D.LEARNERS
BLOCKS = {"optical points": lambda c, meas: c < 1.1, "NIRSpec channel": lambda c, meas: (c > 1.1) & (c < 1.95), "both, all below 1.95 um": lambda c, meas: c < 1.95, "AIRS bins": lambda c, meas: c > 1.95,
          "filled bins below 1.95 um": lambda c, meas: (c < 1.95) & ~meas, "measured bins below 1.95 um": lambda c, meas: (c < 1.95) & meas}
ORDER = ["real", "optical points", "NIRSpec channel", "both, all below 1.95 um", "filled bins below 1.95 um", "measured bins below 1.95 um", "AIRS bins"]


def main(n=200):
    edges = C.edges; cen = 0.5 * (edges[1:] + edges[:-1]); sc = D.Scorer(edges); M = C.M; rows, scaled = [], []
    for pl, ab in C.TWINS.items():
        if pl not in RR.PANEL: continue
        d = RR.PANEL[pl]; cov = pd.read_csv(d["coverage"]); x = cov.depth.to_numpy().copy(); filled = cov.filled.to_numpy(bool); meas = ~filled; x[filled] = np.nanmean(x[meas]); err = cov.err.to_numpy()
        r = M[M.name == pl.replace("-alt", "").replace("-hu", "").replace("-hm", "")].iloc[0].to_dict(); r.update({"atm top_pressure": 1e-4, "atm CO": -15.0, "atm O3": -15.0, **{f"atm {k}": v for k, v in ab.items()}})
        t = C.render(r)
        if t is None: continue
        airs = meas & (cen > 1.95); t = t - np.mean(t[airs]) + np.mean(x[airs]) if airs.sum() > 2 else t - np.mean(t[meas]) + np.mean(x[meas])
        variants = {"real": x}
        for name, f in BLOCKS.items():
            v = x.copy(); v[f(cen, meas)] = t[f(cen, meas)]; variants[name] = v
        P = sc.probs(np.vstack(list(variants.values())))
        for i, name in enumerate(variants):
            for m in MOLS:
                rows.append(dict(planet=pl, molecule=m, label=d["labels"].get(m), variant=name, learners_present=int(sum(P["all"][(m, k)][i] >= 0.5 for k in LEARNERS)), measured_optical=int((meas & (cen < 1.1)).sum()), measured_nirspec=int((meas & (cen > 1.1) & (cen < 1.95)).sum()), measured_airs=int(airs.sum())))
        s_real = np.std(x[meas]); noise = np.sqrt(np.nanmean(err[meas] ** 2)); s_corr = np.sqrt(max(s_real ** 2 - noise ** 2, 0.0)); s_twin = np.std(t[meas])
        if s_corr > 0:
            sT = J.ariel_sigma(pl.replace("-alt", "").replace("-hu", "").replace("-hm", ""), edges)
            for tag, f in (("twin", 1.0), ("scaled twin", s_corr / s_twin)):
                base = t.copy(); mu = np.mean(base[meas]); base = mu + (base - mu) * f; base[filled] = np.mean(base[meas])
                e = np.where(meas, np.nan_to_num(err), 0.0)
                for prec, noise_ in (("JWST", e), ("Tier-1", np.sqrt(e ** 2 + sT ** 2))):
                    rng = np.random.default_rng(405); X = base[None, :] + rng.normal(0, 1, (n, len(base))) * noise_
                    Pz = sc.probs(X)
                    for m in MOLS:
                        for var in ("all", "airs_only"):
                            v = sum((Pz[var][(m, k)] >= 0.5).astype(int) for k in LEARNERS)
                            scaled.append(dict(planet=pl, molecule=m, label=d["labels"].get(m), test=tag, precision=prec, variant=var, scale=f, present_freq=float((v >= 3).mean()), split_freq=float((v == 2).mean())))
        print(pl, flush=True)
    A = pd.DataFrame(rows); B = pd.DataFrame(scaled); A.to_csv(os.path.join(OUT, "panel_mechanism_transplants.csv"), index=False); B.to_csv(os.path.join(OUT, "panel_mechanism_scaled.csv"), index=False)
    L = ["Transplants on the real reductions: learners of four calling each present label present, with one block replaced by the clear twin's (all-input design)", ""]
    q = A[A.label == "present"].pivot_table(index=["planet", "molecule", "measured_optical", "measured_nirspec", "measured_airs"], columns="variant", values="learners_present")[ORDER].astype(int)
    L.append(q.to_string()); L.append("")
    for mol, g in q.groupby(level="molecule"):
        L.append(f"{mol}: of {len(g)} present labels, majority (at least 3) with: " + ", ".join(f"{c} {int((g[c] >= 3).sum())}" for c in g.columns))
    L.append(f"all: of {len(q)}: " + ", ".join(f"{c} {int((q[c] >= 3).sum())}" for c in q.columns)); L.append("")
    m1 = q[q["real"] < 3]; L.append(f"of the {len(m1)} present labels the real spectrum misses, repaired (at least 3) by: " + ", ".join(f"{c} {int((m1[c] >= 3).sum())}" for c in m1.columns[1:]))
    m2 = m1.xs("CH4", level="molecule"); L.append(f"of the {len(m2)} methane misses with a twin, repaired by: " + ", ".join(f"{c} {int((m2[c] >= 3).sum())}" for c in m2.columns[1:])); L.append("")
    for prec in ("JWST", "Tier-1"):
        L.append(f"Scaled twins at {prec} noise: detection frequency of each present label, clear twin through the mask against the twin scaled to the real dispersion (all-input and AIRS-only)")
        w = B[(B.label == "present") & (B.precision == prec)].pivot_table(index=["planet", "molecule"], columns=["variant", "test"], values="present_freq"); L.append(w.round(2).to_string()); L.append("")
        for var in ("all", "airs_only"):
            a = w[(var, "twin")]; b = w[(var, "scaled twin")]; L.append(f"{prec}, {var}: mean detection frequency twin {a.mean():.2f}, scaled twin {b.mean():.2f}; labels found (>= 0.5): twin {int((a >= .5).sum())}, scaled {int((b >= .5).sum())} of {len(a)}")
        L.append("")
    txt = "\n".join(L); open(os.path.join(OUT, "panel_mechanism.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
