"""AD7 (REVISION_EXPECTATIONS.md, section AD): on the two held-out grids at both precisions, the routing method's false-present
rate on truly absent labels and its recall on truly present labels (per gas and pooled); then the matched ensemble's recall
when its probability threshold is moved, per gas, so that its false-present rate equals the routing method's for that gas.

The routing method's verdicts have no split on the grids (present is 0 or 1). The ensemble's threshold for a gas is the
lowest threshold t on its averaged probability (votes_C+B-mlp_<grid>.csv.gz, prob) with false-present (prob >= t on
absent labels) at or below the routing method's rate; the achieved rate is printed. Pooled values pool the four gases with
these per-gas thresholds. Intervals: 95 % bootstrap over planets (base planet, 4000 resamples, seed 405) of the recall
difference, with the thresholds re-matched in every resample (and, beside it, with the thresholds held fixed).

Writes results/round3/matched_fp_recall.txt.
"""
import os

import numpy as np
import pandas as pd

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(V3)
MOLS = ["CH4", "H2O", "CO2", "NH3"]; BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
GRIDS = [("local", "local condensation"), ("crossfield", "photochemical")]; NB = 4000


def load(tag):
    R = pd.read_csv(f"results/goyal_test/votes_frozen_{tag}.csv.gz"); R = R[R.design == "the method"][["planet", "precision", "molecule", "row", "present", "absent"]]
    E = pd.read_csv(f"results/round3/votes_C+B-mlp_{tag}.csv.gz")[["planet", "precision", "molecule", "row", "prob", "present"]].rename(columns={"present": "ens_present"})
    Y = pd.read_csv(f"results/goyal_test/truth_{tag}.csv").melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in MOLS], var_name="molecule", value_name="v"); Y["molecule"] = Y.molecule.str[2:]
    A = R.merge(E, on=["planet", "precision", "molecule", "row"]).merge(Y, on=["planet", "row", "molecule"]); assert len(A) == len(R) == len(E); A["y"] = A.v > -4; A["u"] = A.planet.map(BASE); return A


def match(pa, target):
    """Lowest threshold whose false-present rate on the absent probabilities pa is <= target."""
    s = np.sort(pa)[::-1]; k = int(np.floor(target * len(s) + 1e-9))   # at most k absent labels may be called present
    if k >= len(s): return -np.inf
    t = np.nextafter(s[k], np.inf)                                      # strictly above the (k+1)-th largest absent probability
    return t


def main():
    L = ["AD7: recall of truly present labels at the routing method's false-present rate (per-gas thresholds on the matched ensemble's probability).", ""]; summ = []
    for tag, gname in GRIDS:
        A = load(tag); us = sorted(A.u.unique()); nu = len(us); ix = np.random.default_rng(405).integers(0, nu, (NB, nu)); ui = {u: i for i, u in enumerate(us)}
        for prec in ("JWST", "Tier-1"):
            q = A[A.precision == prec]; assert set(q.present.unique()) <= {0.0, 1.0}
            L.append(f"== {gname}, {prec}"); L.append(f"  {'gas':<6} {'n pres':>6} {'n abs':>6} | {'routing FP':>10} {'routing recall':>14} | {'ens FP (own t)':>14} {'ens recall (own t)':>18} | {'matched t':>9} {'ens FP':>7} {'ens recall':>10} | {'diff':>6} (95 %, rematched) (fixed t)")
            parts = {}; tot = dict(rtp=0, rfp=0, etp=0, efp=0, np=0, na=0)
            for m in MOLS + ["pooled"]:
                if m != "pooled":
                    g = q[q.molecule == m]; y = g.y.to_numpy(); pr = g.present.to_numpy() > 0.5; p = g.prob.to_numpy(); u = g.u.map(ui).to_numpy(); fp_r = pr[~y].mean(); rec_r = pr[y].mean()
                    t = match(p[~y], fp_r); call = p >= t; parts[m] = (y, pr, p, u, call)
                    for k, v in (("rtp", (pr & y).sum()), ("rfp", (pr & ~y).sum()), ("etp", (call & y).sum()), ("efp", (call & ~y).sum()), ("np", y.sum()), ("na", (~y).sum())): tot[k] += v
                    own = g.ens_present.to_numpy() > 0.5; row = (y.sum(), (~y).sum(), fp_r, rec_r, own[~y].mean(), own[y].mean(), t, call[~y].mean(), call[y].mean())
                else:
                    row = (tot["np"], tot["na"], tot["rfp"] / tot["na"], tot["rtp"] / tot["np"], (q.ens_present[~q.y] > 0.5).mean(), (q.ens_present[q.y] > 0.5).mean(), np.nan, tot["efp"] / tot["na"], tot["etp"] / tot["np"])
                # bootstrap over planets: counts per planet, thresholds re-matched in each resample
                ms = MOLS if m == "pooled" else [m]; dre, dfx = np.zeros(NB), np.zeros(NB); B = np.zeros((NB, 4))
                for mm in ms:
                    y, pr, p, u, call = parts[mm]
                    cnt = lambda w: np.stack([np.bincount(u, weights=w.astype(float), minlength=nu)]); W = np.zeros((NB, nu)); np.add.at(W, (np.repeat(np.arange(NB), nu), ix.ravel()), 1)
                    rtp, rfp, npos, nab = (W @ cnt(pr & y).T)[:, 0], (W @ cnt(pr & ~y).T)[:, 0], (W @ cnt(y).T)[:, 0], (W @ cnt(~y).T)[:, 0]; ftp = (W @ cnt(call & y).T)[:, 0]
                    # rematched: for each resample, the threshold giving the resampled routing FP on the resampled absent probabilities
                    pa_order = np.argsort(-p[~y]); pa_sorted = p[~y][pa_order]; ua = u[~y][pa_order]; Wa = W[:, ua]; cum_abs = np.cumsum(Wa, 1)   # weighted count of absent labels with prob >= each sorted value
                    k = np.floor(rfp + 1e-9); pos = np.array([np.searchsorted(cum_abs[b], k[b], side="right") for b in range(NB)]); tb = np.where(pos >= len(pa_sorted), -np.inf, np.nextafter(pa_sorted[np.minimum(pos, len(pa_sorted) - 1)], np.inf))
                    # ties: searchsorted on weights may land inside a run of equal probabilities; the strict threshold above that value keeps FP <= target
                    yp = p[y]; upos = u[y]; Wp = W[:, upos]; etp = (Wp * (yp[None, :] >= tb[:, None])).sum(1)
                    B += np.column_stack([rtp, npos, etp, ftp])
                dre = (B[:, 2] - B[:, 0]) / B[:, 1]; dfx = (B[:, 3] - B[:, 0]) / B[:, 1]; d = row[8] - row[3]
                L.append(f"  {m:<6} {row[0]:>6} {row[1]:>6} | {row[2]:>10.3f} {row[3]:>14.3f} | {row[4]:>14.3f} {row[5]:>18.3f} | {row[6]:>9.4f} {row[7]:>7.3f} {row[8]:>10.3f} | {d:+.3f} ({np.percentile(dre, 2.5):+.3f} to {np.percentile(dre, 97.5):+.3f}) ({np.percentile(dfx, 2.5):+.3f} to {np.percentile(dfx, 97.5):+.3f})")
                if m == "pooled": summ.append((gname, prec, row[2], row[3], row[7], row[8], d, np.percentile(dre, 2.5), np.percentile(dre, 97.5)))
                else: summ.append((gname, prec, m, row[3], row[8]))
            L.append("")
    P = [s for s in summ if len(s) == 9]; G = [s for s in summ if len(s) == 5]
    L.append("AD7 check: at the routing method's false-present rate, ensemble recall > routing recall")
    for g, p, fr, rr, fe, re_, d, lo, hi in P: L.append(f"  {g:<20} {p:<7} pooled FP {fr:.3f} (ens {fe:.3f})  recall routing {rr:.3f}  ensemble {re_:.3f}  diff {d:+.3f} ({lo:+.3f} to {hi:+.3f})")
    L.append(f"  pooled: ensemble higher in {sum(s[6] > 0 for s in P)} of 4 (interval above zero in {sum(s[7] > 0 for s in P)}); per gas: higher in {sum(s[4] > s[3] for s in G)} of {len(G)}"
             + ("; not higher: " + ", ".join(f"{s[0]} {s[1]} {s[2]} ({s[3]:.3f} vs {s[4]:.3f})" for s in G if not s[4] > s[3]) if any(not s[4] > s[3] for s in G) else ""))
    L.append(f"  AD7 {'met' if all(s[6] > 0 for s in P) else 'not met'} (pooled)")
    t = "\n".join(L); open("results/round3/matched_fp_recall.txt", "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
