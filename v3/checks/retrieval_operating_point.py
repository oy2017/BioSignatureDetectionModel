"""AF1 (REVISION_EXPECTATIONS.md, section AF): does the ensemble's lead over the four-gas retrieval on the 90 photochemical-grid
spectra survive when the two are read at a common operating point?

Scoring follows checks/grid_retrieval_intervals.py (finished spectra only; balanced credit = mean of the credit on truly present
and on truly absent labels, pooled over the four gases or per gas). Four comparisons per run (first run, rerun with the
temperature prior fix) and precision:
  1. baseline        retrieval at p_present >= 0.5 against the ensemble at its own thresholds (reproduces the earlier numbers);
  2. matched FP      per gas, the retrieval's threshold on p_present moved so that its false-present rate on truly absent labels
                     equals the ensemble's on the same spectra (linear interpolation between the two bracketing achievable
                     operating points, i.e. a randomised threshold), and the reverse: the ensemble's averaged probability
                     thresholded to the retrieval's false-present rate at 0.5;
  3. oracle          each design at its best per-gas threshold chosen on these test spectra, for that gas's balanced credit (3)
                     and for the pooled balanced credit (3p; the pooled score is linear in each gas's recall and false-present
                     rate, weighted by that gas's share of present and of absent labels, so the pooled optimum is also per gas;
                     it is degenerate here, calling every H2O label present and every NH3 label absent, and is printed only as
                     the formal upper bound). Beside the pooled score, 'macro' is the mean over the four gases of the per-gas
                     balanced credit, the natural score for per-gas thresholds;
  4. prior removed   the retrieval read at Bayes factor > 1 for present, i.e. posterior odds above the prior odds. The
                     manuscript's Section 2.4 prior (log-uniform 10^-9 to 10^-2, label at 10^-4) gives 0.71 absent / 0.29 present,
                     prior odds 5:2, so p_present > 2/7 = 0.286; the grid runs themselves (grid_retrieval.py, AB = (-9, -1)) use
                     10^-9 to 10^-1, 0.625 / 0.375, p_present > 3/8. Both are reported.
Intervals: 95 % bootstrap over planets (the reductions of one planet, -alt/-hu/-hm, resampled together), 2000 replicates,
seed 411, with the matched and oracle thresholds re-chosen in every replicate; a per-gas interval ignores replicates in which
that gas has no truly present (or no truly absent) label, and their number is printed. Writes results/round3/retrieval_operating_point.txt.
"""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOLS = ["CH4", "H2O", "CO2", "NH3"]; G = "results/goyal_test"; O = "results/round3"; NB = 2000; SEED = 411
BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
Y = pd.read_csv(f"{G}/truth_crossfield.csv"); Ve = pd.read_csv(f"{O}/votes_C+B-mlp_crossfield.csv.gz")


def table(tag, prec, pp):
    R = pd.read_csv(f"{O}/grid_retrieval{tag}_{prec}.csv"); R[["planet", "row"]] = R.key.str.split(":", expand=True); R["row"] = R.row.astype(int)
    R = R[R[[f"{m}_p_present" for m in MOLS]].notna().all(1)]; rows = []
    for _, q in R.iterrows():
        t = Y[(Y.planet == q.planet) & (Y.row == q.row)].iloc[0]
        for m in MOLS: rows.append(dict(planet=q.planet, row=q.row, molecule=m, y=bool(t[f"v_{m}"] > -4), p_ret=float(q[f"{m}_p_present"])))
    A = pd.DataFrame(rows).merge(Ve[Ve.precision == pp][["planet", "row", "molecule", "present", "prob"]], on=["planet", "row", "molecule"])
    assert len(A) == 4 * len(R); A["unit"] = A.planet.map(BASE); return A.reset_index(drop=True)


def bal(c, y): return 0.5 * (c[y].mean() + c[~y].mean())


def sweep(p, y):
    """Operating points of the rule 'present if p >= t' over all distinct thresholds, as per-label credit vectors, ordered by rising false-present rate; at equal false-present rate the point with the higher recall is kept."""
    ts = np.concatenate([[np.inf], np.unique(p)[::-1]]); pts = []
    for t in ts:
        v = p >= t; c = np.where(y, v, ~v).astype(float); pts.append((v[~y].mean(), v[y].mean(), t, c))
    out = {}
    for fp, rec, t, c in pts:
        if fp not in out or rec > out[fp][0]: out[fp] = (rec, t, c)
    fps = np.array(sorted(out)); return fps, [out[f] for f in fps]


def at_fp(p, y, target):
    """Credit vector of the threshold rule on p whose false-present rate equals target (interpolating between the two bracketing achievable rates); returns (credits, recall, achieved fp, threshold description)."""
    fps, pts = sweep(p, y); j = np.searchsorted(fps, target)
    if j < len(fps) and np.isclose(fps[j], target): rec, t, c = pts[j]; return c, rec, fps[j], f"{t:.3g}"
    i = j - 1; r0, t0, c0 = pts[i]; r1, t1, c1 = pts[j]; w = (target - fps[i]) / (fps[j] - fps[i])
    return (1 - w) * c0 + w * c1, (1 - w) * r0 + w * r1, target, f"{t0:.3g}/{t1:.3g} w={w:.2f}"


def oracle(p, y, wp=1.0, wa=1.0):
    """Threshold maximising wp * recall - wa * false-present rate (wp = wa = 1: this gas's balanced credit; wp, wa = the gas's share of present and of absent labels: the pooled balanced credit). With no present label the recall term is dropped."""
    fps, pts = sweep(p, y); b = [wp * (r if np.isfinite(r) else 0.0) - wa * f for f, (r, t, c) in zip(fps, pts)]; k = int(np.argmax(b)); return pts[k][2], pts[k][0], fps[k], pts[k][1]


def credits(A):
    """Per-label credit vectors for every design and operating point, plus the chosen thresholds (point estimate only)."""
    y = A.y.to_numpy(); pr = A.p_ret.to_numpy(); pe = A.prob.to_numpy(); ep = A.present.to_numpy() > 0.5; m = A.molecule.to_numpy()
    C = {"ret_0.5": np.where(y, pr >= 0.5, pr < 0.5).astype(float), "ens_own": np.where(y, ep, ~ep).astype(float),
         "ret_bf1_2/7": np.where(y, pr > 2 / 7, pr <= 2 / 7).astype(float), "ret_bf1_3/8": np.where(y, pr > 3 / 8, pr <= 3 / 8).astype(float)}
    for k in ("ret_matched", "ens_matched", "ret_oracle", "ens_oracle", "ret_poracle", "ens_poracle"): C[k] = np.zeros(len(A))
    info = {}; Np, Na = y.sum(), (~y).sum()
    for g in MOLS:
        s = m == g; yg = y[s]; f_e = ep[s][~yg].mean(); f_r = (pr[s][~yg] >= 0.5).mean(); wp, wa = yg.sum() / Np, (~yg).sum() / Na
        c, rec, fa, td = at_fp(pr[s], yg, f_e); C["ret_matched"][s] = c; info[g, "ret_matched"] = (td, fa, rec)
        c, rec, fa, td = at_fp(pe[s], yg, f_r); C["ens_matched"][s] = c; info[g, "ens_matched"] = (td, fa, rec)
        c, rec, fa, t = oracle(pr[s], yg); C["ret_oracle"][s] = c; info[g, "ret_oracle"] = (f"{t:.3g}", fa, rec)
        c, rec, fa, t = oracle(pe[s], yg); C["ens_oracle"][s] = c; info[g, "ens_oracle"] = (f"{t:.3g}", fa, rec)
        c, rec, fa, t = oracle(pr[s], yg, wp, wa); C["ret_poracle"][s] = c; info[g, "ret_poracle"] = (f"{t:.3g}", fa, rec)
        c, rec, fa, t = oracle(pe[s], yg, wp, wa); C["ens_poracle"][s] = c; info[g, "ens_poracle"] = (f"{t:.3g}", fa, rec)
        info[g, "ens_own"] = ("own", f_e, ep[s][yg].mean()); info[g, "ret_0.5"] = ("0.5", f_r, (pr[s][yg] >= 0.5).mean())
        for k, t in (("ret_bf1_2/7", 2 / 7), ("ret_bf1_3/8", 3 / 8)): info[g, k] = (f"{t:.3f}", (pr[s][~yg] > t).mean(), (pr[s][yg] > t).mean())
    return C, info


COMP = [("1 baseline", "ens_own", "ret_0.5"), ("2a matched FP (retrieval moved to ensemble's rate)", "ens_own", "ret_matched"),
        ("2b matched FP (ensemble moved to retrieval's rate at 0.5)", "ens_matched", "ret_0.5"), ("3 oracle thresholds (both in-sample)", "ens_oracle", "ret_oracle"),
        ("3p pooled-credit oracle thresholds (both in-sample)", "ens_poracle", "ret_poracle"), ("3' retrieval pooled oracle against ensemble's own", "ens_own", "ret_poracle"),
        ("4 prior removed, p > 2/7 (manuscript 0.71/0.29)", "ens_own", "ret_bf1_2/7"),
        ("4' prior removed, p > 3/8 (grid run's 10^-9 to 10^-1 prior)", "ens_own", "ret_bf1_3/8")]
L = [__doc__.strip(), ""]
for tag, run in (("", "first run"), ("_tp", "rerun (T prior fix)")):
    for prec, pp in (("jwst", "JWST"), ("tier1", "Tier-1")):
        A = table(tag, prec, pp); y = A.y.to_numpy(); m = A.molecule.to_numpy(); C, info = credits(A)
        units = A.unit.unique(); idx = {u: A.index[A.unit == u].to_numpy() for u in units}; rng = np.random.default_rng(SEED)
        L.append(f"=== {run}, {pp}: {A[['planet', 'row']].drop_duplicates().shape[0]} finished spectra, {A.planet.nunique()} planets, {len(units)} bootstrap units (base planets); present/absent labels "
                 + ", ".join(f"{g} {int(y[m == g].sum())}/{int((~y[m == g]).sum())}" for g in MOLS) + " ===")
        L.append("  retrieval posterior saturation: share of absent labels with p_present >= 0.99 / of present labels with p_present <= 0.01: "
                 + ", ".join(f"{g} {(A.p_ret[(m == g) & ~y] >= 0.99).mean():.2f}/{(A.p_ret[(m == g) & y] <= 0.01).mean():.2f}" for g in MOLS))
        L.append("  operating points (threshold, false-present rate on absent labels, recall of present labels):")
        for k in ("ret_0.5", "ens_own", "ret_matched", "ens_matched", "ret_oracle", "ens_oracle", "ret_poracle", "ens_poracle", "ret_bf1_2/7", "ret_bf1_3/8"):
            L.append(f"    {k:<12}" + "  ".join(f"{g}: t={info[g, k][0]:<14} fp={info[g, k][1]:.3f} rec={info[g, k][2]:.3f}" for g in MOLS))
        # bootstrap: every replicate re-chooses the matched and oracle thresholds on the resampled labels
        B = {name: {"pooled": [], **{g: [] for g in MOLS}} for name, _, _ in COMP}
        for _ in range(NB):
            S = A.loc[np.concatenate([idx[u] for u in rng.choice(units, len(units))])].reset_index(drop=True); ys = S.y.to_numpy(); ms = S.molecule.to_numpy(); Cs, _ = credits(S)
            for name, e, r in COMP:
                d = Cs[e] - Cs[r]; B[name]["pooled"].append(bal(d, ys))
                for g in MOLS: B[name][g].append(bal(d[ms == g], ys[ms == g]))
        for name, e, r in COMP:
            lo, hi = np.percentile(B[name]["pooled"], [2.5, 97.5]); d = C[e] - C[r]
            mac = np.nanmean(np.array([B[name][g] for g in MOLS]), axis=0); mlo, mhi = np.percentile(mac, [2.5, 97.5])
            me, mr = (np.mean([bal(C[k][m == g], y[m == g]) for g in MOLS]) for k in (e, r))
            L.append(f"  {name:<58} ensemble {bal(C[e], y):.3f}  retrieval {bal(C[r], y):.3f}  gap {bal(d, y):+.3f} ({lo:+.3f}, {hi:+.3f})   macro: ensemble {me:.3f}  retrieval {mr:.3f}  gap {me - mr:+.3f} ({mlo:+.3f}, {mhi:+.3f})")
            L.append("      per gas: " + "  ".join(f"{g} ens {bal(C[e][m == g], y[m == g]):.3f} ret {bal(C[r][m == g], y[m == g]):.3f} gap {bal(d[m == g], y[m == g]):+.3f} ({np.nanpercentile(B[name][g], 2.5):+.3f}, {np.nanpercentile(B[name][g], 97.5):+.3f})" for g in MOLS))
        dropped = {g: int(np.isnan(B[COMP[0][0]][g]).sum()) for g in MOLS}
        if any(dropped.values()): L.append("  replicates without a present or an absent label, left out of that gas's interval: " + ", ".join(f"{g} {n}" for g, n in dropped.items() if n))
        L.append("")
t = "\n".join(L); open(f"{O}/retrieval_operating_point.txt", "w").write(t + "\n"); print(t)
