"""Is the budgeted fallback's model choice oracle-assisted, and what would an operational version cost?

In e3_budget.py the planets a rule routes to the retrieval are re-scored by the free-haze retrieval of the hazed
spectrum when they are truly hazed, and by the clean-model retrieval of the clean spectrum when they are truly clean.
A pipeline does not know which routed planets are clean. The free-haze retrieval of the clean spectrum exists for 27
planets (identifiability_calibration_freehaze.csv). This check (REVISION_EXPECTATIONS.md row AF4):
  1. on those 27 planets compares the clean-model and the free-haze retrieval of the clean spectrum;
  2. replays the q = 0.2 draws of e3_budget.py (same seed and consumption order) and, at f = 20 %, counts the routed
     clean planets per ordering, how many of them have a free-haze run, and re-scores the targets lost with the
     free-haze retrieval substituted wherever it exists, with a bound for the rest from the agreement rate of step 1;
  3. counts the free-haze retrievals of clean spectra that would make the f = 20 % column fully operational.
Usage: OMP_NUM_THREADS=1 python checks/fallback_model_choice.py   ->  results/fallback_model_choice.txt
Reads only; writes nothing but the output file.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import e3_budget as E  # noqa: E402

MOLS = D.MOLS; BUDGET = D.BUDGET; RULES = E.RULES; FS = E.FS; H = "haze3e7"; F_SHOW = 0.2; Q_SHOW = 0.2
QS, DRAWS = [0.1, 0.2, 0.5], 20
NICE = {"random": "Random", "optical": "Optical-excess score", "disagree": "Input disagreement", "either": "Either (optical or disagreement)",
        "bandmetric": "Band metric", "margin": "Margin", "ensemble": "Learner-spread", "mahalanobis": "Mahalanobis", "knn": "k-NN distance",
        "pca_recon": "PCA", "oracle (knows the hazed)": "Haze oracle"}
L = []


def say(s=""):
    print(s, flush=True); L.append(s)


def main():
    # ---- 1. the 27 clean planets with both retrievals ------------------------------------------------------
    A = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv")).drop_duplicates("name")
    C = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration_freehaze.csv")).drop_duplicates("name")
    if "error" in C: C = C[C["error"].isna()]
    M = C.merge(A, on="name", suffixes=("_fh", "_cl")); n27 = len(M)
    say(f"1. Clean spectra retrieved by both the clean model and the free-haze model: {n27} planets, {4 * n27} labels")
    say(f"   (free-haze runs: median {C.minutes.median():.1f} min, mean {C.minutes.mean():.1f}; clean-model runs of the same planets: "
        f"median {M.minutes_cl.median():.1f} min)")
    say(f"   {'gas':<5} {'agree':>6} {'n':>4}   {'acc clean':>9} {'acc FH':>7}   {'decid clean':>11} {'decid FH':>8}   {'mean dp_true':>12} {'median dp_true':>14}  flips (clean->FH: label, truth)")
    from scipy.stats import beta
    agree = {}; ub = {}; tot = dict(agree=0, n=0, acc_c=0, acc_f=0, dec_c=0, dec_f=0, dp=[])
    for m in MOLS:
        pc, pf = M[f"{m}_p_present_cl"].to_numpy(), M[f"{m}_p_present_fh"].to_numpy(); tc, tf = M[f"{m}_p_true_cl"].to_numpy(), M[f"{m}_p_true_fh"].to_numpy()
        truth = M[f"{m}_true_cl"].to_numpy() > -4
        lc, lf = pc >= 0.5, pf >= 0.5; ag = lc == lf; agree[m] = ag.mean()
        ub[m] = float(beta.ppf(0.95, (~ag).sum() + 1, ag.sum()))   # Clopper-Pearson one-sided 95 % upper bound on the disagreement rate
        acc_c, acc_f = (lc == truth).mean(), (lf == truth).mean(); dec_c, dec_f = (tc > 0.95).mean(), (tf > 0.95).mean()
        flips = [f"{M.name.iloc[i]} ({int(lc[i])}->{int(lf[i])}, truth {int(truth[i])})" for i in np.where(~ag)[0]]
        say(f"   {m:<5} {100*ag.mean():5.1f}% {n27:>4}   {100*acc_c:8.1f}% {100*acc_f:6.1f}%   {100*dec_c:10.1f}% {100*dec_f:7.1f}%   {np.mean(tf-tc):+12.3f} {np.median(tf-tc):+14.3f}  {'; '.join(flips) or '-'}")
        tot["agree"] += ag.sum(); tot["n"] += n27; tot["acc_c"] += (lc == truth).sum(); tot["acc_f"] += (lf == truth).sum()
        tot["dec_c"] += (tc > 0.95).sum(); tot["dec_f"] += (tf > 0.95).sum(); tot["dp"] += list(tf - tc)
    agree_all = tot["agree"] / tot["n"]
    say(f"   {'all':<5} {100*agree_all:5.1f}% {tot['n']:>4}   {100*tot['acc_c']/tot['n']:8.1f}% {100*tot['acc_f']/tot['n']:6.1f}%   "
        f"{100*tot['dec_c']/tot['n']:10.1f}% {100*tot['dec_f']/tot['n']:7.1f}%   {np.mean(tot['dp']):+12.3f} {np.median(tot['dp']):+14.3f}")
    say(f"   decidable at 0.95 by the clean model but not by the free-haze model: {sum(int(((M[f'{m}_p_true_cl']>0.95)&~(M[f'{m}_p_true_fh']>0.95)).sum()) for m in MOLS)} labels; "
        f"the reverse: {sum(int((~(M[f'{m}_p_true_cl']>0.95)&(M[f'{m}_p_true_fh']>0.95)).sum()) for m in MOLS)} labels")
    say(f"   row AF4 asks for agreement >= 95 % of the 108 labels: {'met' if agree_all >= 0.95 else 'NOT met'} ({tot['agree']} of {tot['n']})")
    say("   one-sided 95 % upper bound on the per-label disagreement rate (Clopper-Pearson, 27 planets): " + ", ".join(f"{m} {100*ub[m]:.1f} %" for m in MOLS))
    say()

    # ---- 2. replay e3_budget.py (fallback freehaze, haze3e7, q 0.1 0.2 0.5, 20 draws) ---------------------
    say("2. Replay of e3_budget.py --fallback freehaze --q 0.1 0.2 0.5 --draws 20 (same seed and random-number order), q = 0.2, f = 20 %")
    cases, Pte, e, sig = D.test_cases(extra=[H]); sc = D.Scorer(e); n = len(Pte)
    X = {c: cases[c] for c in ("clean", H)}; S, P = {}, {}
    for c in X:
        S[c], P[c], _ = sc.scores(X[c])
    qq = lambda s: np.quantile(s[np.isfinite(s)], 1 - D.DECLINE)
    for c in X:
        for m in MOLS:
            for k in ("KNN", "MLP", "RFC", "SVC"):
                S[c][("either", (m, k))] = np.fmax(S[c][("optical", None)] / qq(S["clean"][("optical", None)]), S[c][("disagree", (m, k))] / qq(S["clean"][("disagree", (m, k))]))
    y_true = {m: (Pte[f"atm {m}"].to_numpy() > D.THRESH if hasattr(D, "THRESH") else Pte[f"atm {m}"].to_numpy() > -4).astype(int) for m in MOLS}
    R_h, n_h = E.posteriors(Pte, f"identifiability_calibration_{H}_freehaze.csv")
    R_c, n_c = E.posteriors(Pte, "identifiability_calibration.csv")           # what e3_budget used on the clean side (27 < 0.9 * 954)
    R_fh, n_fh = E.posteriors(Pte, "identifiability_calibration_freehaze.csv")  # the free-haze retrieval of the clean spectrum, 27 planets
    have = np.isfinite(R_h["CH4"]) & np.isfinite(R_c["CH4"]); idx = np.where(have)[0]; N = len(idx); slots = int(round(BUDGET * N))
    names = Pte.name.to_numpy()[idx]; has_fh = np.isfinite(R_fh["CH4"])[idx]
    say(f"   population {N} planets (posteriors on {n_h} hazed and {n_c} clean spectra); {int(has_fh.sum())} have a free-haze retrieval of the clean spectrum; "
        f"{slots} Tier-2 slots")
    rng = np.random.default_rng(SEED)
    cls = {c: {m: np.mean([P[c]["all"][(m, k)] for k in ("KNN", "MLP", "RFC")], axis=0) for m in MOLS} for c in X}
    rule_s = {c: {r: (np.nanmean(np.column_stack([sc.get(S[c], r, m, k) for m in MOLS for k in ("KNN", "MLP", "RFC")]), axis=1)
                      if r not in ("random", "oracle (knows the hazed)") else None) for r in RULES} for c in X}
    rows = []; union = {r: set() for r in RULES}; union_all = set(); per_draw_union = []
    for q in QS:
        for d in range(DRAWS):
            h = rng.uniform(0, 1, N) < q
            pick = lambda A_c, A_h: np.where(h, A_h[idx], A_c[idx])
            routed_this_draw = set()
            for target in ("CH4", "H2O", "joint"):
                mols = ["CH4", "H2O"] if target == "joint" else [target]
                y = np.clip(sum(y_true[m][idx] for m in mols), 0, 1); ntar = int(y.sum())
                score = sum(pick(cls["clean"][m], cls[H][m]) for m in mols)
                ret = sum(pick(R_c[m], R_h[m]) for m in mols)                       # oracle-assisted: clean retrieval on clean planets
                ret_op = sum(pick(np.where(np.isfinite(R_fh[m]), R_fh[m], R_c[m]), R_h[m]) for m in mols)   # free-haze where it exists
                def lost(s_):
                    sel = np.zeros(N, bool); sel[np.argsort(-s_)[:slots]] = True; return int(ntar - (sel & (y == 1)).sum())
                for r in RULES:
                    if r == "random": s = rng.uniform(0, 1, N)
                    elif r.startswith("oracle"): s = h + rng.uniform(0, 1e-3, N)
                    else: s = pick(rule_s["clean"][r], rule_s[H][r]); s = np.where(np.isfinite(s), s, -np.inf)
                    order = np.argsort(-s)
                    for f in FS[1:-1]:
                        if q != Q_SHOW or f != F_SHOW:
                            continue
                        flag = np.zeros(N, bool); flag[order[:int(round(f * N))]] = True
                        rc = flag & ~h; rc_fh = rc & has_fh; rc_miss = rc & ~has_fh
                        s_or = np.where(flag & np.isfinite(ret), ret, score); s_op = np.where(flag & np.isfinite(ret_op), ret_op, score)
                        # routed clean planets whose substituted free-haze posterior changes the label at 0.5 (per target molecule)
                        flip = sum(int((rc_fh & ((R_fh[m][idx] >= 0.5) != (R_c[m][idx] >= 0.5))).sum()) for m in mols)
                        rows.append(dict(q=q, draw=d, target=target, rule=r, f=f, targets=ntar, routed=int(flag.sum()), routed_hazed=int((flag & h).sum()),
                                         routed_clean=int(rc.sum()), routed_clean_fh=int(rc_fh.sum()), routed_clean_missing=int(rc_miss.sum()),
                                         flips_known=flip, lost_oracle=lost(s_or), lost_sub=lost(s_op)))
                        union[r] |= set(names[rc]); union_all |= set(names[rc]); routed_this_draw |= set(names[rc])
            if q == Q_SHOW: per_draw_union.append(len(routed_this_draw))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "fallback_model_choice.csv"), index=False)
    for target in ("CH4", "H2O", "joint"):
        mols = ["CH4", "H2O"] if target == "joint" else [target]
        dis = sum(1 - agree[m] for m in mols)   # expected label flips per routed clean planet without a free-haze run (summed over the target's molecules)
        dis_ub = sum(ub[m] for m in mols)       # the same at the 95 % upper bound of the disagreement rate
        g = df[(df.target == target) & (df.q == Q_SHOW) & (df.f == F_SHOW)].groupby("rule", sort=False).agg(
            targets=("targets", "mean"), routed=("routed", "mean"), rh=("routed_hazed", "mean"), rc=("routed_clean", "mean"), rcf=("routed_clean_fh", "mean"),
            rcm=("routed_clean_missing", "mean"), flips=("flips_known", "mean"), lo=("lost_oracle", "mean"), lo_sd=("lost_oracle", "std"),
            ls=("lost_sub", "mean"), ls_sd=("lost_sub", "std")).reset_index()
        g["d"] = df[(df.target == target) & (df.q == Q_SHOW) & (df.f == F_SHOW)].assign(d=lambda t: t.lost_sub - t.lost_oracle).groupby("rule", sort=False).d.mean().to_numpy()
        g["dabs"] = df[(df.target == target) & (df.q == Q_SHOW) & (df.f == F_SHOW)].assign(d=lambda t: (t.lost_sub - t.lost_oracle).abs()).groupby("rule", sort=False).d.max().to_numpy()
        say(f"   == target {target} ({g.targets.iloc[0]:.0f} targets), q = {Q_SHOW}, f = {F_SHOW:.0%}: mean over {DRAWS} draws")
        say(f"      per routed clean planet without a free-haze run, expected label flips = {dis:.3f} (1 - agreement of step 1{', summed over CH4 and H2O' if target == 'joint' else ''}), "
            f"{dis_ub:.3f} at the 95 % upper bound of the disagreement rate")
        say(f"      {'ordering':<34} {'routed':>6} {'hazed':>6} {'clean':>6} {'w/ FH':>6} {'no FH':>6} | {'lost (oracle-assisted)':>22} {'lost (FH substituted)':>21} {'max |change|':>12} | {'known flips':>11} {'exp. flips no-FH':>16} {'at 95% UB':>9}  bound on lost")
        for _, r in g.iterrows():
            exp_flips, ub_flips = r.rcm * dis, r.rcm * dis_ub
            say(f"      {NICE[r.rule]:<34} {r.routed:6.1f} {r.rh:6.1f} {r.rc:6.1f} {r.rcf:6.1f} {r.rcm:6.1f} | {r.lo:12.1f} ± {r.lo_sd:4.1f}     {r.ls:11.1f} ± {r.ls_sd:4.1f}     {r.dabs:8.0f}     | {r.flips:11.2f} {exp_flips:16.1f} {ub_flips:9.1f}  "
                f"{r.ls:.1f} ± {exp_flips:.1f} (± {ub_flips:.1f} at the 95 % UB; worst ± {r.rcm:.0f})")
        say()
    say("   Reading: 'lost (FH substituted)' re-scores the routed clean planets that have a free-haze run by that run (the rest keep the clean-model")
    say("   retrieval, as in Table 12). 'expected flips, no-FH' = routed clean planets without a free-haze run x (1 - agreement); each flip moves at")
    say("   most one target in or out of the selection, so the operational number lies within 'lost (FH substituted)' ± that many targets in")
    say("   expectation (± the 'at 95% UB' column at the one-sided 95 % upper bound of the disagreement rate that 27 planets allow), and within")
    say("   ± the number of routed clean planets without a run in the worst case.")
    say()

    # ---- 3. what would make the f = 20 % column fully operational ----------------------------------------
    med = C.minutes.median(); med_h = pd.read_csv(os.path.join(RESULTS, f"identifiability_calibration_{H}_freehaze.csv")).minutes.median()
    say(f"3. Free-haze retrievals of clean spectra needed for the f = {F_SHOW:.0%} column at q = {Q_SHOW} to be fully operational (every routed clean planet")
    say(f"   scored by the free-haze retrieval of its clean spectrum), union over the {DRAWS} draws and the three targets; cost at the median of the")
    say(f"   27 existing runs ({med:.1f} min each on one core; the hazed free-haze runs took {med_h:.1f} min median)")
    say(f"   {'ordering':<34} {'union of routed clean':>21} {'already run':>11} {'missing':>8} {'core-hours':>10}")
    for r in RULES:
        u = union[r]; miss = len(u - set(names[has_fh]))
        say(f"   {NICE[r]:<34} {len(u):21d} {len(u) - miss:11d} {miss:8d} {miss * med / 60:10.0f}")
    miss_all = len(union_all - set(names[has_fh]))
    say(f"   {'all orderings (union)':<34} {len(union_all):21d} {len(union_all) - miss_all:11d} {miss_all:8d} {miss_all * med / 60:10.0f}")
    nclean = N - int(has_fh.sum())
    say(f"   {'every planet (a pipeline that cannot know h)':<34} {N:21d} {int(has_fh.sum()):11d} {nclean:8d} {nclean * med / 60:10.0f}")
    say(f"   per draw, the routed clean planets of all orderings together number {np.mean(per_draw_union):.0f} on average (min {min(per_draw_union)}, max {max(per_draw_union)}),")
    say(f"   so a single draw would need about {np.mean(per_draw_union):.0f} runs ({np.mean(per_draw_union) * med / 60:.0f} core-hours); the union over draws is")
    say(f"   large because the random, margin and learner-spread orderings route different clean planets in every draw.")
    say(f"   For comparison, the manuscript's 520 core-hours for the free-haze retrieval of every planet, and 127 for the budget.")
    open(os.path.join(RESULTS, "fallback_model_choice.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
