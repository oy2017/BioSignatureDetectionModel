"""AF3: class-conditional calibration of the dchi2 identifiability test.

The current test (identifiability_calibrate.summarize) sets one dchi2 threshold per molecule by a logistic fit of
1[p_true > 0.95] on log10 dchi2 over all 955 planets, pooling truly present and truly absent labels. Under the
retrieval's prior (POP-III, 1e-9..1e-2, boundary 1e-4) a present label needs a Bayes factor of 47 to reach 0.95 and an
absent label 8, so the pooled threshold blends two standards. Here the same logistic rule is fitted (i) separately for
present and absent labels, and (ii) against the Bayes-factor criterion BF_true > 19 relative to the prior odds (the
Table 4 "Bayes factor < 19" row), pooled and per class. The classifier's errors (four learners pooled, all-bins variant,
as alfnoor_diagnostic) are then re-classed on clean spectra and under each departure (departure-aware dchi2 files; white
noise as dchi2 / c^2), giving the counterparts of Table 7's rows.

Usage: OMP_NUM_THREADS=1 python checks/class_conditional_calibration.py   Writes results/class_conditional_calibration.txt
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(V3); sys.path.insert(0, V3)
import numpy as np, pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
import alfnoor_diagnostic as D
import alfnoor_jwst as AJ
import alfnoor_trust as AT

MOLS, LEARNERS = D.MOLS, D.LEARNERS
PI_PRESENT = 2.0 / 7.0                    # POP-III prior mass above 1e-4: 2 of 7 decades (0.29 present / 0.71 absent)
Q = 0.95; BF_CRIT = Q / (1 - Q)          # 19
DEPARTURES = [("haze3e7", "haze3e7", 1.0), ("spots20", "spots20", 1.0), ("cloud1e2", "cloud1e2", 1.0), ("white_x2", "none", 0.25), ("white_x3", "none", 1 / 9)]
L = []
say = lambda s="": (L.append(s), print(s, flush=True))


def bf_true(p_true, label):
    """Bayes factor of the true class against the opposite one, relative to the prior odds."""
    p = np.clip(p_true, 1e-12, 1 - 1e-12); pi = np.where(label == 1, PI_PRESENT, 1 - PI_PRESENT)
    return (p / (1 - p)) / (pi / (1 - pi))


def logistic_threshold(dchi2, ind):
    """dchi2 at which a logistic fit of the indicator on log10 dchi2 crosses 1/2 (identifiability_calibrate.summarize)."""
    x = np.log10(np.clip(dchi2, 1e-3, None)); ind = ind.astype(int)
    if ind.min() == ind.max():
        return np.nan
    lr = LogisticRegression(C=1e6).fit(x[:, None], ind)
    try:
        return brentq(lambda t: lr.predict_proba([[np.log10(t)]])[0, 1] - 0.5, 1e-3, 1e4)
    except ValueError:
        return np.nan


# ---- data: posteriors, clean dchi2, thresholds ------------------------------------------------------------------
cal = pd.read_csv("results/identifiability_calibration.csv").drop_duplicates("name")
idf = pd.read_csv("results/identifiability_none_tier1.csv"); idf = idf[np.isfinite(idf.dchi2)]
old = pd.read_csv("results/identifiability_calibration_thresholds.csv").set_index("molecule").dchi2_at_95
say("Class-conditional calibration of the dchi2 test (AF3). Posterior: identifiability_calibration.csv (955 planets, noise-free spectra, Tier-1 sigma);")
say(f"clean dchi2: identifiability_none_tier1.csv. Prior class probabilities under POP-III's prior: present {PI_PRESENT:.3f}, absent {1-PI_PRESENT:.3f};")
say(f"BF_true = odds(p_true) / prior odds of the true class; decidable by BF if BF_true > {BF_CRIT:.0f} (p_true > 0.95 for a class of prior probability 0.5).")
say()
say("1. Undecidable fractions from the posterior (Table 4 check) and thresholds (logistic fit of the indicator on log10 dchi2, crossing 1/2)")
say(f"  {'mol':<4} {'undec p<.95':>11} {'present':>8} {'absent':>7} {'BF<19':>6} | {'pooled p95':>10} {'(file)':>6} {'present p95':>11} {'absent p95':>10} | {'pooled BF19':>11} {'present BF19':>12} {'absent BF19':>11}")
THR = {}; A = {}
for m in MOLS:
    a = idf[idf.molecule == m][["index", "dchi2", "label"]].merge(cal[["index", f"{m}_p_true"]], on="index")
    p = a[f"{m}_p_true"].to_numpy(); y = a.label.to_numpy(); d = a.dchi2.to_numpy(); bf = bf_true(p, y)
    dec95 = p > Q; decbf = bf > BF_CRIT
    t = dict(pooled=logistic_threshold(d, dec95), present=logistic_threshold(d[y == 1], dec95[y == 1]), absent=logistic_threshold(d[y == 0], dec95[y == 0]),
             bf_pooled=logistic_threshold(d, decbf), bf_present=logistic_threshold(d[y == 1], decbf[y == 1]), bf_absent=logistic_threshold(d[y == 0], decbf[y == 0]))
    THR[m] = t; A[m] = a.assign(p_true=p, bf=bf)
    say(f"  {m:<4} {100*(~dec95).mean():11.1f} {100*(~dec95[y==1]).mean():8.1f} {100*(~dec95[y==0]).mean():7.1f} {100*(~decbf).mean():6.1f} | {t['pooled']:10.2f} {old[m]:6.2f} {t['present']:11.2f} {t['absent']:10.2f} | {t['bf_pooled']:11.2f} {t['bf_present']:12.2f} {t['bf_absent']:11.2f}")
say("  (Table 4: undecidable 32/54/44/32 %, present 38/57/58/46, absent 29/52/35/23, BF < 19 39/62/50/35; thresholds 0.45/0.98/0.90/0.56)")
say()

# ---- classifier predictions -------------------------------------------------------------------------------------
cases, Pte, edges, sig = D.test_cases(); Lrn = AJ.learners("all")
P = {c: {(m, k): AT.prob(Lrn["models"][("1e-4", m, k)], AJ.znorm(X, Lrn["cols"])) for m in MOLS for k in LEARNERS} for c, X in cases.items() if c != "compound"}
name_to_i = {n: i for i, n in enumerate(Pte.name)}
y_true = {m: (Pte[f"atm {m}"] > -4).astype(int).to_numpy() for m in MOLS}
VARIANTS = ["pooled", "class-cond", "BF pooled", "BF class-cond"]


def decid(m, d, y, variant):
    t = THR[m]
    if variant == "pooled": return d > t["pooled"]
    if variant == "class-cond": return d > np.where(y == 1, t["present"], t["absent"])
    if variant == "BF pooled": return d > t["bf_pooled"]
    return d > np.where(y == 1, t["bf_present"], t["bf_absent"])


def share(mask, sub):
    return 100 * mask[sub].mean() if sub.any() else np.nan


# ---- 2. clean spectra: recoverable share of the classifier's errors ----------------------------------------------
say("2. Clean spectra: recoverable share (%) of the classifier's errors (four learners pooled) by each test and by the posterior")
say("   test variants: pooled = one threshold per molecule (current); class-cond = separate present / absent thresholds; BF = calibrated on BF_true > 19")
say(f"  {'mol':<4} {'class':<8} {'errors':>6} {'pooled':>7} {'class-cond':>10} {'BF pooled':>9} {'BF cl-cond':>10} | {'post p>.95':>10} {'post BF>19':>10} | {'Table7 mean-over-learners test (post)':>36}")
agree = {}
for m in MOLS:
    a = A[m]; idx = a["index"].to_numpy()   # cal/idf share the parquet index; map to Pte rows by name
    rows = cal.set_index("index").loc[idx]; I = np.array([name_to_i[n] for n in rows.name]); y = y_true[m][I]; d = a.dchi2.to_numpy(); pt = a.p_true.to_numpy(); bf = a.bf.to_numpy()
    assert np.array_equal(y, a.label.to_numpy())
    W = np.concatenate([(P["clean"][(m, k)][I] >= 0.5).astype(int) != y for k in LEARNERS]); Y = np.tile(y, 4); Dd = np.tile(d, 4); PT = np.tile(pt, 4); BF = np.tile(bf, 4)
    dec = {v: decid(m, Dd, Y, v) for v in VARIANTS}; post = PT > Q; postbf = BF > BF_CRIT
    per_learner = [np.mean([share(decid(m, d, y, "pooled"), w) for w in [(P["clean"][(m, k)][I] >= 0.5).astype(int) != y for k in LEARNERS]]),
                   np.mean([share(pt > Q, w) for w in [(P["clean"][(m, k)][I] >= 0.5).astype(int) != y for k in LEARNERS]])]
    for cls, sub in (("all", W), ("present", W & (Y == 1)), ("absent", W & (Y == 0))):
        say(f"  {m:<4} {cls:<8} {int(sub.sum()):6d} " + " ".join(f"{share(dec[v], sub):{w}.1f}" for v, w in zip(VARIANTS, (7, 10, 9, 10))) +
            f" | {share(post, sub):10.1f} {share(postbf, sub):10.1f} | " + (f"{per_learner[0]:.0f} ({per_learner[1]:.0f})".rjust(36) if cls == "all" else ""))
        for v in VARIANTS:
            ref = postbf if v.startswith("BF") else post
            agree[(m, cls, v)] = (100 * (dec[v][sub] == ref[sub]).mean(), 100 * (dec[v][sub] & ~ref[sub]).mean(), 100 * (~dec[v][sub] & ref[sub]).mean())
say("  (Table 7 clean row: 50 (25), 34 (20), 31 (20), 36 (19) for CH4, H2O, CO2, NH3)")
say()
say("3. Agreement (%) of each test with its posterior criterion on the clean error subset, per class: agree / test decidable but posterior not / posterior decidable but test not")
say("   (the p>.95 posterior for the pooled and class-cond tests, the BF>19 posterior for the BF tests)")
say(f"  {'mol':<4} {'class':<8} " + " ".join(f"{v:>22}" for v in VARIANTS))
for m in MOLS:
    for cls in ("all", "present", "absent"):
        say(f"  {m:<4} {cls:<8} " + " ".join(f"{agree[(m,cls,v)][0]:6.1f} / {agree[(m,cls,v)][1]:4.1f} / {agree[(m,cls,v)][2]:4.1f}".rjust(22) for v in VARIANTS))
say()

# ---- 4. departures ----------------------------------------------------------------------------------------------
say("4. Departures: recoverable share (%) of the errors the departure introduces (not made on the clean spectrum), four learners pooled, by each test;")
say("   dchi2 from the departure-aware run (identifiability_<departure>_tier1.csv), white noise as clean dchi2 / c^2; 'all' = all errors under the departure")
say(f"  {'case':<9} {'mol':<4} {'undec% pooled/cl-cond':>21} {'introd':>6} {'(pres/abs)':>10} {'pooled':>7} {'class-cond':>10} {'BF pooled':>9} {'BF cl-cond':>10} | {'cl-cond present':>15} {'cl-cond absent':>14} | {'all: pooled':>11} {'cl-cond':>8} {'BF pooled':>9} {'Table7 introd/all':>17}")
T7 = {("haze3e7", "CH4"): (87, 75), ("haze3e7", "H2O"): (64, 45), ("haze3e7", "CO2"): (81, 70), ("haze3e7", "NH3"): (79, 67), ("spots20", "CH4"): (79, 66), ("spots20", "H2O"): (65, 47), ("spots20", "CO2"): (57, 43), ("spots20", "NH3"): (80, 62),
      ("cloud1e2", "CH4"): (82, 68), ("cloud1e2", "H2O"): (51, 36), ("cloud1e2", "CO2"): (52, 36), ("cloud1e2", "NH3"): (76, 61), ("white_x2", "CH4"): (61, 45), ("white_x2", "H2O"): (32, 20), ("white_x2", "CO2"): (23, 17), ("white_x2", "NH3"): (53, 36),
      ("white_x3", "CH4"): (56, 41), ("white_x3", "H2O"): (23, 13), ("white_x3", "CO2"): (10, 8), ("white_x3", "NH3"): (41, 28)}
summary = {}
for c, base, scale in DEPARTURES:
    I_ = pd.read_csv(f"results/identifiability_{base}_tier1.csv"); I_ = I_[np.isfinite(I_.dchi2) & I_.name.isin(name_to_i)].copy(); I_["dchi2"] *= scale
    for m in MOLS:
        Dm = I_[I_.molecule == m]; I = Dm.name.map(name_to_i).to_numpy(); d = Dm.dchi2.to_numpy(); y = y_true[m][I]
        W = np.concatenate([(P[c][(m, k)][I] >= 0.5).astype(int) != y for k in LEARNERS]); W0 = np.concatenate([(P["clean"][(m, k)][I] >= 0.5).astype(int) != y for k in LEARNERS])
        Y = np.tile(y, 4); Dd = np.tile(d, 4); intro = W & ~W0; dec = {v: decid(m, Dd, Y, v) for v in VARIANTS}
        und = (100 * (~decid(m, d, y, "pooled")).mean(), 100 * (~decid(m, d, y, "class-cond")).mean())
        summary[(c, m)] = (share(dec["pooled"], intro), share(dec["class-cond"], intro), share(dec["BF pooled"], intro), share(dec["BF class-cond"], intro))
        say(f"  {c:<9} {m:<4} {und[0]:10.1f}/{und[1]:10.1f} {int(intro.sum()):6d} {int((intro & (Y==1)).sum()):5d}/{int((intro & (Y==0)).sum()):<4d} " + " ".join(f"{share(dec[v], intro):{w}.1f}" for v, w in zip(VARIANTS, (7, 10, 9, 10))) +
            f" | {share(dec['class-cond'], intro & (Y == 1)):15.1f} {share(dec['class-cond'], intro & (Y == 0)):14.1f} | {share(dec['pooled'], W):11.1f} {share(dec['class-cond'], W):8.1f} {share(dec['BF pooled'], W):9.1f} {T7[(c, m)][0]:8d} / {T7[(c, m)][1]:<5d}")
say()
say("5. Summary: introduced errors recoverable (%) under the class-conditional calibration [BF-calibrated pooled in brackets]")
say(f"  {'case':<9}" + "".join(f"{m:>16}" for m in MOLS))
for c, _, _ in DEPARTURES:
    say(f"  {c:<9}" + "".join(f"{summary[(c, m)][1]:6.0f} [{summary[(c, m)][2]:3.0f}]    " for m in MOLS))
maj = lambda c: all(summary[(c, m)][1] > 50 for m in MOLS); maj_not = lambda c: all(summary[(c, m)][1] < 50 for m in MOLS)
say(f"  class-conditional: haze majority recoverable on all gases: {maj('haze3e7')}; spots: {maj('spots20')}; noise x2 majority irrecoverable on all gases: {maj_not('white_x2')}; noise x3: {maj_not('white_x3')}")
open("results/class_conditional_calibration.txt", "w").write("\n".join(L) + "\n")
