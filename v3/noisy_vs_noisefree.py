"""Noise-free versus noisy-realisation retrievals on the same random 200-planet POP-I subsample.

Every retrieval in the identifiability study is run on the noise-free spectrum with the Tier-1 sigma in the
likelihood (an expected-posterior definition of decidability), while the classifier is scored on one noisy
realisation. identifiability_calibrate.py --noisy --random-sample retrieves the classifier's own realisation of
the same planets. This script compares the two on the subsample: how the posterior of the true class moves, how
many decidable labels flip, the retrieval's own accuracy on the realisation ('noisy ceiling'), and what share
of the classifier's errors a retrieval of the realisation actually recovers, so that the recoverable shares
in the paper (defined on the noise-free posterior) can be read as the upper bounds they are.

Usage: python noisy_vs_noisefree.py [--departure haze3e7] [--free-haze]
Writes results/noisy_vs_noisefree[_<departure>[_freehaze]].txt
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
MOLS = D.MOLS


def load(fn):
    p = os.path.join(RESULTS, fn)
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    C = pd.read_csv(p).drop_duplicates("name")
    return C[C["error"].isna()] if "error" in C else C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--departure", default="none"); ap.add_argument("--free-haze", action="store_true")
    a = ap.parse_args()
    tag = ("" if a.departure == "none" else f"_{a.departure}") + ("_freehaze" if a.free_haze else "")
    A = load(f"identifiability_calibration{tag}.csv"); B = load(f"identifiability_calibration{tag}_noisy.csv")
    names = sorted(set(A.name) & set(B.name)); A = A.set_index("name").loc[names]; B = B.set_index("name").loc[names]
    cases, Pte, e, sig = D.test_cases(); sc = D.Scorer(e)
    case = "clean" if a.departure == "none" else a.departure
    S, P, _ = sc.scores(cases[case]); idx = {n: i for i, n in enumerate(Pte.name)}; i = np.array([idx[n] for n in names])
    L = [f"Noise-free vs noisy-realisation retrieval, {len(names)} planets, spectrum case '{case}'" + (" (free-haze model)" if a.free_haze else ""),
         "  decidable = p_true > 0.95; the noisy run retrieves the classifier's own realisation of each spectrum", ""]
    rows = []
    for m in MOLS:
        y = (Pte[f"atm {m}"].to_numpy()[i] > -4).astype(int)
        ptA = A[f"{m}_p_true"].to_numpy(float); ptB = B[f"{m}_p_true"].to_numpy(float)
        ppA = A[f"{m}_p_present"].to_numpy(float); ppB = B[f"{m}_p_present"].to_numpy(float)
        decA, decB = ptA > 0.95, ptB > 0.95
        rightA, rightB = (ppA >= 0.5) == (y == 1), (ppB >= 0.5) == (y == 1)
        misA, misB = ptA < 0.05, ptB < 0.05
        # the classifier's errors on the same realisation: per learner on the all-bins variant, pooled over the four
        wrong = np.concatenate([((P["all"][(m, k)][i] >= 0.5) != (y == 1)) for k in ("KNN", "MLP", "RFC", "SVC")])
        decA4, decB4, rightA4, rightB4 = (np.tile(v, 4) for v in (decA, decB, rightA, rightB))
        r = dict(molecule=m, n=len(y), present=int(y.sum()),
                 dec_noisefree=decA.mean(), dec_noisy=decB.mean(), dec_both=(decA & decB).mean(), dec_lost=(decA & ~decB).mean(), dec_gained=(~decA & decB).mean(),
                 acc_noisefree=rightA.mean(), acc_noisy=rightB.mean(), misled_noisefree=misA.mean(), misled_noisy=misB.mean(),
                 conf_wrong_noisefree=int(misA.sum()), conf_wrong_noisy=int(misB.sum()),
                 n_wrong=int(wrong.sum()), rec_noisefree=decA4[wrong].mean(), rec_noisy=decB4[wrong].mean(),
                 rec_noisefree_right=rightA4[wrong].mean(), rec_noisy_right=rightB4[wrong].mean(),
                 corr_pt=np.corrcoef(ptA, ptB)[0, 1], med_abs_dpt=np.median(np.abs(ptA - ptB)))
        for cls_name, mask in (("present", y == 1), ("absent", y == 0)):
            r[f"dec_noisefree_{cls_name}"] = decA[mask].mean(); r[f"dec_noisy_{cls_name}"] = decB[mask].mean()
        rows.append(r)
    T = pd.DataFrame(rows)
    L.append("== decidable fraction (p_true > 0.95): noise-free / noisy / both / lost under noise / gained under noise; by true class")
    for r in T.itertuples():
        L.append(f"  {r.molecule:<4} n={r.n:3d}  {100*r.dec_noisefree:5.1f} / {100*r.dec_noisy:5.1f} / {100*r.dec_both:5.1f} / {100*r.dec_lost:5.1f} / {100*r.dec_gained:5.1f} %"
                 f"   present {100*r.dec_noisefree_present:5.1f} -> {100*r.dec_noisy_present:5.1f} %  absent {100*r.dec_noisefree_absent:5.1f} -> {100*r.dec_noisy_absent:5.1f} %"
                 f"   corr(p_true) {r.corr_pt:.2f}  median |dp_true| {r.med_abs_dpt:.3f}")
    L.append(""); L.append("== retrieval accuracy (p_present >= 0.5) and confidently wrong labels (p_true < 0.05): noise-free -> noisy")
    for r in T.itertuples():
        L.append(f"  {r.molecule:<4} acc {100*r.acc_noisefree:5.1f} -> {100*r.acc_noisy:5.1f} %   confidently wrong {r.conf_wrong_noisefree:3d} -> {r.conf_wrong_noisy:3d} of {r.n}")
    L.append(""); L.append("== the classifier's errors on the same realisation (four learners pooled, all bins): share 'recoverable' by the noise-free posterior")
    L.append("   (p_true > 0.95), share decidable by the noisy posterior, and share each retrieval gets right at p_present >= 0.5")
    for r in T.itertuples():
        L.append(f"  {r.molecule:<4} wrong {r.n_wrong:4d}   recoverable (noise-free) {100*r.rec_noisefree:5.1f} %   decidable (noisy) {100*r.rec_noisy:5.1f} %   retrieval right: noise-free {100*r.rec_noisefree_right:5.1f} %  noisy {100*r.rec_noisy_right:5.1f} %")
    w = T.n_wrong
    L.append(f"  all  wrong {int(w.sum()):4d}   recoverable (noise-free) {100*np.average(T.rec_noisefree, weights=w):5.1f} %   decidable (noisy) {100*np.average(T.rec_noisy, weights=w):5.1f} %   retrieval right: noise-free {100*np.average(T.rec_noisefree_right, weights=w):5.1f} %  noisy {100*np.average(T.rec_noisy_right, weights=w):5.1f} %")
    out = os.path.join(RESULTS, f"noisy_vs_noisefree{tag}.txt"); open(out, "w").write("\n".join(L) + "\n"); T.to_csv(out.replace(".txt", ".csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
