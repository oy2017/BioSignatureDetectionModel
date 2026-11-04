"""AD6 (REVISION_EXPECTATIONS.md, section AD): at Ariel coverage, the re-implemented classifier's majority vote against the
mean of its four learners, on the cases and noise draw of Table 16 (pipeline_sim.py, row "published, all inputs";
round3_ariel.py for the matched ensemble).

The per-planet predictions of the four learners are not stored, so they are regenerated from the cached learners
(data/alfnoor_jwst/learners_all.joblib, the same fits pipeline_sim.py uses through alfnoor_diagnostic.Scorer) on the
test cases of alfnoor_diagnostic.test_cases (fixed noise draw, SEED + 405) and the mute3 case built as pipeline_sim.py
builds it. A learner calls present at p >= 0.5 (as pipeline_sim.py). Vote rule as panel_tier1.py and round3_learner.py:
present if at least 3 of 4, absent if at most 1, a 2-2 split earns one half.

Writes results/round3/ariel_majority.txt.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(V3); sys.path.insert(0, V3)
from bin_spectra import bin_native  # noqa: E402
from common import DATA, RESULTS  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import alfnoor_jwst as AJ  # noqa: E402
import alfnoor_trust as AT  # noqa: E402

MOLS, LEARNERS = D.MOLS, D.LEARNERS
CASES = ["clean", "haze2e6", "haze3e7", "plhaze29", "cloud1e2", "spots20", "compound", "white_x2", "white_x3", "mute3"]
TABLE16 = ["clean", "haze3e7", "plhaze29", "cloud1e2", "spots20", "white_x2", "mute3"]


def main():
    cases, Pte, e, sig = D.test_cases(extra=("haze2e6", "plhaze29")); cases = dict(cases)
    wl = np.load(os.path.join(DATA, "native_wl.npy")); X0 = np.load(os.path.join(AF.DIRS["faithful"], "pop1_native.npy")).astype(float); nf = bin_native(X0[np.isfinite(X0).all(1)], wl, e)
    eps = cases["clean"] - nf; mu = nf.mean(1, keepdims=True); cases["mute3"] = mu + (nf - mu) / 3 + eps
    L = AJ.learners("all"); rows = []
    for c in CASES:
        Z = AJ.znorm(cases[c], L["cols"])
        for m in MOLS:
            y = (Pte[f"atm {m}"] > -4).to_numpy(); calls = np.array([AT.prob(L["models"][("1e-4", m, k)], Z) >= 0.5 for k in LEARNERS]); v = calls.sum(0)
            pr, ab = v >= 3, v <= 1; sp = ~(pr | ab); right = np.where(y, pr, ab); credit = right + 0.5 * sp
            single = (calls == y[None, :]).mean(1)
            rows.append(dict(case=c, molecule=m, mean_single=100 * single.mean(), **{f"acc_{k}": 100 * a for k, a in zip(LEARNERS, single)}, vote_half=100 * credit.mean(), vote_decided=100 * right[~sp].mean(), split=100 * sp.mean(), vote_ties_absent=100 * (pr == y).mean(), vote_ties_present=100 * ((v >= 2) == y).mean()))
        print(c, flush=True)
    A = pd.DataFrame(rows); g = A.groupby("case").mean(numeric_only=True).loc[CASES]
    ens = pd.read_csv(os.path.join(RESULTS, "round3", "round3_ariel.csv")).groupby("case").accuracy.mean().reindex(CASES)
    ref = pd.read_csv(os.path.join(RESULTS, "pipeline_sim.csv")); ref = ref[ref.pipeline == "published, all inputs"].groupby("case").accuracy.mean().reindex(CASES)
    out = ["AD6: re-implemented classifier at Ariel coverage (104 bins, Tier-1 noise, POP-I, %d planets), the noise draw of Table 16." % len(Pte),
           "Accuracy %, mean over four molecules. Majority: present if >= 3 of 4 learners, absent if <= 1, a 2-2 split one half (panel_tier1.py).", ""]
    T = pd.DataFrame({"mean single learner": g.mean_single, "  (pipeline_sim.csv, check)": ref, "majority, split = 1/2": g.vote_half, "majority, decided only": g.vote_decided, "split share": g.split,
                      "majority, 2-2 -> absent": g.vote_ties_absent, "majority, 2-2 -> present": g.vote_ties_present, "best single learner": g[[f"acc_{k}" for k in LEARNERS]].max(1),
                      "matched ensemble": ens}).T
    T["Table 16 mean"] = T[TABLE16].mean(1); out.append(T.round(1).to_string())
    d = (g.vote_half - g.mean_single); de = ens - g.vote_half
    out += ["", "majority (split 1/2) minus mean single learner: " + ", ".join(f"{c} {d[c]:+.2f}" for c in CASES),
            f"  largest |difference| over Table 16's cases: {d[TABLE16].abs().max():.1f} points; within 2 points in {(d[TABLE16].abs() <= 2).sum()} of {len(TABLE16)}",
            "matched ensemble minus majority (split 1/2): " + ", ".join(f"{c} {de[c]:+.1f}" for c in CASES),
            "matched ensemble minus mean single learner: " + ", ".join(f"{c} {(ens - g.mean_single)[c]:+.1f}" for c in CASES), "",
            "Per learner, mean over molecules:", g[[f"acc_{k}" for k in LEARNERS]].round(1).T.to_string(), "",
            "Per molecule, Table 16 cases (mean single / majority split 1/2 / split share):"]
    for m in MOLS:
        q = A[A.molecule == m].set_index("case").loc[TABLE16]; out.append(f"  {m}: " + ", ".join(f"{c} {q.mean_single[c]:.1f}/{q.vote_half[c]:.1f}/{q.split[c]:.0f}" for c in TABLE16))
    t = "\n".join(out); open(os.path.join(RESULTS, "round3", "ariel_majority.txt"), "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
