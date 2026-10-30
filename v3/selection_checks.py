"""Four checks the review of 2026-10-30 asked of the prevalence and selection results.

1. How much of the undecidable fraction is labels next to the 10^-4 threshold (within half a decade)?
2. Do the undecided planets fit in the Tier-2 slots, if they are sent there rather than dismissed?
3. Selection on the classifier's own noisy realisation: the retrieval of the noisy spectrum against the retrieval of the noise-free one and the classifier, on the 200-planet subsample.
4. The mixed-population budget at the hazed fractions 10 and 50 %, beside 20 %.
Writes results/selection_checks.txt.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402

MOLS = ["CH4", "H2O", "CO2", "NH3"]; L = []
C = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv")).drop_duplicates("name"); C = C[C["error"].isna()] if "error" in C else C

L += [f"1. Labels within half a decade of 10^-4 ({len(C)} planets, p_true < 0.95 undecidable)", f"{'':<6}{'share of labels':>18}{'undecidable among them':>26}{'all labels':>12}{'excluding them':>16}{'absent, excl.':>15}{'present, excl.':>16}"]
und_any = np.zeros(len(C), bool)
for m in MOLS:
    t = C[f"{m}_true"]; p = C[f"{m}_p_true"]; near = (t + 4).abs() < 0.5; und = p < 0.95; lab = C[f"{m}_label"] == 1; und_any |= und.to_numpy()
    L.append(f"{m:<6}{100 * near.mean():18.1f}{100 * und[near].mean():26.1f}{100 * und.mean():12.1f}{100 * und[~near].mean():16.1f}{100 * und[~near & ~lab].mean():15.1f}{100 * und[~near & lab].mean():16.1f}")
L.append(f"planets with at least one undecidable label of the four: {100 * und_any.mean():.1f} %; with at least one of CH4 and H2O: {100 * ((C.CH4_p_true < .95) | (C.H2O_p_true < .95)).mean():.1f} %")

n = len(C); slots = int(round(0.6 * n)); L += ["", f"2. Do the undecided fit in {slots} slots ({n} planets)?", f"{'target':<8}{'decided present':>17}{'undecided':>11}{'decided absent':>16}{'present + undecided':>21}{'fits':>6}{'targets lost, ranked by posterior':>35}"]
for tag, ms in (("CH4", ["CH4"]), ("H2O", ["H2O"]), ("CO2", ["CO2"]), ("NH3", ["NH3"]), ("either", ["CH4", "H2O"])):
    pres = np.max([C[f"{m}_p_present"] for m in ms], axis=0); dp = pres > 0.95; da = np.all([C[f"{m}_p_present"] < 0.05 for m in ms], axis=0); un = ~dp & ~da; y = np.clip(sum((C[f"{m}_label"] == 1).astype(int) for m in ms), 0, 1).to_numpy()
    sc = sum(C[f"{m}_p_present"] for m in ms).to_numpy(); top = np.argsort(-sc)[:slots]
    L.append(f"{tag:<8}{dp.sum():17d}{un.sum():11d}{da.sum():16d}{(dp | un).sum():21d}{'yes' if (dp | un).sum() <= slots else 'no':>6}{int(y.sum() - y[top].sum()):35d}")

cases, Pte, e, sig = D.test_cases(); sc = D.Scorer(e); P = sc.probs(cases["clean"]); nm = {a: i for i, a in enumerate(Pte.name)}
N = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration_noisy.csv")).drop_duplicates("name"); N = N[N.name.isin(nm)]; idx = N.name.map(nm).to_numpy(); F = C.set_index("name").loc[N.name]
n2 = len(N); s2 = int(round(0.6 * n2)); L += ["", f"3. Selection at 60 % on the {n2}-planet subsample ({s2} slots), classifier on its noisy realisation; targets lost"]
for m in ("CH4", "H2O"):
    y = (N[f"{m}_label"] == 1).to_numpy().astype(int); cl = np.mean([P["all"][(m, k)][idx] for k in ("KNN", "MLP", "RFC")], axis=0); ln = lambda s: int(y.sum() - y[np.argsort(-np.asarray(s))[:s2]].sum())
    L.append(f"{m}: targets {y.sum()}; classifier {ln(cl)}; retrieval of the noisy realisation {ln(N[f'{m}_p_present'])}; retrieval of the noise-free spectrum {ln(F[f'{m}_p_present'])}")

E = pd.read_csv(os.path.join(RESULTS, "e3_budget_freehaze.csv")); L += ["", "4. Mixed population by hazed fraction, 20 % budget, targets lost; columns of e3_budget_freehaze.csv: " + ", ".join(E.columns[:12])]
txt = "\n".join(L); open(os.path.join(RESULTS, "selection_checks.txt"), "w").write(txt + "\n"); print(txt)
