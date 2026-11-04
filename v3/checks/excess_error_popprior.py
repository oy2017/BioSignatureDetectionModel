"""AE1: the re-implemented classifier's excess error over the noisy retrieval posterior read at p >= 0.5, with the retrieval run under POP-I's own abundance prior (10^-7 to 10^-2) on the same 200 noisy realisations, beside the POP-III-prior result of checks/excess_error.py. Writes results/excess_error_popprior.txt."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE); sys.path.insert(0, HERE)
import alfnoor_diagnostic as D
MOLS = ["CH4", "H2O", "CO2", "NH3"]
load = lambda f: pd.read_csv(f"results/{f}").drop_duplicates("name").pipe(lambda c: c[c["error"].isna()] if "error" in c else c)
A, B = load("identifiability_calibration_noisy.csv"), load("identifiability_calibration_noisy_popprior.csv")
names = sorted(set(A.name) & set(B.name)); A, B = A.set_index("name").loc[names], B.set_index("name").loc[names]
cases, Pte, e, sig = D.test_cases(); sc = D.Scorer(e); S, P, _ = sc.scores(cases["clean"]); idx = {n: i for i, n in enumerate(Pte.name)}; i = np.array([idx[n] for n in names])
L = [f"Classifier (four learners pooled, all bins) against the noisy retrieval posterior at p >= 0.5 on the same {len(names)} noisy realisations, under POP-III's prior (10^-9 to 10^-2) and under POP-I's own prior (10^-7 to 10^-2)", ""]
for m in MOLS:
    y = (Pte[f"atm {m}"].to_numpy()[i] > -4).astype(int); wrong = np.concatenate([((P["all"][(m, k)][i] >= 0.5) != (y == 1)) for k in ("KNN", "MLP", "RFC", "SVC")]); e_cls = wrong.mean()
    out = [f"{m:<4} classifier error {100*e_cls:5.1f} %"]
    for tag, C in (("POP-III prior", A), ("POP-I prior  ", B)):
        right = (C[f"{m}_p_present"].to_numpy(float) >= 0.5) == (y == 1); e_ref = 1 - right.mean(); ex = e_cls - e_ref; fix = np.tile(right, 4)[wrong].mean()
        out.append(f"{tag}: reference error {100*e_ref:5.1f} %  excess {100*ex:5.1f} points ({100*ex/e_cls:4.1f} % of the classifier's error)  reference right on {100*fix:4.1f} % of its errors")
    L.append("  |  ".join(out))
t = "\n".join(L); open("results/excess_error_popprior.txt", "w").write(t + "\n"); print(t)
