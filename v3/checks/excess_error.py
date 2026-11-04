"""AD1: the re-implemented classifier's error on the 200 noisy realisations against a reference rule, the noisy retrieval posterior read at p >= 0.5 (POP-III prior, the training forward model). Reads results/noisy_vs_noisefree.csv. Writes results/excess_error.txt."""
import os, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = pd.read_csv("results/noisy_vs_noisefree.csv"); L = ["Classifier (four learners pooled, all bins) against the noisy retrieval posterior at p >= 0.5, on the same 200 noisy realisations (POP-III prior; a reference rule, not a bound)", ""]
for _, r in T.iterrows():
    n = 4 * 200; e_cls = r.n_wrong / n; e_ref = 1 - r.acc_noisy; ex = e_cls - e_ref; fix = r.rec_noisy_right
    L.append(f"{r.molecule:<4} classifier error {100*e_cls:5.1f} %  reference error {100*e_ref:5.1f} %  excess {100*ex:5.1f} points ({100*ex/e_cls:4.1f} % of the classifier's error)  reference right on {100*fix:4.1f} % of the classifier's errors")
t = "\n".join(L); open("results/excess_error.txt", "w").write(t + "\n"); print(t)
