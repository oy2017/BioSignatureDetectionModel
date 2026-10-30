"""The per-reduction design for Ariel: complete coverage, one set of learners, Tier-1 noise; scored on POP-I as pipeline_sim.py (REVISION_EXPECTATIONS.md, section I). Writes results/matched_learner/matched_ariel.txt."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import matched_learner as ML  # noqa: E402

CASES = ["clean", "haze2e6", "haze3e7", "cloud1e2", "spots20", "compound", "white_x2", "white_x3", "mute3"]


def feats(X, sig):
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    return np.column_stack([Z, np.log10(X.std(1) / np.sqrt((sig ** 2).mean(1)))])


def main():
    cases, Pte, e, sig = D.test_cases(extra=("haze2e6",)); Xn = cases["clean"]; Xtr, Ptr = ML.training(); rng = np.random.default_rng(SEED + 811); s_tr = AF.sigma(Ptr, e, "radiometric")
    f = np.where(rng.uniform(size=len(Xtr)) < 0.5, 10 ** rng.uniform(-1, 0, len(Xtr)), 1.0)[:, None]; mu = Xtr.mean(1, keepdims=True); T = mu + (Xtr - mu) * f + rng.normal(0, 1, Xtr.shape) * s_tr; Ztr = feats(T, s_tr)
    # the mute3 case as pipeline_sim.py builds it, from the noise-free clean spectra and the same noise
    wl = np.load(os.path.join(ML.DATA, "native_wl.npy")); src = AF.DIRS["faithful"]; X0 = np.load(os.path.join(src, "pop1_native.npy")).astype(float); nf = ML.bin_native(X0[np.isfinite(X0).all(1)], wl, e); eps = Xn - nf; m_ = nf.mean(1, keepdims=True); cases["mute3"] = m_ + (nf - m_) / 3 + eps
    rows = []
    for m in ML.MOLS:
        y = (Ptr[f"atm {m}"] > -4).astype(int).to_numpy(); t = (Pte[f"atm {m}"] > -4).to_numpy()
        for k, mk in ML.MAKERS().items():
            L = mk().fit(Ztr, y)
            for c in CASES:
                p = L.predict(feats(cases[c], sig)).astype(bool); rows.append(dict(case=c, molecule=m, learner=k, accuracy=100 * (p == t).mean(), recall=100 * p[t].mean(), spec=100 * (~p[~t]).mean()))
            print(m, k, flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RESULTS, "matched_learner", "matched_ariel.csv"), index=False); g = R.groupby("case")[["accuracy", "recall", "spec"]].mean().loc[CASES].round(1)
    t = "Design for Ariel on POP-I, mean over four learners and four molecules\n" + g.T.to_string(); open(os.path.join(RESULTS, "matched_learner", "matched_ariel.txt"), "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
