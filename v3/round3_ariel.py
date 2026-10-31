"""The frozen third-round design at Ariel coverage (REVISION_EXPECTATIONS.md, section O): complete coverage, Tier-1 noise, scored on POP-I as pipeline_sim.py. Writes results/round3/round3_ariel.txt."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from bin_spectra import bin_native  # noqa: E402
from common import DATA, RESULTS, SEED  # noqa: E402
import alfnoor_diagnostic as D  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_ariel as MA  # noqa: E402
import round3_learner as R  # noqa: E402

MOLS = R.MOLS


def population(e):
    wl = np.load(os.path.join(DATA, "native_wl.npy")); Xs, Ps = [], []
    for f in ("correlated", "broad"):
        X = np.load(os.path.join(DATA, f, f"{f}_native.npy")).astype(float); P = pd.read_parquet(os.path.join(DATA, f, f"{f}_params.parquet")); ok = np.isfinite(X).all(1); Xs.append(bin_native(X[ok], wl, e)); Ps.append(P[ok])
    return np.vstack(Xs), pd.concat(Ps, ignore_index=True)


def draw(X, s, rng):
    f = np.where(rng.uniform(size=len(X)) < 0.5, 10 ** rng.uniform(-1, 0, len(X)), 1.0)[:, None]; mu = X.mean(1, keepdims=True); return MA.feats(mu + (X - mu) * f + rng.normal(0, 1, X.shape) * s, s)


def main():
    cases, Pte, e, sig = D.test_cases(extra=("haze2e6", "plhaze29")); X, P = population(e); s = AF.sigma(P, e, "radiometric"); Y = (P[[f"atm {m}" for m in MOLS]].to_numpy() > -4).astype(int); hold = np.random.default_rng(SEED + 77).uniform(size=len(X)) < 0.15; nets = []
    for k in range(R.MEMBERS):
        rng = np.random.default_rng(SEED + 8100 + k); Z = np.vstack([draw(X[~hold], s[~hold], rng) for _ in range(R.COPIES)]); nets.append(MLPClassifier((256, 128), batch_size=256, early_stopping=True, n_iter_no_change=6, max_iter=60, random_state=SEED + 8100 + k).fit(Z, np.tile(Y[~hold], (R.COPIES, 1)))); print(k, flush=True)
    prob = lambda Z: np.mean([n.predict_proba(Z) for n in nets], axis=0); t = R.thresholds(prob(draw(X[hold], s[hold], np.random.default_rng(SEED + 8199))), Y[hold])
    wl = np.load(os.path.join(DATA, "native_wl.npy")); X0 = np.load(os.path.join(AF.DIRS["faithful"], "pop1_native.npy")).astype(float); nf = bin_native(X0[np.isfinite(X0).all(1)], wl, e); Xn = cases["clean"]; eps = Xn - nf; mu = nf.mean(1, keepdims=True); cases = dict(cases); cases["mute3"] = mu + (nf - mu) / 3 + eps; rows = []
    slots = int(round(0.6 * len(Pte))); rows2 = []; Rc = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv")).drop_duplicates("name").set_index("name").reindex(Pte.name.to_numpy())
    for c in MA.CASES + ["plhaze29"]:
        p = prob(MA.feats(cases[c], sig))
        for i, m in enumerate(MOLS):
            q = p[:, i] >= t[i]; y = (Pte[f"atm {m}"] > -4).to_numpy(); pt = Rc[f"{m}_p_true"].to_numpy(); e = q != y; rows2.append(dict(case=c, molecule=m, recoverable=int((e & (pt > 0.95)).sum()), undecidable=int((e & (pt <= 0.95)).sum()))); top = np.argsort(-p[:, i])[:slots]; rows.append(dict(case=c, molecule=m, accuracy=100 * (q == y).mean(), recall=100 * q[y].mean(), spec=100 * (~q[~y]).mean(), lost=int(y.sum() - y[top].sum()), targets=int(y.sum())))
    A = pd.DataFrame(rows); A.to_csv(os.path.join(R.OUT, "round3_ariel.csv"), index=False); g = A.groupby("case")[["accuracy", "recall", "spec"]].mean().loc[MA.CASES + ["plhaze29"]].round(1); T = "C+B-mlp for Ariel on POP-I, mean over four molecules; thresholds " + str(dict(zip(MOLS, t.round(2)))) + "\n" + g.T.to_string() + "\n\nTargets lost at the 60 % selection\n" + A[A.molecule.isin(["CH4", "H2O"])].pivot_table(index="molecule", columns="case", values="lost")[MA.CASES + ["plhaze29"]].to_string(); T += "\n\nErrors by class on the clean-spectrum posterior, summed over molecules\n" + pd.DataFrame(rows2).groupby("case")[["recoverable", "undecidable"]].sum().loc[MA.CASES + ["plhaze29"]].T.to_string(); open(os.path.join(R.OUT, "round3_ariel.txt"), "w").write(T + "\n"); print(T)


if __name__ == "__main__":
    main()
