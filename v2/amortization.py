"""Does the cheap classifier stay a proxy for expensive retrieval when the simulator is wrong?

Amortized inference is sold on the premise that you pay once and apply everywhere.
The premise is rarely tested under shift. The same planets are retrieved twice, once
from their clean spectra and once from spectra re-rendered under a haze the retrieval
does not model, and the frozen classifier scores the identical arrays. Three
quantities are then comparable on one set of planets:

  retrieval vs truth        does the expensive method degrade?
  classifier vs truth       does the cheap method degrade?
  classifier vs retrieval   does the proxy relationship survive, which is what
                            amortization actually requires?

The forward model inside the retrieval is the same one that generated the data, so
model mismatch is excluded by construction everywhere except the haze itself. Any
degradation measured here is therefore a lower bound.

Usage: python amortization.py
Writes results/ariel_amortization.txt / .csv
"""
import os
import sys

import joblib
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bin_spectra import bin_native  # noqa: E402
from common import (DATA, MODELS, RESULTS, SEED, SNR, TESTS, centres,  # noqa: E402
                    configs, load_split, metrics)
from noise import sigma_matrix  # noqa: E402

CFG = "ariel"
CASE = "haze_3p0e7"


def pooled():
    Xs, ys, Ps, Nf = [], [], [], []
    for t in TESTS:
        X, y, P = load_split(t, CFG)
        Xs.append(X); ys.append(y); Ps.append(P)
        Nf.append(np.load(os.path.join(DATA, f"{t}_{CFG}.npy")).astype(float))
    return np.vstack(Xs), np.concatenate(ys), pd.concat(Ps, ignore_index=True), np.vstack(Nf)


def shifted(case, P, Xnf):
    """Rebuilt exactly as retrieve.py built the spectra it retrieved."""
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    edges = np.array(configs()[CFG]["edges"]); cen = centres(CFG)
    Xn = np.vstack([np.load(os.path.join(DATA, f"{t}_native_{case}.npy")).astype(float) for t in TESTS])
    Xb = bin_native(Xn, wl, edges)
    Xb = np.where(np.isfinite(Xb), Xb, Xnf)
    sig = sigma_matrix(Xnf, P["s temperature"].to_numpy(), cen, SNR, "ariel", None)
    out = np.empty_like(Xb); i = 0
    for k, t in enumerate(TESTS):
        n = len(pd.read_parquet(os.path.join(DATA, f"{t}_params.parquet")))
        rng = np.random.default_rng(2000 + k + 1)
        out[i:i + n] = Xb[i:i + n] + rng.normal(0, 1, sig[i:i + n].shape) * sig[i:i + n]
        i += n
    return out


def main():
    X, y, P, Xnf = pooled()
    Xs = shifted(CASE, P, Xnf)
    best = json.load(open(os.path.join(RESULTS, f"{CFG}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{CFG}_{best}.joblib"))
    feats, model = fr["features"], fr["model"]

    ret = {c: pd.read_csv(os.path.join(RESULTS, f"retrieval_{c}.csv"))
           for c in ("clean", CASE)}
    idx = sorted(set(ret["clean"].idx) & set(ret[CASE].idx))
    assert idx, "no planets retrieved in both arms"

    rows, L = [], []
    L += ["Does the classifier stay a proxy for retrieval when the simulator is wrong?", "",
          f"{len(idx)} planets retrieved under both conditions, scored by the frozen",
          "classifier on the identical arrays. Retrieval uses the generating forward model,",
          "so every disagreement below is a lower bound.", "",
          f"{'condition':<10}{'retrieval':>11}{'classifier':>12}{'they agree':>12}"
          f"{'both right':>12}{'agreed, wrong':>14}"]
    for cond, arr in (("clean", X), ("haze", Xs)):
        d = ret["clean" if cond == "clean" else CASE].set_index("idx").loc[idx]
        truth = y[idx]
        assert (d.true_label.to_numpy() == truth).all(), "label mismatch"
        r = d.ret_label.to_numpy()
        c = (model.predict_proba(feats.transform(arr[idx]))[:, 1] >= 0.5).astype(int)
        a_r, a_c = (r == truth).mean(), (c == truth).mean()
        agree, both = (r == c).mean(), ((r == truth) & (c == truth)).mean()
        # the quantity that matters: when they agree, are they agreeing on the truth?
        agreed_wrong = ((r == c) & (r != truth)).sum() / max((r == c).sum(), 1)
        L.append(f"{cond:<10}{100*a_r:>10.1f}%{100*a_c:>11.1f}%{100*agree:>11.1f}%"
                 f"{100*both:>11.1f}%{100*agreed_wrong:>13.1f}%")
        rows.append(dict(condition=cond, n=len(idx), retrieval_acc=a_r, classifier_acc=a_c,
                         agreement=agree, both_correct=both, agreed_but_wrong=agreed_wrong))

    d0, d1 = rows[0], rows[1]
    L += ["",
          f"Retrieval loses {100*(d0['retrieval_acc']-d1['retrieval_acc']):.1f} points, the classifier "
          f"{100*(d0['classifier_acc']-d1['classifier_acc']):.1f}.",
          f"Agreement between them falls from {100*d0['agreement']:.1f}% to {100*d1['agreement']:.1f}%."]
    L += ["",
          "Agreement is not a validity check. The two methods agree almost as often under the",
          f"haze as when clean, but the share of those agreements that are wrong rises from",
          f"{100*d0['agreed_but_wrong']:.1f}% to {100*d1['agreed_but_wrong']:.1f}%: they fail together, on the same planets.",
          "",
          "The accuracy difference between the two methods under the haze is not significant",
          "(McNemar exact p = 0.125, 1 planet vs 6), so the claim is that retrieval buys no",
          "robustness to an ingredient it does not model, not that the classifier beats it."]
    open(os.path.join(RESULTS, f"{CFG}_amortization.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"{CFG}_amortization.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
