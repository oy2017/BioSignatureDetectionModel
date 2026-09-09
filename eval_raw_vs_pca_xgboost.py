"""Does the PCA rotation cost XGBoost accuracy?

Section 4.1 notes that tree models occupy a privileged basis and that random
rotations reverse the tree-vs-network ordering (21), then asks only whether the
tree advantage survives a PCA rotation. The more direct question was never put:
XGBoost is the recommended model, and it has only ever been evaluated on the 102
principal components. This script adds the missing baseline -- the same tuned
XGBoost on the raw 550 wavelength bins -- with everything else held fixed.

Three configurations, identical hyperparameters, identical five test sets:

  raw            550 bins, untouched
  raw + z-score  550 bins, per-channel standardization (the first pipeline step)
  PCA-102        the published pipeline: z-score -> PCA(102) -> z-score

Trees split on axes, so the z-scored raw run is included to separate "PCA helps"
from "per-channel scaling helps"; histogram splits are near-invariant to
monotonic per-feature rescaling, so raw and raw+z-score should agree closely.
"""
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
FLOAT_COL = re.compile(r"^-?\d+\.\d+$")
XGB = dict(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
           eval_metric="logloss", random_state=SEED, n_jobs=-1)

label = lambda df: df["biosignature"].apply(lambda x: 1 if x == "yes" else 0).values


def load():
    tr = pd.read_parquet("multirex_spectra_H2_train.parquet")
    cols = [c for c in tr.columns
            if isinstance(c, float) or (isinstance(c, str) and FLOAT_COL.match(c))]
    tests = [pd.read_parquet(f"multirex_spectra_H2_test_set_{i}.parquet") for i in range(1, 6)]
    return tr, tests, cols


def metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (p == y).mean(), f1


def run(name, Xtr, ytr, tests_X, tests_y):
    Xtr, ytr = shuffle(Xtr, ytr, random_state=SEED)
    clf = XGBClassifier(**XGB).fit(Xtr, ytr)
    accs, f1s = [], []
    pooled_p, pooled_y = [], []
    for X, y in zip(tests_X, tests_y):
        p = clf.predict(X)
        a, f = metrics(y, p)
        accs.append(a); f1s.append(f)
        pooled_p.append(p); pooled_y.append(y)
    pa, pf = metrics(np.concatenate(pooled_y), np.concatenate(pooled_p))
    print(f"  {name:<16} {100*np.mean(accs):6.2f}% ± {100*np.std(accs, ddof=1):.2f}"
          f"   F1 {100*np.mean(f1s):6.2f}%   pooled {100*pa:6.2f}%  (n={Xtr.shape[1]})")
    return np.mean(accs), pa


def main():
    tr, tests, cols = load()
    ytr = label(tr)
    tests_y = [label(t) for t in tests]
    Rtr = tr[cols].values
    Rte = [t[cols].values for t in tests]
    print(f"train {Rtr.shape}, {len(tests)} test sets, "
          f"{sum(len(y) for y in tests_y)} pooled planets\n")
    print("  config              5-set mean          F1        pooled")

    run("raw", Rtr, ytr, Rte, tests_y)

    sc = StandardScaler().fit(Rtr)
    run("raw + z-score", sc.transform(Rtr), ytr, [sc.transform(x) for x in Rte], tests_y)

    pca = PCA(n_components=102, random_state=SEED).fit(sc.transform(Rtr))
    post = StandardScaler().fit(pca.transform(sc.transform(Rtr)))
    tf = lambda x: post.transform(pca.transform(sc.transform(x)))
    run("PCA-102", tf(Rtr), ytr, [tf(x) for x in Rte], tests_y)


if __name__ == "__main__":
    main()
