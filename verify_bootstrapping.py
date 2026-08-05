"""Paired bootstrap of the F1 difference between XGBoost and the Random Forest.

Replaces an earlier version that did not bootstrap the models at all: it took two
hardcoded F1 values, simulated each as independent Bernoulli draws, and
bootstrapped the difference of those simulated means. That is invalid twice over
-- F1 is a ratio of counts rather than a mean of independent Bernoulli trials,
and simulating the two models independently discards the pairing, which is the
whole reason the result disagreed with McNemar's test.

This version resamples the actual test planets. Each iteration draws 2,697
planet indices with replacement and applies the *same* indices to both models, so
the pairing is preserved, and recomputes F1 for each from its own confusion
counts.

Note on interpretation: overlapping marginal confidence intervals for two F1
scores do not imply the difference is insignificant. The paired difference
interval below is the quantity that answers that question.
"""
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
ITERATIONS = 10000
FLOAT_COL = re.compile(r"^-?\d+\.\d+$")

label = lambda df: df["biosignature"].apply(lambda x: 1 if x == "yes" else 0).values


def get_data():
    """Same pipeline as run_master_5set_evaluation.py, evaluated on the pooled sets."""
    df_train = pd.read_parquet("multirex_spectra_H2_train.parquet")
    cols = [c for c in df_train.columns
            if isinstance(c, float) or (isinstance(c, str) and FLOAT_COL.match(c))]
    y_train = label(df_train)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_train[cols].values)
    pca = PCA(n_components=102, random_state=SEED)
    X = pca.fit_transform(X)
    post = StandardScaler()
    X_train = post.fit_transform(X)
    X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

    df_test = pd.concat([pd.read_parquet(f"multirex_spectra_H2_test_set_{i}.parquet")
                         for i in range(1, 6)], ignore_index=True)
    X_test = post.transform(pca.transform(scaler.transform(df_test[cols].values)))
    return X_train, y_train, X_test, label(df_test)


def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def paired_bootstrap(y, pred_a, pred_b, iterations=ITERATIONS, seed=SEED):
    """Resample planets with replacement, same indices for both models."""
    n = len(y)
    ind = {}
    for name, p in (("a", pred_a), ("b", pred_b)):
        ind[name] = (((y == 1) & (p == 1)).astype(np.int64),   # tp
                     ((y == 0) & (p == 1)).astype(np.int64),   # fp
                     ((y == 1) & (p == 0)).astype(np.int64))   # fn
    rng = np.random.default_rng(seed)
    fa = np.empty(iterations)
    fb = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        fa[i] = f1_from_counts(*[v[idx].sum() for v in ind["a"]])
        fb[i] = f1_from_counts(*[v[idx].sum() for v in ind["b"]])
    return fa, fb


def main():
    print("--- Loading data and training ---")
    X_train, y_train, X_test, y_test = get_data()
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                        eval_metric="logloss", random_state=SEED,
                        n_jobs=-1).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=2,
                                max_depth=None, random_state=SEED,
                                n_jobs=-1).fit(X_train, y_train)
    p_xgb, p_rf = xgb.predict(X_test), rf.predict(X_test)

    obs = {}
    for name, p in (("XGBoost", p_xgb), ("Random Forest", p_rf)):
        tp = int(((y_test == 1) & (p == 1)).sum())
        fp = int(((y_test == 0) & (p == 1)).sum())
        fn = int(((y_test == 1) & (p == 0)).sum())
        obs[name] = float(f1_from_counts(tp, fp, fn))
        print(f"    {name:<14} accuracy {100*(p == y_test).mean():.2f}%   F1 {100*obs[name]:.2f}%")

    print(f"\n--- Paired bootstrap, {ITERATIONS} iterations, n = {len(y_test)} ---")
    fa, fb = paired_bootstrap(y_test, p_xgb, p_rf)
    diff = fa - fb

    for name, arr in (("XGBoost", fa), ("Random Forest", fb)):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"    {name:<14} F1 95% CI [{100*lo:.2f}%, {100*hi:.2f}%]")

    lo, hi = np.percentile(diff, [2.5, 97.5])
    # Two-sided bootstrap p: twice the smaller tail mass at zero.
    p_val = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    p_val = min(p_val, 1.0)
    print(f"\n    observed F1 difference   {100*(obs['XGBoost'] - obs['Random Forest']):+.2f} points")
    print(f"    paired 95% CI of diff    [{100*lo:+.2f}, {100*hi:+.2f}] points")
    print(f"    two-sided bootstrap p    {p_val:.4f}")

    if lo > 0 or hi < 0:
        print("\n-> The paired interval excludes zero: the F1 difference IS significant.")
    else:
        print("\n-> The paired interval includes zero: the F1 difference is not significant.")

    a_lo, a_hi = np.percentile(fa, [2.5, 97.5])
    b_lo, b_hi = np.percentile(fb, [2.5, 97.5])
    if a_lo < b_hi and b_lo < a_hi:
        print("   (The two marginal intervals overlap. That is expected and is not")
        print("    evidence against a difference -- the paired interval is the test.)")


if __name__ == "__main__":
    main()
