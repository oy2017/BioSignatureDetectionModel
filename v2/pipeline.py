"""Stage 3: in-domain study at one observing configuration.

For a configuration (ariel | r100 | r200) and the noise convention in common.py:
  1. tune XGBoost and Random Forest (5-fold CV on the training set) on each
     feature set, and the MLP on the whitened PCA features;
  2. evaluate on the five test sets: accuracy, precision, recall, F1, Brier,
     ECE, AUC, mean +- sd; pooled McNemar tests between models; paired
     bootstrap on F1 for the two best;
  3. save the frozen best pipeline (features + model) to models/, used by every
     shift-axis script; save per-planet probabilities for the pooled test sets.

Usage:
    python pipeline.py --config ariel                # full: tune + evaluate
    python pipeline.py --config r100 --reuse-params  # ladder rung: reuse ariel's tuned hyperparameters
    python pipeline.py --config ariel --quick        # small grids, for smoke tests
Outputs: results/{config}_tuning.json, results/{config}_indomain.txt,
         results/{config}_probs.parquet, models/{config}_{features}_{model}.joblib
"""
import argparse
import itertools
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from common import (MODELS, RESULTS, SEED, TESTS, Features, fmt, load_split,
                    metrics, summarize)

warnings.filterwarnings("ignore")

XGB_GRID = {"n_estimators": [200, 500], "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8]}
RF_GRID = {"n_estimators": [300, 500], "max_depth": [None, 20],
           "min_samples_leaf": [1, 2, 4]}
MLP_GRID = {"hidden": [(256, 128, 64), (512, 256, 128)], "dropout": [0.2, 0.4],
            "batch": [128]}
QUICK = {"xgb": {"n_estimators": [200], "max_depth": [5], "learning_rate": [0.1], "subsample": [0.8]},
         "rf": {"n_estimators": [200], "max_depth": [None], "min_samples_leaf": [2]},
         "mlp": {"hidden": [(256, 128, 64)], "dropout": [0.3], "batch": [128]}}
FEATURE_SETS = ["raw", "pca", "norm"]      # tree models
MLP_FEATURES = ["norm", "pcaw", "pca"]      # per-spectrum normalized; whitened vs unwhitened PCA (the whitening trade-off)


def grid(d):
    keys = list(d)
    for vals in itertools.product(*[d[k] for k in keys]):
        yield dict(zip(keys, vals))


def make_xgb(p):
    return XGBClassifier(**p, eval_metric="logloss", random_state=SEED, n_jobs=8, tree_method="hist")


def make_rf(p):
    return RandomForestClassifier(**p, random_state=SEED, n_jobs=8)


def tune_sklearn(make, gridspec, Z, y, cv):
    best, best_score, log = None, -1, []
    for p in grid(gridspec):
        s = cross_val_score(make(p), Z, y, cv=cv, scoring="accuracy", n_jobs=1).mean()
        log.append({**p, "cv_accuracy": s})
        if s > best_score:
            best, best_score = p, s
    return best, best_score, log


# ---- MLP -------------------------------------------------------------------
def build_mlp(n_in, hidden, dropout, lr=1e-3):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.random.set_seed(SEED)
    m = models.Sequential([layers.Input(shape=(n_in,))])
    for u in hidden:
        m.add(layers.Dense(u))
        m.add(layers.BatchNormalization())
        m.add(layers.Activation("relu"))
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def fit_mlp(Z, y, p, seed=SEED, epochs=200):
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)
    m = build_mlp(Z.shape[1], p["hidden"], p["dropout"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    m.fit(Z, y, epochs=epochs, batch_size=p["batch"], validation_split=0.2, callbacks=[es], verbose=0)
    return m


class MLPWrap:
    """Pickle-free wrapper: keeps the Keras model in memory; saved separately."""
    def __init__(self, model):
        self.model = model

    def predict_proba(self, Z):
        p = self.model.predict(Z, verbose=0).ravel()
        return np.c_[1 - p, p]


def tune_mlp(gridspec, Z, y, cv):
    best, best_score, log = None, -1, []
    for p in grid(gridspec):
        accs = []
        for tr, va in cv.split(Z, y):
            m = fit_mlp(Z[tr], y[tr], p, epochs=60)
            accs.append(((m.predict(Z[va], verbose=0).ravel() >= 0.5) == y[va]).mean())
        s = float(np.mean(accs))
        log.append({**{k: str(v) for k, v in p.items()}, "cv_accuracy": s})
        if s > best_score:
            best, best_score = p, s
    return best, best_score, log


# ---- statistics ------------------------------------------------------------
def mcnemar(y, a, b):
    """Exact McNemar test on paired predictions; returns (b, c, p)."""
    ca, cb = a == y, b == y
    n01 = int((ca & ~cb).sum())
    n10 = int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, p


def bootstrap_f1_gap(y, pa, pb, n=10000, seed=SEED):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    gaps = np.empty(n)
    idx = np.arange(len(y))
    for i in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        gaps[i] = f1_score(y[s], pa[s]) - f1_score(y[s], pb[s])
    return gaps.mean(), np.percentile(gaps, [2.5, 97.5])


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ariel")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse-params", action="store_true",
                    help="reuse results/ariel_tuning.json hyperparameters (ladder rungs)")
    ap.add_argument("--mlp-restarts", type=int, default=5)
    ap.add_argument("--no-mlp", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    cfg = a.config
    t0 = time.time()

    Xtr, ytr, Ptr = load_split("train", cfg)
    tests = [load_split(t, cfg) for t in TESTS]
    print(f"[{cfg}] train {Xtr.shape}, tests {[t[0].shape[0] for t in tests]}, "
          f"positives {ytr.mean():.3f}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tuning = {}
    if a.reuse_params:
        with open(os.path.join(RESULTS, "ariel_tuning.json")) as f:
            tuning = json.load(f)
    xgb_grid, rf_grid, mlp_grid = (QUICK["xgb"], QUICK["rf"], QUICK["mlp"]) if a.quick else (XGB_GRID, RF_GRID, MLP_GRID)

    feats = {}
    for kind in set(FEATURE_SETS + MLP_FEATURES):
        feats[kind] = Features(kind).fit(Xtr)
    kinfo = {k: (f.k if f.pca is not None else Xtr.shape[1]) for k, f in feats.items()}
    print(f"[{cfg}] feature dims: {kinfo}", flush=True)

    Ztr = {k: f.transform(Xtr) for k, f in feats.items()}
    Zte = {k: [f.transform(t[0]) for t in tests] for k, f in feats.items()}
    yte = [t[1] for t in tests]
    ypool = np.concatenate(yte)

    lines = [f"In-domain study, configuration = {cfg}, SNR {15} (peak-to-peak convention), "
             f"Ariel-shaped noise", f"train n = {len(ytr)}, test sets n = {[len(y) for y in yte]}",
             f"feature dimensions: {kinfo}", ""]
    results = {}        # (features, model) -> dict(summary, pooled_pred, pooled_prob)
    probs_out = {}

    # --- tree models on each feature set
    for kind in FEATURE_SETS:
        for name, make, gspec in (("xgb", make_xgb, xgb_grid), ("rf", make_rf, rf_grid)):
            key = f"{kind}_{name}"
            if a.reuse_params and key in tuning:
                best = tuning[key]["best"]
            else:
                best, score, log = tune_sklearn(make, gspec, Ztr[kind], ytr, cv)
                tuning[key] = {"best": best, "cv_accuracy": score, "log": log}
                print(f"[{cfg}] tuned {key}: {best} cv {score:.4f} ({time.time()-t0:.0f}s)", flush=True)
            model = make(best).fit(Ztr[kind], ytr)
            rows, preds, pp = [], [], []
            for Z, y in zip(Zte[kind], yte):
                p = model.predict_proba(Z)[:, 1]
                rows.append(metrics(y, p)); preds.append((p >= 0.5).astype(int)); pp.append(p)
            results[key] = dict(summary=summarize(rows), pred=np.concatenate(preds),
                                prob=np.concatenate(pp), per_set=rows)
            probs_out[key] = results[key]["prob"]
            joblib.dump({"features": feats[kind], "model": model, "config": cfg, "params": best},
                        os.path.join(MODELS, f"{cfg}_{key}.joblib"))
            lines.append(f"{key:10s} {fmt(results[key]['summary'])}")
            print(lines[-1], flush=True)

    # --- MLP: whitened vs unwhitened PCA, restarts
    if not a.no_mlp:
        for kind in MLP_FEATURES:
            key = f"{kind}_mlp"
            if a.reuse_params and key in tuning:
                best = tuning[key]["best"]
                best["hidden"] = tuple(best["hidden"])
            else:
                best, score, log = tune_mlp(mlp_grid, Ztr[kind], ytr, cv)
                tuning[key] = {"best": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
                               "cv_accuracy": score, "log": log}
                print(f"[{cfg}] tuned {key}: {best} cv {score:.4f} ({time.time()-t0:.0f}s)", flush=True)
            restarts = []
            for r in range(a.mlp_restarts):
                m = fit_mlp(Ztr[kind], ytr, best, seed=SEED + r)
                rows, preds, pp = [], [], []
                for Z, y in zip(Zte[kind], yte):
                    p = m.predict(Z, verbose=0).ravel()
                    rows.append(metrics(y, p)); preds.append((p >= 0.5).astype(int)); pp.append(p)
                restarts.append(dict(summary=summarize(rows), pred=np.concatenate(preds),
                                     prob=np.concatenate(pp), per_set=rows))
                if r == 0:
                    m.save(os.path.join(MODELS, f"{cfg}_{key}.keras"))
                    joblib.dump({"features": feats[kind], "config": cfg, "params": best,
                                 "keras_path": os.path.join(MODELS, f"{cfg}_{key}.keras")},
                                os.path.join(MODELS, f"{cfg}_{key}.joblib"))
            # report mean over restarts of the five-set means, and restart scatter
            acc_r = [x["summary"]["accuracy"][0] for x in restarts]
            summ = {k: (np.mean([x["summary"][k][0] for x in restarts]),
                        np.mean([x["summary"][k][1] for x in restarts])) for k in restarts[0]["summary"]}
            results[key] = dict(summary=summ, pred=restarts[0]["pred"], prob=restarts[0]["prob"],
                                restart_acc=acc_r)
            probs_out[key] = restarts[0]["prob"]
            lines.append(f"{key:10s} {fmt(summ)}   restarts acc {np.mean(acc_r)*100:.2f}±{np.std(acc_r, ddof=1)*100:.2f} "
                         f"(n={a.mlp_restarts})")
            print(lines[-1], flush=True)

    # --- statistics on the pooled test sets
    lines += ["", "Pairwise McNemar (exact) on pooled test sets: (b = A right/B wrong, c = A wrong/B right, p)"]
    keys = list(results)
    best_key = max(keys, key=lambda k: results[k]["summary"]["accuracy"][0])
    for k in keys:
        if k == best_key:
            continue
        b, c, p = mcnemar(ypool, results[best_key]["pred"], results[k]["pred"])
        lines.append(f"  {best_key} vs {k:10s}: b={b:5d} c={c:5d} p={p:.2e}")
    second = sorted([k for k in keys if k != best_key], key=lambda k: -results[k]["summary"]["accuracy"][0])[0]
    g, ci = bootstrap_f1_gap(ypool, results[best_key]["pred"], results[second]["pred"])
    lines.append(f"Paired bootstrap F1 gap {best_key} - {second}: {g*100:.2f} pts, 95% CI [{ci[0]*100:.2f}, {ci[1]*100:.2f}]")
    lines.append(f"\nBest pipeline: {best_key}  (frozen at models/{cfg}_{best_key}.joblib)")
    with open(os.path.join(RESULTS, f"{cfg}_indomain.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS, f"{cfg}_tuning.json"), "w") as f:
        json.dump(tuning, f, indent=1, default=str)
    pd.DataFrame({"y": ypool, **probs_out}).to_parquet(os.path.join(RESULTS, f"{cfg}_probs.parquet"))
    json.dump({"best": best_key, "feature_dims": kinfo}, open(os.path.join(RESULTS, f"{cfg}_best.json"), "w"))
    summary = {k: {m: [float(v[0]), float(v[1])] for m, v in r["summary"].items()} for k, r in results.items()}
    for k, r in results.items():
        if "restart_acc" in r:
            summary[k]["restart_acc"] = [float(x) for x in r["restart_acc"]]
    json.dump({"config": cfg, "n_train": int(len(ytr)), "feature_dims": kinfo, "best": best_key,
               "models": summary}, open(os.path.join(RESULTS, f"{cfg}_summary.json"), "w"), indent=1)
    print("\n".join(lines))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
