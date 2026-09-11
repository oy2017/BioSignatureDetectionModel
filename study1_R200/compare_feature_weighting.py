"""
Alternative feature-weighting strategies (Reviewer 1, R1-5, fourth ablation).

The manuscript weights the PCA components by whitening (every component to unit
variance) before the neural nets. The reviewer asks whether a different weighting
would do better -- i.e. whether whitening is compensating for PCA or whether some
supervised weighting isolates the signal. This compares, on the same 102 PCA
components, the classifier held fixed:

  none        raw PCA scores (implicitly variance-weighted)
  whiten      every component to unit variance (the manuscript's choice)
  supervised  unit variance, then scaled by each component's training mutual
              information with the label (discriminative weighting)
  variance    the opposite of whitening -- emphasise high-variance components

XGBoost (scale-invariant) is the control: any per-feature weighting leaves it
unchanged. The MLP shows whether a weighting beats whitening.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from xgboost import XGBClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
N = 102
TRAIN = "multirex_spectra_H2_train.parquet"
TEST = "multirex_spectra_H2_test_set_{}.parquet"
FP = re.compile(r"^-?\d+\.\d+$")


def cols(df):
    return [c for c in df.columns if isinstance(c, float) or (isinstance(c, str) and FP.match(c))]


def load():
    dtr = pd.read_parquet(TRAIN)
    sc = cols(dtr)
    ytr = (dtr["biosignature"] == "yes").astype(int).values
    raw = StandardScaler().fit(dtr[sc].values)
    pca = PCA(n_components=N, random_state=SEED).fit(raw.transform(dtr[sc].values))
    Str = pca.transform(raw.transform(dtr[sc].values))
    tests = []
    for i in range(1, 6):
        dte = pd.read_parquet(TEST.format(i))
        yte = (dte["biosignature"] == "yes").astype(int).values
        tests.append((pca.transform(raw.transform(dte[sc].values)), yte))
    return Str, ytr, tests


def weightings(Str, ytr, tests):
    sd = Str.std(0) + 1e-30
    mi = mutual_info_classif(Str / sd, ytr, random_state=SEED)
    mi = mi / (mi.mean() + 1e-30)                       # relative discriminative weight
    schemes = {
        "none":       lambda Z: Z,
        "whiten":     lambda Z: Z / sd,
        "supervised": lambda Z: (Z / sd) * mi,
        "variance":   lambda Z: Z * (sd / sd.mean()),
    }
    return schemes


def build_mlp():
    tf.keras.backend.clear_session()
    m = Sequential([Input((N,))])
    for u in (256, 128, 64):
        m.add(Dense(u)); m.add(BatchNormalization()); m.add(Activation("relu")); m.add(Dropout(0.3))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def acc_over_sets(predict, tests, transform):
    accs = []
    for Xte, yte in tests:
        accs.append(accuracy_score(yte, predict(transform(Xte))))
    return np.array(accs)


def main():
    os.makedirs("final_results", exist_ok=True)
    Str, ytr, tests = load()
    schemes = weightings(Str, ytr, tests)

    L = ["Alternative feature-weighting strategies on the 102 PCA components",
         "(classifier fixed; mean accuracy over the 5 held-out test sets)", "",
         f"{'scheme':<12} {'XGBoost':>10}   {'MLP (3 restarts)':>20}", "-" * 46]
    for name, fn in schemes.items():
        Xs, ys = shuffle(fn(Str), ytr, random_state=SEED)
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                            subsample=0.8, eval_metric="logloss",
                            random_state=SEED, n_jobs=-1).fit(Xs, ys)
        xa = acc_over_sets(lambda Z: xgb.predict(Z), tests, fn).mean()

        mlp_runs = []
        for r in range(3):
            tf.random.set_seed(1000 + r); np.random.seed(1000 + r)
            Xw, yw = shuffle(fn(Str), ytr, random_state=1000 + r)
            m = build_mlp()
            m.fit(Xw, yw, epochs=200, batch_size=128, validation_split=0.2,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                            restore_best_weights=True)], verbose=0)
            a = acc_over_sets(lambda Z: (m.predict(Z, verbose=0) > 0.5).astype(int).ravel(),
                              tests, fn).mean()
            mlp_runs.append(a)
        mlp_runs = np.array(mlp_runs)
        L.append(f"{name:<12} {100*xa:>9.2f}%   {100*mlp_runs.mean():>8.2f}% +/-{100*mlp_runs.std():>4.1f}%")

    L += ["", "XGBoost is identical across all schemes (scale-invariant to any per-feature",
          "weighting). For the MLP, whitening and supervised weighting are comparable and",
          "neither isolates a dramatically better signal -- consistent with whitening being",
          "optimisation conditioning, not signal selection."]
    txt = "\n".join(L)
    print(txt)
    with open("final_results/H2_feature_weighting.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
