"""
MLP aerosol robustness, averaged over restarts.

The single-run MLP columns in H2_cloudy_evaluation and H2_hazy_evaluation are
not quotable: the network's clear baseline varies by several accuracy points
between identically-configured training runs, and the hazy column is
non-monotonic in single runs. This script retrains the same whitened MLP
N_RESTARTS times - identical frozen preprocessing, architecture and data,
different initialisation and training-order seeds - and reports mean +/- sd
for every cloudy and hazy test set, which is the form the manuscript can
quote. XGBoost needs none of this (bit-identical every run) and is not
re-evaluated here.

Usage:
    python evaluate_aerosol_mlp_restarts.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

SEED = 42
N_COMPONENTS = 102
N_RESTARTS = 5
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
CLEAN_TEST_FMT = "multirex_spectra_H2_test_set_{}.parquet"


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def spectra_of(df, reference_grid):
    """Extract spectra by grid position (see evaluate_cloudy.py for why)."""
    cols = spectral_cols(df)
    grid = np.array([float(c) for c in cols])
    assert len(grid) == len(reference_grid)
    assert np.allclose(grid, reference_grid, rtol=1e-9)
    return df[cols].values


def build_mlp(dim):
    import tensorflow as tf
    from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                         Dropout, Input)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    tf.keras.backend.clear_session()
    m = Sequential()
    m.add(Input(shape=(dim,)))
    for u in [256, 128, 64]:
        m.add(Dense(u)); m.add(BatchNormalization())
        m.add(Activation("relu")); m.add(Dropout(0.3))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")

    df_tr = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df_tr)
    ref_grid = np.array([float(c) for c in cols])
    X_tr = df_tr[cols].values
    y_tr = (df_tr["biosignature"] == "yes").astype(int).values

    # Frozen preprocessing, fit once on clean training data - identical for
    # every restart, so only the network training itself varies.
    scaler_raw = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(scaler_raw.transform(X_tr))
    P_tr = pca.transform(scaler_raw.transform(X_tr))
    scaler_pca = StandardScaler().fit(P_tr)
    W_tr = scaler_pca.transform(P_tr)

    # Test sets: 5 clear (averaged into one baseline row), 5 cloudy, 5 hazy.
    clear = [pd.read_parquet(CLEAN_TEST_FMT.format(i)) for i in range(1, 6)]
    cloudy = sorted(glob.glob("multirex_spectra_H2_cloudy_*Pa.parquet"),
                    key=lambda f: -float(re.search(r"cloudy_(.+?)Pa", f).group(1)))
    hazy = sorted(glob.glob("multirex_spectra_H2_hazy_*.parquet"),
                  key=lambda f: float(re.search(r"hazy_(.+?)\.parquet", f).group(1)
                                      .replace("p", ".")))
    named = ([("clear (5 test sets)", None)]
             + [(f"cloud top {float(re.search(r'cloudy_(.+?)Pa', f).group(1)):.0e} Pa", f)
                for f in cloudy]
             + [(f"haze {float(re.search(r'hazy_(.+?)[.]parquet', f).group(1).replace('p','.')):.1e} m^-3", f)
                for f in hazy])

    feats = {}
    for name, f in named[1:]:
        d = pd.read_parquet(f)
        feats[name] = (scaler_pca.transform(pca.transform(
            scaler_raw.transform(spectra_of(d, ref_grid)))),
            (d["biosignature"] == "yes").astype(int).values)
    clear_feats = [(scaler_pca.transform(pca.transform(
        scaler_raw.transform(spectra_of(d, ref_grid)))),
        (d["biosignature"] == "yes").astype(int).values) for d in clear]

    acc = {name: [] for name, _ in named}
    bri = {name: [] for name, _ in named}
    for r in range(N_RESTARTS):
        rs = 1000 + r
        tf.random.set_seed(rs)
        Xw, yw = shuffle(W_tr, y_tr, random_state=rs)
        mlp = build_mlp(N_COMPONENTS)
        mlp.fit(Xw, yw, epochs=200, batch_size=128, validation_split=0.2,
                callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True)], verbose=0)

        ca, cb = [], []
        for Xc, yc in clear_feats:
            p = mlp.predict(Xc, verbose=0).ravel()
            ca.append(accuracy_score(yc, p > 0.5))
            cb.append(brier_score_loss(yc, np.clip(p, 1e-6, 1 - 1e-6)))
        acc["clear (5 test sets)"].append(np.mean(ca))
        bri["clear (5 test sets)"].append(np.mean(cb))
        for name, _ in named[1:]:
            Xc, yc = feats[name]
            p = mlp.predict(Xc, verbose=0).ravel()
            acc[name].append(accuracy_score(yc, p > 0.5))
            bri[name].append(brier_score_loss(yc, np.clip(p, 1e-6, 1 - 1e-6)))
        print(f"restart {r + 1}/{N_RESTARTS}: clear {acc['clear (5 test sets)'][-1]:.2%}",
              flush=True)

    hdr = (f"{'test set':<22} {'MLP acc mean':>12} {'acc sd':>7} "
           f"{'Brier mean':>11} {'Brier sd':>9}   individual restarts")
    lines = [f"Whitened MLP on aerosol test sets, {N_RESTARTS} training restarts",
             "",
             "Same frozen preprocessing and architecture as evaluate_cloudy.py /",
             "evaluate_hazy.py; each restart differs only in initialisation and",
             "training-order seed. Quote mean +/- sd - single runs vary by several",
             "accuracy points. XGBoost is deterministic and needs no averaging.",
             "", hdr, "-" * len(hdr)]
    rows = []
    for name, _ in named:
        a, b = np.array(acc[name]), np.array(bri[name])
        lines.append(f"{name:<22} {a.mean():>12.2%} {a.std():>7.2%} "
                     f"{b.mean():>11.4f} {b.std():>9.4f}   "
                     + " ".join(f"{v:.1%}" for v in a))
        rows.append((name, a.mean(), a.std(), b.mean(), b.std(),
                     *[round(v, 4) for v in a]))

    out = "\n".join(lines)
    print("\n" + out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_aerosol_mlp_restarts.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["test_set", "mlp_acc_mean", "mlp_acc_sd",
                                "mlp_brier_mean", "mlp_brier_sd",
                                *[f"restart_{i}" for i in range(N_RESTARTS)]]
                 ).to_csv("final_results/H2_aerosol_mlp_restarts.csv", index=False)
    print("\nWrote final_results/H2_aerosol_mlp_restarts.{txt,csv}")


if __name__ == "__main__":
    main()
