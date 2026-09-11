"""Frozen pipeline on Exo-Transmit spectra: independent code, same opacity data.

R1-3 axis 1. The comparison is paired planet-by-planet against the committed
clear test sets, so the 88.92% baseline is valid and nothing is resampled.

What this isolates, and what it does not:

  * Opacity data is IDENTICAL - Exo-Transmit's tables are the byte-identical
    files MultiREx ships. Whatever difference appears is the radiative transfer
    implementation, not the cross sections. That is the complement of the
    axis-2 experiment, which swaps line lists while holding the code fixed.
  * One physics difference could not be matched: Exo-Transmit assumes constant
    gravity through the atmosphere, TauREx integrates with gravity falling as
    altitude rises. Rayleigh-only tests show this costs 4-12% of spectral
    contrast, scaling with the atmosphere's vertical extent. It is a genuine
    property of the two codes rather than a setting, and is reported alongside
    the result rather than hidden.

Design invariants match evaluate_cloudy.py: models trained once on the
committed training set, never retrained; scalers and PCA fit on training data
only; MLP figures are restart means.

Usage:
    python evaluate_exotransmit.py
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
N_RESTARTS = 5
N_SETS = 5
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
TAUREX_FMT = "multirex_spectra_H2_test_set_{}.parquet"
EXO_FMT = "multirex_spectra_H2_exotransmit_set_{}.parquet"
OPACITY_SWAP_DELTA = -16.09   # axis 2, from H2_opacity_swap.txt


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


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

    for i in range(1, N_SETS + 1):
        if not os.path.exists(EXO_FMT.format(i)):
            raise SystemExit(f"missing {EXO_FMT.format(i)} - run "
                             f"generate_exotransmit_testset.py first")

    df_tr = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df_tr)
    ref = np.array([float(c) for c in cols])
    X_tr = df_tr[cols].values
    y_tr = (df_tr["biosignature"] == "yes").astype(int).values

    scaler_raw = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(
        scaler_raw.transform(X_tr))
    P_tr = pca.transform(scaler_raw.transform(X_tr))
    scaler_pca = StandardScaler().fit(P_tr)
    amp_tr = np.median(X_tr.std(axis=1) / X_tr.mean(axis=1))

    Pt, Pe, Y, at, ae, cors = [], [], [], [], [], []
    nbad = ntot = 0
    for i in range(1, N_SETS + 1):
        dt = pd.read_parquet(TAUREX_FMT.format(i))
        de = pd.read_parquet(EXO_FMT.format(i))
        assert len(dt) == len(de)
        assert (dt["biosignature"].values == de["biosignature"].values).all()
        ct, ce = spectral_cols(dt), spectral_cols(de)
        assert np.allclose([float(c) for c in ct], ref, rtol=1e-9)
        v = de["exotransmit_valid"].values.astype(bool)
        nbad += int((~v).sum()); ntot += len(v)

        Xt = dt[ct].values[v]
        Xe = de[ce].values[v]
        Pt.append(pca.transform(scaler_raw.transform(Xt)))
        Pe.append(pca.transform(scaler_raw.transform(Xe)))
        Y.append((dt["biosignature"] == "yes").astype(int).values[v])
        at.append(np.median(Xt.std(axis=1) / Xt.mean(axis=1)) / amp_tr)
        ae.append(np.median(Xe.std(axis=1) / Xe.mean(axis=1)) / amp_tr)
        cors += [np.corrcoef(Xt[k], Xe[k])[0, 1] for k in range(len(Xt))]

    print(f"paired on {ntot - nbad} of {ntot} planets ({nbad} excluded)")
    print(f"feature amplitude: TauREx {np.mean(at):.3f}x  "
          f"Exo-Transmit {np.mean(ae):.3f}x  ratio {np.mean(ae)/np.mean(at):.3f}")
    print(f"median per-planet correlation {np.median(cors):.4f}\n", flush=True)

    Xs, ys = shuffle(P_tr, y_tr, random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                        subsample=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(Xs, ys)

    def sc(P, y):
        p = xgb.predict_proba(P)[:, 1]
        return (accuracy_score(y, p > 0.5), f1_score(y, p > 0.5, zero_division=0),
                brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), p > 0.5)

    rt = [sc(P, y) for P, y in zip(Pt, Y)]
    re_ = [sc(P, y) for P, y in zip(Pe, Y)]
    yall = np.concatenate(Y)
    acc_t = accuracy_score(yall, np.concatenate([r[3] for r in rt]))
    acc_e = accuracy_score(yall, np.concatenate([r[3] for r in re_]))
    bri_t = float(np.mean([r[2] for r in rt]))
    bri_e = float(np.mean([r[2] for r in re_]))
    print(f"XGBoost  TauREx {acc_t:.2%}   Exo-Transmit {acc_e:.2%}   "
          f"({100*(acc_e-acc_t):+.2f} pts)\n", flush=True)

    W_tr = scaler_pca.transform(P_tr)
    mt, me = [], []
    for r in range(N_RESTARTS):
        rs = 1000 + r
        tf.random.set_seed(rs)
        Xw, yw = shuffle(W_tr, y_tr, random_state=rs)
        m = build_mlp(N_COMPONENTS)
        m.fit(Xw, yw, epochs=200, batch_size=128, validation_split=0.2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)], verbose=0)
        mt.append(np.mean([accuracy_score(y, m.predict(scaler_pca.transform(P),
                  verbose=0).ravel() > 0.5) for P, y in zip(Pt, Y)]))
        me.append(np.mean([accuracy_score(y, m.predict(scaler_pca.transform(P),
                  verbose=0).ravel() > 0.5) for P, y in zip(Pe, Y)]))
        print(f"restart {r+1}/{N_RESTARTS}", flush=True)
    mt, me = np.array(mt), np.array(me)

    d = 100 * (acc_e - acc_t)
    L = ["Frozen pipeline on Exo-Transmit spectra (independent code, identical opacities)",
         "",
         "Exo-Transmit is the code MultiREx's opacity tables come from; the .dat",
         "files are byte-identical. The radiative transfer implementation changes",
         "while every cross section stays the same file. Paired planet-by-planet",
         "against the committed clear test sets.",
         "",
         f"Paired on {ntot - nbad} of {ntot} planets.",
         f"Feature amplitude vs training set: TauREx {np.mean(at):.3f}x, "
         f"Exo-Transmit {np.mean(ae):.3f}x.",
         f"Median per-planet spectral correlation: {np.median(cors):.4f}.",
         "",
         f"{'model':<24}{'TauREx':>12}{'Exo-Transmit':>15}{'change (pts)':>14}",
         "-" * 65,
         f"{'XGBoost':<24}{acc_t:>12.2%}{acc_e:>15.2%}{d:>14.2f}",
         f"{'MLP (5 restarts)':<24}{mt.mean():>8.2%} +/-{mt.std():>3.1%}"
         f"{me.mean():>11.2%} +/-{me.std():>3.1%}{100*(me.mean()-mt.mean()):>14.2f}",
         f"{'XGBoost Brier':<24}{bri_t:>12.4f}{bri_e:>15.4f}{bri_e-bri_t:>+14.4f}",
         "",
         "=" * 65,
         "Code versus opacity data", "",
         f"  change the OPACITY DATA, hold the code fixed (axis 2)  "
         f"{OPACITY_SWAP_DELTA:>+7.2f}",
         f"  change the CODE, hold the opacity data fixed (axis 1)  {d:>+7.2f}",
         ""]

    if abs(d) < abs(OPACITY_SWAP_DELTA) / 2:
        L += ["  The classifier is substantially more sensitive to which opacity",
              "  data is used than to which radiative transfer code evaluates it.",
              "  That is a statement about the physics inputs rather than about",
              "  simulator choice, and it is the useful form of the result: a",
              "  cross-code test alone would have conflated the two."]
    else:
        L += ["  The code change costs a comparable amount to the opacity change,",
              "  so radiative transfer implementation is not a second-order",
              "  effect here and both must be reported as sources of fragility."]

    L += ["", "=" * 65, "The difference that could not be matched", "",
          "  Exo-Transmit assumes constant gravity through the atmosphere;",
          "  TauREx integrates hydrostatically with gravity falling as altitude",
          "  rises. Rayleigh-only comparisons (all molecular opacity off, pure",
          "  H2) show mean transit depth agreeing to 0.4% - so geometry, mean",
          "  molecular weight and the radius convention are correct - while",
          "  spectral contrast differs by 4-12%, scaling with the atmosphere's",
          "  vertical extent relative to planet radius:",
          "",
          "     thickness/R 0.006 -> 3.7% contrast deficit",
          "     thickness/R 0.061 -> 7.1%",
          "     thickness/R 0.147 -> 11.9%",
          "",
          "  These atmospheres span 11-12 scale heights, so the top sits at up",
          "  to 0.15 planet radii where gravity is ~25% lower than at the base.",
          "  This is a genuine difference between the codes, not a setting: it",
          "  cannot be removed without editing Exo-Transmit's source, which",
          "  would defeat the point of using an independent implementation.",
          "  Report it as part of what 'different radiative transfer code'",
          "  means, and note that it inflates the measured difference relative",
          "  to a pure numerical-scheme comparison."]

    out = "\n".join(L)
    print("\n" + out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_exotransmit.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame([("XGBoost", acc_t, acc_e, 100 * (acc_e - acc_t), bri_t, bri_e),
                  ("MLP", mt.mean(), me.mean(), 100 * (me.mean() - mt.mean()),
                   np.nan, np.nan)],
                 columns=["model", "taurex_acc", "exotransmit_acc", "delta_pts",
                          "taurex_brier", "exotransmit_brier"]
                 ).to_csv("final_results/H2_exotransmit.csv", index=False)
    print("\nWrote final_results/H2_exotransmit.{txt,csv}")


if __name__ == "__main__":
    main()
