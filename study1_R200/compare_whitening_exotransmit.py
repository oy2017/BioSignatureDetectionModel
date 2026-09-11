"""
Whitening robustness under an independent radiative transfer code (R1-6).

Reviewer 1 asks whether whitening -- which amplifies every low-variance component
equally regardless of content -- reduces robustness *out of distribution*, and
asks specifically for a test on an independent simulation environment. The R1-6
evidence so far is within-simulator (TauREx spectra perturbed). This runs the
same whitened-vs-unwhitened comparison but evaluates on spectra recomputed for
the held-out planets with Exo-Transmit, an independently written code -- a real
domain shift rather than an injected perturbation.

Two otherwise identical MLPs are trained on the clean TauREx training set and
differ only in whether the post-PCA components are whitened (standardised to unit
variance). Each is trained five times; every model is scored on both the TauREx
clean test sets and the Exo-Transmit test sets. If whitening reduces robustness,
the whitened network should lose more accuracy moving to the independent code.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

N = 102
RESTARTS = 5
TRAIN = "multirex_spectra_H2_train.parquet"
TAUREX = "multirex_spectra_H2_test_set_{}.parquet"
EXO = "multirex_spectra_H2_exotransmit_set_{}.parquet"
FP = re.compile(r"^-?\d+\.\d+$")


def cols(df):
    return [c for c in df.columns if isinstance(c, float) or (isinstance(c, str) and FP.match(c))]


def build_mlp():
    tf.keras.backend.clear_session()
    m = Sequential([Input((N,))])
    for u in (256, 128, 64):
        m.add(Dense(u)); m.add(BatchNormalization()); m.add(Activation("relu")); m.add(Dropout(0.3))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def load():
    dtr = pd.read_parquet(TRAIN)
    sc = cols(dtr)
    ytr = (dtr["biosignature"] == "yes").astype(int).values
    raw = StandardScaler().fit(dtr[sc].values)
    pca = PCA(n_components=N, random_state=42).fit(raw.transform(dtr[sc].values))
    Ptr = pca.transform(raw.transform(dtr[sc].values))
    wsc = StandardScaler().fit(Ptr)                       # whitening

    def proj(fmt):
        P, y = [], []
        for i in range(1, 6):
            d = pd.read_parquet(fmt.format(i))
            P.append(pca.transform(raw.transform(d[sc].values)))
            y.append((d["biosignature"] == "yes").astype(int).values)
        return P, y

    taurex = proj(TAUREX)
    exo = proj(EXO)
    return Ptr, ytr, wsc, taurex, exo


def pooled_acc(model, sets, transform):
    correct = total = 0
    for P, y in zip(*sets):
        pred = (model.predict(transform(P), verbose=0) > 0.5).astype(int).ravel()
        correct += int((pred == y).sum()); total += len(y)
    return correct / total


def main():
    os.makedirs("final_results", exist_ok=True)
    Ptr, ytr, wsc, taurex, exo = load()
    ident = lambda P: P
    whiten = lambda P: wsc.transform(P)

    res = {"whitened": {"taurex": [], "exo": []}, "unwhitened": {"taurex": [], "exo": []}}
    for name, tf_fn in [("whitened", whiten), ("unwhitened", ident)]:
        for r in range(RESTARTS):
            tf.random.set_seed(100 + r); np.random.seed(100 + r)
            Xs, ys = shuffle(tf_fn(Ptr), ytr, random_state=100 + r)
            m = build_mlp()
            m.fit(Xs, ys, epochs=200, batch_size=128, validation_split=0.2,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                            restore_best_weights=True)], verbose=0)
            res[name]["taurex"].append(pooled_acc(m, taurex, tf_fn))
            res[name]["exo"].append(pooled_acc(m, exo, tf_fn))

    def ms(a):
        a = 100 * np.array(a)
        return a.mean(), a.std()

    L = ["Whitening robustness under an independent radiative transfer code (R1-6)",
         "",
         "Two identical MLPs, differing only in whitening, trained on clean TauREx and",
         f"scored on the TauREx clean test sets and on Exo-Transmit spectra of the same",
         f"planets. Means +/- sd over {RESTARTS} restarts (pooled accuracy).",
         "",
         f"{'model':<12} {'TauREx (clean)':>18} {'Exo-Transmit':>18} {'drop to indep. code':>22}",
         "-" * 74]
    for name in ("whitened", "unwhitened"):
        tm, ts = ms(res[name]["taurex"])
        em, es = ms(res[name]["exo"])
        L.append(f"{name:<12} {tm:>10.1f}% +/-{ts:>3.1f} {em:>11.1f}% +/-{es:>3.1f} "
                 f"{tm-em:>18.1f} pts")
    dw = ms(res["whitened"]["taurex"])[0] - ms(res["whitened"]["exo"])[0]
    du = ms(res["unwhitened"]["taurex"])[0] - ms(res["unwhitened"]["exo"])[0]
    L += ["",
          f"Drop moving to the independent code: whitened {dw:.1f} pts, unwhitened {du:.1f} pts.",
          ("Whitening loses more under the code shift, consistent with his hypothesis that a "
           "content-blind amplification of low-variance components hurts out-of-distribution."
           if dw > du + 0.3 else
           "The two lose comparably under the code shift; the effect seen under injected noise "
           "does not clearly reproduce under this particular (mild) code shift -- report as such.")]
    txt = "\n".join(L)
    print(txt)
    with open("final_results/H2_whitening_exotransmit.txt", "w") as f:
        f.write(txt + "\n")
    print("Wrote final_results/H2_whitening_exotransmit.txt")


if __name__ == "__main__":
    main()
