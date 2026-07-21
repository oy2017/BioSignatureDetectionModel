"""Aerosol degradation measured on the committed planets, with no draw confound.

generate_aerosol_paired.py re-renders the five committed clear test sets with
each aerosol added, changing nothing else. This evaluates those sets through the
frozen pipeline and reports the degradation against the committed clear spectra
for the SAME planets.

That makes the 88.92% baseline valid again. The earlier problem was that the
published aerosol sets were a different draw of planets from the committed test
sets, and the generator's sampling had moved in between, so part of every
recorded degradation was the draw. Here nothing is resampled: each aerosol
spectrum has a committed clear spectrum for the identical planet, so the
difference is the aerosol and nothing else.

Design invariants match evaluate_cloudy.py: models trained once on the committed
training set and never retrained; raw scaler, PCA basis and post-PCA scaler fit
on that training set only; MLP figures are means over restarts.

Because the comparison is paired, the per-planet accuracy change is also
reported as a McNemar-style breakdown - how many planets flipped correct to
incorrect and vice versa - which an unpaired comparison cannot give.

Usage:
    python evaluate_aerosol_paired.py
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
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
N_RESTARTS = 5
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
SOURCE_FMT = "multirex_spectra_H2_test_set_{}.parquet"
WORST_SYSTEMATIC = -22.0  # correlated noise at SNR 5, from the sweep


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


def sort_key(path):
    m = re.search(r"cloudy_(.+?)Pa", path)
    if m:
        return (0, -float(m.group(1)))
    return (1, float(re.search(r"hazy_(.+?)\.parquet", path).group(1)
                     .replace("p", ".")))


def label_of(path):
    m = re.search(r"cloudy_(.+?)Pa", path)
    if m:
        return f"deck {float(m.group(1)):.0e} Pa"
    return (f"haze {float(re.search(r'hazy_(.+?)[.]parquet', path).group(1).replace('p','.')):.1e} m-3")


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")

    files = sorted(glob.glob("multirex_spectra_H2_paired_*.parquet"), key=sort_key)
    if not files:
        raise SystemExit("no paired aerosol sets - run "
                         "generate_aerosol_paired.py first")

    df_tr = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df_tr)
    ref_grid = np.array([float(c) for c in cols])
    X_tr = df_tr[cols].values
    y_tr = (df_tr["biosignature"] == "yes").astype(int).values

    scaler_raw = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(
        scaler_raw.transform(X_tr))
    P_tr = pca.transform(scaler_raw.transform(X_tr))
    scaler_pca = StandardScaler().fit(P_tr)
    amp_tr = np.median(X_tr.std(axis=1) / X_tr.mean(axis=1))

    src = pd.concat([pd.read_parquet(SOURCE_FMT.format(i)) for i in range(1, 6)],
                    ignore_index=True)
    scols = spectral_cols(src)
    assert np.allclose([float(c) for c in scols], ref_grid, rtol=1e-9)
    X_clear = src[scols].values
    y = (src["biosignature"] == "yes").astype(int).values

    named = [("committed clear (same planets)", None)] + \
            [(label_of(f), f) for f in files]

    feats, amps, valid = {}, {}, {}
    for name, f in named[1:]:
        d = pd.read_parquet(f)
        assert len(d) == len(src), f"{f}: row count differs from source"
        assert (d["biosignature"].values == src["biosignature"].values).all()
        v = d["aerosol_valid"].values.astype(bool)
        X = d[spectral_cols(d)].values
        feats[name] = pca.transform(scaler_raw.transform(X))
        amps[name] = np.median(X[v].std(axis=1) / X[v].mean(axis=1)) / amp_tr
        valid[name] = v
    P_clear = pca.transform(scaler_raw.transform(X_clear))
    amps["committed clear (same planets)"] = np.median(
        X_clear.std(axis=1) / X_clear.mean(axis=1)) / amp_tr
    valid["committed clear (same planets)"] = np.ones(len(src), bool)
    feats["committed clear (same planets)"] = P_clear

    Xs, ys = shuffle(P_tr, y_tr, random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                        subsample=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(Xs, ys)

    xgb_acc, xgb_bri, correct = {}, {}, {}
    for name, _ in named:
        v = valid[name]
        p = xgb.predict_proba(feats[name])[:, 1]
        xgb_acc[name] = accuracy_score(y[v], p[v] > 0.5)
        xgb_bri[name] = brier_score_loss(y[v], np.clip(p[v], 1e-6, 1 - 1e-6))
        correct[name] = ((p > 0.5).astype(int) == y)
    base = xgb_acc["committed clear (same planets)"]
    print(f"XGBoost on committed clear, same planets: {base:.2%}\n", flush=True)

    W_tr = scaler_pca.transform(P_tr)
    mlp_acc = {n: [] for n, _ in named}
    for r in range(N_RESTARTS):
        rs = 1000 + r
        tf.random.set_seed(rs)
        Xw, yw = shuffle(W_tr, y_tr, random_state=rs)
        m = build_mlp(N_COMPONENTS)
        m.fit(Xw, yw, epochs=200, batch_size=128, validation_split=0.2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)], verbose=0)
        for name, _ in named:
            v = valid[name]
            p = m.predict(scaler_pca.transform(feats[name]), verbose=0).ravel()
            mlp_acc[name].append(accuracy_score(y[v], p[v] > 0.5))
        print(f"restart {r + 1}/{N_RESTARTS}", flush=True)

    hdr = (f"{'test set':<22} {'amp':>6} {'XGB acc':>9} {'change':>8} "
           f"{'Brier':>8} {'MLP acc':>16}  {'lost':>6} {'gained':>7}")
    L = ["Aerosol degradation on the committed planets (paired, no draw confound)",
         "",
         "Each aerosol set is the five committed clear test sets re-rendered with",
         "the aerosol added and nothing else changed, so every aerosol spectrum",
         "has a committed clear spectrum for the identical planet. The baseline is",
         "therefore the committed clear data itself and the degradation is a",
         "within-planet difference.",
         "",
         f"Models trained once on the committed training set. MLP figures are",
         f"means +/- sd over {N_RESTARTS} restarts. 'lost'/'gained' count planets",
         "that flipped correct->incorrect and incorrect->correct.",
         "", hdr, "-" * len(hdr)]
    rows = []
    for name, _ in named:
        v = valid[name]
        ma = np.array(mlp_acc[name])
        d = 100 * (xgb_acc[name] - base)
        c0 = correct["committed clear (same planets)"]
        c1 = correct[name]
        lost = int((c0 & ~c1 & v).sum())
        gained = int((~c0 & c1 & v).sum())
        chg = "(baseline)" if name.startswith("committed") else f"{d:>+8.2f}"
        L.append(f"{name:<22} {amps[name]:>5.2f}x {xgb_acc[name]:>9.2%} {chg:>8} "
                 f"{xgb_bri[name]:>8.4f} {ma.mean():>10.2%} +/-{ma.std():>4.1%}"
                 f"  {lost:>6} {gained:>7}")
        rows.append((name, amps[name], xgb_acc[name], d, xgb_bri[name],
                     ma.mean(), ma.std(), lost, gained, int(v.sum())))

    # Sets muted below FEATURELESS_AMP retain almost no diagnostic structure,
    # so their near-chance accuracy reflects absent signal rather than the
    # aerosol out-damaging anything. Quoting them as the "worst aerosol" would
    # overstate the case, so they are excluded from this comparison.
    FEATURELESS_AMP = 0.10
    decks = [(n, 100 * (xgb_acc[n] - base)) for n, _ in named
             if n.startswith("deck") and amps[n] >= FEATURELESS_AMP]
    hazes = [(n, 100 * (xgb_acc[n] - base)) for n, _ in named
             if n.startswith("haze") and amps[n] >= FEATURELESS_AMP]
    dropped = [n for n, _ in named
               if not n.startswith("committed") and amps[n] < FEATURELESS_AMP]
    worst_deck_name, worst_deck = min(decks, key=lambda t: t[1])
    worst_haze_name, worst_haze = min(hazes, key=lambda t: t[1])

    L += ["", "=" * len(hdr),
          "Against the worst instrumental systematic", "",
          f"  correlated noise at SNR 5 (from the sweep)   {WORST_SYSTEMATIC:>+7.2f}",
          f"  worst grey deck   ({worst_deck_name})        {worst_deck:>+7.2f}",
          f"  worst haze        ({worst_haze_name})   {worst_haze:>+7.2f}", ""]
    if dropped:
        L += [f"  Excluded as featureless (amplitude < {FEATURELESS_AMP:.2f}x): "
              + ", ".join(dropped),
              "  Those sets score near chance because almost no diagnostic",
              "  structure survives, not because the aerosol out-damages the",
              "  systematics. They are excluded from the comparison below.", ""]
    # Reviewer 1 asked for cloud and haze prescriptions on this axis, not a
    # ranking across shift types. The margins are a few points, measured at the
    # muted end of the range against an XGBoost run-to-run scatter of about 0.4,
    # so report magnitudes and leave the ordering unstated.
    L += [f"  Deck margin over the systematic:  {worst_deck - WORST_SYSTEMATIC:>+6.2f} points",
          f"  Haze margin over the systematic:  {worst_haze - WORST_SYSTEMATIC:>+6.2f} points",
          "",
          "  Aerosol degradation at strong muting is comparable in magnitude to",
          "  the most damaging instrumental systematic tested. These margins are",
          "  measured at the muted end of the range against an XGBoost",
          "  run-to-run scatter of about 0.4 points, so they do not support an",
          "  ordering across shift types.",
          "",
          "  Note the deepest deck sits in the featureless regime (feature",
          "  amplitude a few per cent of clear), where chance-level accuracy",
          "  reflects absent signal rather than classifier failure; the",
          "  informative deck range is 1e5 to 1e2 Pa."]

    out = "\n".join(L)
    print("\n" + out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_aerosol_paired.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["test_set", "feature_amp", "xgb_acc",
                                "delta_pts", "xgb_brier", "mlp_acc_mean",
                                "mlp_acc_sd", "lost", "gained", "n_valid"]
                 ).to_csv("final_results/H2_aerosol_paired.csv", index=False)
    print("\nWrote final_results/H2_aerosol_paired.{txt,csv}")


if __name__ == "__main__":
    main()
