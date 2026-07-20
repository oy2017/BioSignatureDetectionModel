"""
Evaluate the frozen pipeline on hazy test sets, against the grey-deck result.

Completes the haze half of axis 3 of Reviewer 1's R1-3 list. The comparison at
matched feature suppression is the scientific point, and it tests a prediction
made in advance (recorded in revision_plan.md before the haze experiment ran):
because the classifier ignores PC0 and PC1 - which capture mean transit depth
and continuum slope - a haze acting principally as a continuum tilt should
degrade performance considerably less than a grey deck at equal feature
suppression.

Same design invariants as evaluate_cloudy.py:

  * Models are trained once on the clean training set and never retrained.
  * The raw scaler, PCA basis and post-PCA scaler are fit on clean training
    data only and applied unchanged to every hazy set.
  * Calibration is reported alongside accuracy.
  * Feature amplitude is median per-spectrum scatter/mean relative to the
    clear training data - identical to the cloudy evaluation, so the deck and
    haze curves are directly comparable on the amplitude axis.

Run generate_hazy_testset.py first. Requires final_results/H2_cloudy_evaluation.csv
(from evaluate_cloudy.py) for the deck-vs-haze comparison.

Usage:
    python evaluate_hazy.py
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
CLEAN_TEST_FMT = "multirex_spectra_H2_test_set_{}.parquet"
CLOUDY_CSV = "final_results/H2_cloudy_evaluation.csv"
C_XGB, C_MLP = "#2a78d6", "#008300"
C_DECK = "#b0483a"
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#d8d8d4"


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def spectra_of(df, reference_grid):
    """Extract spectra by grid position, not by column name (see
    evaluate_cloudy.py for why)."""
    cols = spectral_cols(df)
    grid = np.array([float(c) for c in cols])
    assert len(grid) == len(reference_grid), (
        f"grid length {len(grid)} != reference {len(reference_grid)}")
    assert np.allclose(grid, reference_grid, rtol=1e-9), \
        "wavelength grids differ by more than floating-point noise"
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


def score(y, p):
    pred = (p > 0.5).astype(int)
    return (accuracy_score(y, pred), f1_score(y, pred, zero_division=0),
            brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR"); tf.random.set_seed(SEED)

    df_tr = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df_tr)
    ref_grid = np.array([float(c) for c in cols])
    X_tr = df_tr[cols].values
    y_tr = (df_tr["biosignature"] == "yes").astype(int).values

    # Frozen pipeline, fit on clean training data only.
    scaler_raw = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(scaler_raw.transform(X_tr))
    P_tr = pca.transform(scaler_raw.transform(X_tr))
    scaler_pca = StandardScaler().fit(P_tr)

    Xs, ys = shuffle(P_tr, y_tr, random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                        subsample=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(Xs, ys)
    Xw, yw = shuffle(scaler_pca.transform(P_tr), y_tr, random_state=SEED)
    mlp = build_mlp(N_COMPONENTS)
    mlp.fit(Xw, yw, epochs=200, batch_size=128, validation_split=0.2,
            callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                     restore_best_weights=True)], verbose=0)

    def evaluate(df):
        P = pca.transform(scaler_raw.transform(spectra_of(df, ref_grid)))
        y = (df["biosignature"] == "yes").astype(int).values
        px = xgb.predict_proba(P)[:, 1]
        pm = mlp.predict(scaler_pca.transform(P), verbose=0).ravel()
        return score(y, px), score(y, pm)

    rows = []

    # Clean baseline, averaged over the five original test sets.
    cx, cm = [], []
    for i in range(1, 6):
        a, b = evaluate(pd.read_parquet(CLEAN_TEST_FMT.format(i)))
        cx.append(a); cm.append(b)
    rows.append(("clear (no haze)", np.nan,
                 *np.mean(cx, axis=0), *np.mean(cm, axis=0), 1.0))

    # Hazy sets, ordered from thinnest (least effect) upward.
    files = sorted(glob.glob("multirex_spectra_H2_hazy_*.parquet"),
                   key=lambda f: float(
                       re.search(r"hazy_(.+?)\.parquet", f).group(1)
                       .replace("p", ".")))
    amp_ref = np.median(X_tr.std(axis=1) / X_tr.mean(axis=1))
    for f in files:
        mix = float(re.search(r"hazy_(.+?)\.parquet", f).group(1).replace("p", "."))
        d = pd.read_parquet(f)
        Xc = spectra_of(d, ref_grid)
        amp = np.median(Xc.std(axis=1) / Xc.mean(axis=1)) / amp_ref
        (ax, fx, bx), (am, fm, bm) = evaluate(d)
        rows.append((f"haze {mix:.1e} m^-3", mix, ax, fx, bx, am, fm, bm, amp))

    hdr = (f"{'test set':<22} {'feat amp':>9} {'XGB acc':>9} {'XGB Brier':>10} "
           f"{'MLP acc':>9} {'MLP Brier':>10}")
    lines = ["Frozen pipeline evaluated on hazy atmospheres (LeeMie, r=0.1um, Q0=40)",
             "",
             "The training set contains no aerosols. Feature amplitude is the median",
             "per-spectrum wavelength scatter relative to the clear training data,",
             "measured identically to the cloudy evaluation.", "",
             hdr, "-" * len(hdr)]
    for name, mix, ax, fx, bx, am, fm, bm, amp in rows:
        lines.append(f"{name:<22} {amp:>8.2f}x {ax:>9.2%} {bx:>10.4f} "
                     f"{am:>9.2%} {bm:>10.4f}")

    base_x = rows[0][2]
    lines += ["", "=" * 72,
              f"Degradation from the clear baseline (XGBoost {base_x:.2%}):", ""]
    for name, mix, ax, *_ in rows[1:]:
        lines.append(f"  {name:<22} {ax:>7.2%}  ({ax - base_x:+.2%})")

    # Deck-vs-haze at matched feature amplitude: interpolate the deck's
    # accuracy-vs-amplitude curve at each hazy set's measured amplitude.
    if os.path.exists(CLOUDY_CSV):
        deck = pd.read_csv(CLOUDY_CSV)
        deck = deck[deck["cloud_pa"].notna()].sort_values("feature_amp")
        lines += ["", "=" * 72,
                  "Deck vs haze at matched feature suppression (XGBoost):", "",
                  f"{'haze set':<22} {'amp':>6} {'haze acc':>9} "
                  f"{'deck acc @ same amp':>20} {'difference':>11}",
                  "-" * 72]
        for name, mix, ax, fx, bx, am, fm, bm, amp in rows[1:]:
            deck_acc = np.interp(amp, deck["feature_amp"], deck["xgb_acc"])
            lines.append(f"{name:<22} {amp:>5.2f}x {ax:>9.2%} "
                         f"{deck_acc:>20.2%} {ax - deck_acc:>+11.2%}")
        lines += ["",
                  "A positive difference means the haze harms less than a grey deck",
                  "that mutes the features by the same amount."]

    out = "\n".join(lines)
    print(out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_hazy_evaluation.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["test_set", "haze_mix", "xgb_acc", "xgb_f1",
                                "xgb_brier", "mlp_acc", "mlp_f1", "mlp_brier",
                                "feature_amp"]).to_csv(
        "final_results/H2_hazy_evaluation.csv", index=False)

    # Accuracy against feature amplitude: deck and haze on a common axis.
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    ax1.axhline(base_x, color=C_MUTED, linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.axhline(0.5, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.6)
    if os.path.exists(CLOUDY_CSV):
        deck = pd.read_csv(CLOUDY_CSV)
        deck = deck[deck["cloud_pa"].notna()].sort_values("feature_amp")
        ax1.plot(deck["feature_amp"], deck["xgb_acc"], color=C_DECK,
                 marker="o", linewidth=2, label="grey deck (XGBoost)")
    hz = [r for r in rows[1:]]
    ax1.plot([r[8] for r in hz], [r[2] for r in hz], color=C_XGB,
             marker="s", linewidth=2, label="LeeMie haze (XGBoost)")
    ax1.set_xlabel("feature amplitude relative to clear training data",
                   fontsize=10, color=C_TEXT)
    ax1.set_ylabel("Accuracy", fontsize=10, color=C_TEXT)
    ax1.set_title("Aerosol prescriptions at matched feature suppression\n"
                  "(dotted = clear baseline, dashed = chance)",
                  fontsize=11, color=C_TEXT)
    ax1.invert_xaxis()
    ax1.grid(True, color=C_GRID, linewidth=0.6); ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=9, colors=C_MUTED)
    for sp in ("top", "right"): ax1.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax1.spines[sp].set_color(C_GRID)
    ax1.legend(frameon=False, fontsize=9, labelcolor=C_TEXT)
    fig.tight_layout()
    fig.savefig("final_results/hazy_generalisation.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nWrote final_results/H2_hazy_evaluation.{txt,csv} "
          "and hazy_generalisation.png")


if __name__ == "__main__":
    main()
