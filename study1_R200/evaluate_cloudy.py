"""
Evaluate the frozen pipeline on cloudy test sets.

Completes axis 3 of Reviewer 1's R1-3 list. The training data contains no
clouds, so this measures generalisation to atmospheres containing physics the
models never saw - a distribution shift produced by a different forward model
configuration rather than by perturbing existing spectra, which is a stronger
test than the injected-systematics sweep in domain_shift_sweep.py.

Same design invariants as that sweep:

  * Models are trained once on the clean training set and never retrained.
  * The raw scaler, PCA basis and post-PCA scaler are fit on clean training
    data only and applied unchanged to every cloudy set.
  * Calibration is reported alongside accuracy, since the manuscript's claim is
    calibrated triage.
  * XGBoost (unwhitened) and the MLP (whitened) are both evaluated, so the
    whitening robustness question from R1-8 is answered for this shift too.

Run generate_cloudy_testset.py first.

Usage:
    python evaluate_cloudy.py
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
C_XGB, C_MLP = "#2a78d6", "#008300"
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#d8d8d4"


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def spectra_of(df, reference_grid):
    """Extract spectra by grid position, not by column name.

    Column labels are stringified bin centres, and MultiREx's binner can differ
    in the last representable digit between runs - here one of 550 labels
    differed while the grids agreed to 4e-16. Selecting the training set's
    labels from a later frame therefore raises KeyError even though the grids
    are physically identical. Positional selection is correct because both
    frames use the same log-spaced grid; the assertion catches the case where
    they genuinely differ.
    """
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
    rows.append(("clear (no clouds)", np.nan,
                 *np.mean(cx, axis=0), *np.mean(cm, axis=0), 0))

    # Cloudy sets, ordered from deepest (least effect) to highest deck.
    files = sorted(glob.glob("multirex_spectra_H2_cloudy_*Pa.parquet"),
                   key=lambda f: -float(re.search(r"cloudy_(.+?)Pa", f).group(1)))
    for f in files:
        cp = float(re.search(r"cloudy_(.+?)Pa", f).group(1))
        d = pd.read_parquet(f)
        # Feature amplitude relative to the clear training data, as a direct
        # measure of how much the cloud deck has muted the spectra.
        # Scale-free: per-spectrum scatter divided by that spectrum's own mean
        # depth. Absolute scatter scales with transit depth and therefore with
        # radius squared, and radius spans 1-26 R_earth, so an absolute measure
        # is dominated by the radius draw rather than by cloud muting.
        Xc = spectra_of(d, ref_grid)
        amp = (np.median(Xc.std(axis=1) / Xc.mean(axis=1))
               / np.median(X_tr.std(axis=1) / X_tr.mean(axis=1)))
        (ax, fx, bx), (am, fm, bm) = evaluate(d)
        rows.append((f"cloud top {cp:.0e} Pa", cp, ax, fx, bx, am, fm, bm, amp))

    hdr = (f"{'test set':<22} {'feat amp':>9} {'XGB acc':>9} {'XGB Brier':>10} "
           f"{'MLP acc':>9} {'MLP Brier':>10}")
    lines = ["Frozen pipeline evaluated on cloudy atmospheres", "",
             "The training set contains no clouds. Feature amplitude is the mean",
             "per-spectrum wavelength scatter relative to the clear training data,",
             "so it shows directly how much each deck mutes the spectra.", "",
             hdr, "-" * len(hdr)]
    for name, cp, ax, fx, bx, am, fm, bm, amp in rows:
        lines.append(f"{name:<22} {amp:>8.2f}x {ax:>9.2%} {bx:>10.4f} "
                     f"{am:>9.2%} {bm:>10.4f}")

    base_x = rows[0][2]
    lines += ["", "=" * 72, "Degradation from the clear baseline "
              f"(XGBoost {base_x:.2%}):", ""]
    for name, cp, ax, *_ in rows[1:]:
        lines.append(f"  {name:<22} {ax:>7.2%}  ({ax - base_x:+.2%})")

    out = "\n".join(lines)
    print(out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_cloudy_evaluation.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["test_set", "cloud_pa", "xgb_acc", "xgb_f1",
                                "xgb_brier", "mlp_acc", "mlp_f1", "mlp_brier",
                                "feature_amp"]).to_csv(
        "final_results/H2_cloudy_evaluation.csv", index=False)

    # Accuracy against cloud-top pressure, clear baseline as a reference line.
    cl = [r for r in rows[1:]]
    if cl:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
        x = np.arange(len(cl))
        ax.axhline(base_x, color=C_XGB, linestyle=":", linewidth=1.5, alpha=0.7)
        ax.axhline(rows[0][5], color=C_MLP, linestyle=":", linewidth=1.5, alpha=0.7)
        ax.plot(x, [r[2] for r in cl], color=C_XGB, marker="o", linewidth=2,
                label="XGBoost (unwhitened)")
        ax.plot(x, [r[5] for r in cl], color=C_MLP, marker="s", linewidth=2,
                label="MLP (whitened)")
        ax.axhline(0.5, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r[1]:.0e}\n({r[8]:.2f}x)" for r in cl], fontsize=8)
        ax.set_xlabel("cloud-top pressure (Pa), and feature amplitude vs clear",
                      fontsize=10, color=C_TEXT)
        ax.set_ylabel("Accuracy", fontsize=10, color=C_TEXT)
        ax.set_title("Generalisation to cloudy atmospheres\n"
                     "(dotted = clear-sky baseline, dashed = chance)",
                     fontsize=11, color=C_TEXT)
        ax.grid(True, color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors=C_MUTED)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(C_GRID)
        ax.legend(frameon=False, fontsize=9, labelcolor=C_TEXT)
        fig.tight_layout()
        fig.savefig("final_results/cloudy_generalisation.png",
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("\nWrote final_results/H2_cloudy_evaluation.{txt,csv} "
              "and cloudy_generalisation.png")


if __name__ == "__main__":
    main()
