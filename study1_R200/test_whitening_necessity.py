"""
Is the post-PCA whitening step still necessary once the uninformative
high-variance components are dropped?

The manuscript applies a StandardScaler to the retained principal components so
that the neural networks are not dominated by PC0/PC1, which carry 98.41% of the
variance. The component ablation (ablate_pc_ranges.py) and the per-component
analysis (analyze_pc_discriminative_power.py) both show those components carry
essentially no label information. If they are simply discarded instead of
rescaled, the variance imbalance largely disappears - and the whitening step may
become unnecessary.

That matters for the revision: a pipeline with no whitening step cannot be
criticised for what whitening does or does not amplify.

This script crosses {component range} x {whitening on/off} for the MLP and
1D-CNN, and includes XGBoost as an invariance control - tree splits are
unaffected by monotonic per-feature rescaling, so its two columns should match
to within run-to-run noise. That demonstrates the invariance the manuscript
currently only asserts.

Architectures and hyperparameters replicate run_master_5set_evaluation.py.

Usage:
    python test_whitening_necessity.py            # MLP + CNN + XGBoost control
    python test_whitening_necessity.py --quick    # MLP only, 1 repeat
"""

import argparse
import os
import random
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

tf.get_logger().setLevel("ERROR")

SEED = 42
N_COMPONENTS = 102
REPEATS = 3          # neural training is stochastic; average over restarts
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
TEST_FILE_FMT = "multirex_spectra_H2_test_set_{}.parquet"

# Ranges chosen from the ablation: the full space, and progressively dropping
# the high-variance components that carry no label information.
RANGES = [(0, 102), (2, 102), (5, 102), (10, 102)]


def load_pca_space():
    """Fit scaler + PCA on training data only. Returns unwhitened PC scores."""
    df_train = pd.read_parquet(TRAIN_FILE)
    float_pattern = re.compile(r"^-?\d+\.\d+$")
    spectral_cols = [
        c for c in df_train.columns
        if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))
    ]

    y_train = (df_train["biosignature"] == "yes").astype(int).values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[spectral_cols].values)

    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)

    tests = []
    for i in range(1, 6):
        df_test = pd.read_parquet(TEST_FILE_FMT.format(i))
        X_test_pca = pca.transform(scaler.transform(df_test[spectral_cols].values))
        y_test = (df_test["biosignature"] == "yes").astype(int).values
        tests.append((X_test_pca, y_test))

    return X_train_pca, y_train, tests


def prepare(X_train_pca, tests, start, end, whiten):
    """Slice the component range, optionally applying the post-PCA scaler."""
    X_train = X_train_pca[:, start:end]
    X_tests = [(Xp[:, start:end], y) for Xp, y in tests]

    if whiten:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_tests = [(scaler.transform(Xt), y) for Xt, y in X_tests]

    return X_train, X_tests


def build_mlp(input_dim):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in [256, 128, 64]:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_dim):
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(32, kernel_size=5, padding="same"), BatchNormalization(),
        Activation("relu"), MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(64, kernel_size=5, padding="same"), BatchNormalization(),
        Activation("relu"), MaxPooling1D(pool_size=2), Dropout(0.3),
        Flatten(),
        Dense(100), BatchNormalization(), Activation("relu"), Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_condition(model_name, X_train, y_train, X_tests, seed):
    """Train one model and return (accuracies, f1s) across the five test sets."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    n_features = X_train.shape[1]
    X_tr, y_tr = shuffle(X_train, y_train, random_state=seed)

    if model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
            eval_metric="logloss", random_state=SEED, n_jobs=-1,
        ).fit(X_tr, y_tr)
        predict = lambda X: model.predict(X)

    elif model_name == "MLP":
        model = build_mlp(n_features)
        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=200, batch_size=128, validation_split=0.2,
                  callbacks=[es], verbose=0)
        predict = lambda X: (model.predict(X, verbose=0) > 0.5).astype(int).ravel()

    else:  # CNN
        if n_features < 8:
            return None, None
        model = build_cnn(n_features)
        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_tr.reshape(-1, n_features, 1), y_tr, epochs=100, batch_size=128,
                  validation_split=0.2, callbacks=[es], verbose=0)
        predict = lambda X: (
            model.predict(X.reshape(-1, n_features, 1), verbose=0) > 0.5
        ).astype(int).ravel()

    accs, f1s = [], []
    for X_test, y_test in X_tests:
        preds = predict(X_test)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, zero_division=0))
    return accs, f1s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="MLP only, single repeat")
    args = parser.parse_args()

    models = ["MLP"] if args.quick else ["MLP", "CNN", "XGBoost"]
    repeats = 1 if args.quick else REPEATS

    os.makedirs("final_results", exist_ok=True)
    X_train_pca, y_train, tests = load_pca_space()

    header = (
        f"{'Model':<9} {'PC range':>11} {'n':>4} {'whitened':>9} "
        f"{'accuracy':>17} {'F1':>17}"
    )
    lines = [
        "Whitening necessity across principal-component ranges",
        f"Repeats per condition: {repeats}   Test sets: 5",
        "XGBoost is an invariance control: its whitened and unwhitened rows",
        "should agree, since tree splits are unaffected by per-feature rescaling.",
        "",
        header,
        "-" * len(header),
    ]
    print("\n".join(lines))

    results = {}
    for model_name in models:
        for start, end in RANGES:
            for whiten in (True, False):
                X_train, X_tests = prepare(X_train_pca, tests, start, end, whiten)

                all_acc, all_f1 = [], []
                for rep in range(repeats):
                    accs, f1s = run_condition(
                        model_name, X_train, y_train, X_tests, SEED + rep
                    )
                    if accs is None:
                        continue
                    all_acc.extend(accs)
                    all_f1.extend(f1s)

                if not all_acc:
                    continue

                results[(model_name, start, end, whiten)] = float(np.mean(all_acc))
                row = (
                    f"{model_name:<9} {f'[{start}:{end})':>11} {end - start:>4}"
                    f" {str(whiten):>9}"
                    f" {np.mean(all_acc):>10.2%} ±{np.std(all_acc):>5.2%}"
                    f" {np.mean(all_f1):>10.2%} ±{np.std(all_f1):>5.2%}"
                )
                print(row)
                lines.append(row)

    # The comparison the revision hinges on.
    baseline = results.get(("MLP", 0, 102, True))
    if baseline is not None:
        lines.append("")
        lines.append("Key comparisons (MLP):")
        lines.append(f"  whitened, all components  [0:102): {baseline:.2%}")
        for start, end in RANGES[1:]:
            unwhitened = results.get(("MLP", start, end, False))
            if unwhitened is not None:
                delta = unwhitened - baseline
                lines.append(
                    f"  UNwhitened, [{start}:{end}):".ljust(38)
                    + f"{unwhitened:.2%}  ({delta:+.2%} vs whitened baseline)"
                )
        lines.append("")
        lines.append(
            "If an unwhitened row matches the whitened baseline, the whitening "
            "step can be dropped from the pipeline entirely."
        )

    if baseline is not None:
        print("\n" + "\n".join(lines[lines.index("Key comparisons (MLP):"):]))

    out = "final_results/H2_whitening_necessity.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
