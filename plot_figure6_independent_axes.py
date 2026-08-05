"""Figure 6 re-drawn over the sampled directions that are actually independent.

The delivered grid couples planet radius, stellar temperature, atmospheric
temperature and semi-major axis into a single sampled direction, and planet
mass with stellar radius into a second (Section 3.1). Plotting four members of
those two groups, as the previous version did, spends three of six panels on
degenerate diagonals that carry no information.

This version keeps one representative of the coupled group (planet radius),
the independent mass direction, and the measured per-spectrum feature
amplitude -- the quantity Section 4.3 identifies as the operative mechanism and
the one axis the sampling coupling cannot touch. All three panels are
informative.

Pipeline is identical to fix_xgboost_corner.py, so the accuracy and the plotted
planets match the error rates quoted in Section 4.3.
"""
import os
import re

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

FILL_GAS = "H2"
SEED = 42
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILES = [f"multirex_spectra_{FILL_GAS}_test_set_{i}.parquet" for i in range(1, 6)]
RESULTS_DIR = "final_results/plots"
OUT = os.path.join(RESULTS_DIR, "figure6_independent_axes.png")

LABELS = [r"Planet Radius ($R_{\oplus}$)",
          r"Planet Mass ($M_{\oplus}$)",
          r"$\log_{10}$ Feature Amplitude"]

os.makedirs(RESULTS_DIR, exist_ok=True)

print("--- Loading and Preparing Data ---")
df_train = pd.read_parquet(TRAIN_FILE)
df_test = pd.concat([pd.read_parquet(f) for f in TEST_FILES], ignore_index=True)
for d in (df_train, df_test):
    d["label"] = d["biosignature"].apply(lambda x: 1 if x == "yes" else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$")
cols = [c for c in df_train.columns
        if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]

# Drop physically impossible transit depths, as in the main pipeline.
df_train = df_train[(df_train[cols].values <= 1.0).all(axis=1)].reset_index(drop=True)
df_test = df_test[(df_test[cols].values <= 1.0).all(axis=1)].reset_index(drop=True)
print(f"    train {len(df_train)}   test {len(df_test)}")

X_train_raw = df_train[cols].values
X_test_raw = df_test[cols].values
y_train = df_train["label"].values
y_test = df_test["label"].values

print("--- Scaling and PCA ---")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s = scaler.transform(X_test_raw)
pca = PCA(n_components=102, random_state=SEED)
X_train_clean = pca.fit_transform(X_train_s)
X_test_clean = pca.transform(X_test_s)

print("--- Training XGBoost ---")
X_train_clean, y_train = shuffle(X_train_clean, y_train, random_state=SEED)
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                      random_state=SEED, n_jobs=-1, eval_metric="logloss")
model.fit(X_train_clean, y_train)
y_pred = model.predict(X_test_clean)
acc = 100 * (y_pred == y_test).mean()
print(f"    pooled test accuracy {acc:.2f}%")

# Feature amplitude: per-spectrum scatter across wavelength, the same quantity
# used in analyze_error_quintiles.py.
amplitude = X_test_raw.std(axis=1)

# Amplitude spans two decades with a long tail, so it is plotted on a log axis;
# on a linear axis the low-amplitude tail that carries the error excess is
# compressed against the origin and invisible.
P = np.column_stack([df_test["p_radius"].astype(float).values,
                     df_test["p_mass"].astype(float).values,
                     np.log10(amplitude)])

correct = y_pred == y_test
print(f"    correct {correct.sum()}   errors {(~correct).sum()}")

# Shared ranges so the two overlaid corner calls align.
rng = [(P[:, i].min(), P[:, i].max()) for i in range(P.shape[1])]

print("--- Generating Figure ---")
figure = plt.figure(figsize=(12, 11))
kw = dict(fig=figure, labels=LABELS, range=rng, plot_contours=False,
          plot_density=False, plot_datapoints=True, label_kwargs={"fontsize": 15})
corner.corner(P[~correct], data_kwargs={"color": "crimson", "alpha": 0.8, "ms": 4}, **kw)
corner.corner(P[correct], data_kwargs={"color": "navy", "alpha": 0.2, "ms": 3}, **kw)

figure.legend(
    handles=[Line2D([0], [0], marker="o", color="w", label="Correct Prediction",
                    markerfacecolor="navy", markersize=12, alpha=0.5),
             Line2D([0], [0], marker="o", color="w", label="Classification Error",
                    markerfacecolor="crimson", markersize=12, alpha=0.9)],
    loc="upper right", fontsize=15, bbox_to_anchor=(0.97, 0.97),
    frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)

plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved to: {OUT}")
