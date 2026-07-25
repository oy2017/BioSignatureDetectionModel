import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import corner
import os
import re

# --- Configuration ---
# Uses the headline pipeline of Section 4.1: all 102 components and the tuned
# XGBoost configuration, evaluated on the five test sets pooled, so this figure
# depicts the same model and the same planets as the error rates quoted in
# Section 4.3 (see analyze_error_quintiles.py).
FILL_GAS = "H2"
SEED = 42
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILES = [f"multirex_spectra_{FILL_GAS}_test_set_{i}.parquet" for i in range(1, 6)]
RESULTS_DIR = "final_results/plots"
PARAMS_TO_PLOT = ['p_radius', 'p_mass', 's temperature', 'atm temperature']
LABELS = ['Planet Radius ($R_{\oplus}$)', 'Planet Mass ($M_{\oplus}$)', 'Star Temp (K)', 'Atmosphere Temp (K)']

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Load and Prepare Data ---
print("--- Loading and Preparing Data ---")
df_train = pd.read_parquet(TRAIN_FILE)
df_test = pd.concat([pd.read_parquet(f) for f in TEST_FILES], ignore_index=True)

df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$")
cols = [c for c in df_train.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]

# Drop physically impossible transit depths, as in the main pipeline.
df_train = df_train[(df_train[cols].values <= 1.0).all(axis=1)].reset_index(drop=True)
df_test = df_test[(df_test[cols].values <= 1.0).all(axis=1)].reset_index(drop=True)
print(f"    train {len(df_train)}   test {len(df_test)}")

X_train_raw = df_train[cols].values
y_train = df_train['label'].values
X_test_raw = df_test[cols].values
y_test = df_test['label'].values

# --- Preprocessing (PCA) ---
print("--- Scaling and PCA ---")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s = scaler.transform(X_test_raw)

pca = PCA(n_components=102, random_state=SEED)
X_train_clean = pca.fit_transform(X_train_s)
X_test_clean = pca.transform(X_test_s)

# --- Train XGBoost ---
print("--- Training XGBoost ---")
X_train_clean, y_train = shuffle(X_train_clean, y_train, random_state=SEED)
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                      random_state=SEED, n_jobs=-1, eval_metric='logloss')
model.fit(X_train_clean, y_train)
y_pred = model.predict(X_test_clean)
print(f"    pooled test accuracy {100*(y_pred == y_test).mean():.2f}%")

# --- Identify Errors ---
correct_mask = (y_pred == y_test)
incorrect_mask = ~correct_mask

params_correct = df_test[correct_mask][PARAMS_TO_PLOT].values
params_incorrect = df_test[incorrect_mask][PARAMS_TO_PLOT].values

# --- Generate Plot ---
print("--- Generating Scatter Corner Plot ---")
# Make figure slightly wider to accommodate external legend
figure = plt.figure(figsize=(16, 15))

# Plot Incorrect points (crimson)
corner.corner(params_incorrect, fig=figure, labels=LABELS,
              plot_contours=False, plot_density=False, plot_datapoints=True,
              data_kwargs={'color': 'crimson', 'alpha': 0.8, 'ms': 4})

# Plot Correct points (navy)
corner.corner(params_correct, fig=figure, labels=LABELS,
              plot_contours=False, plot_density=False, plot_datapoints=True,
              data_kwargs={'color': 'navy', 'alpha': 0.2, 'ms': 3})

# --- Fix Legend (Move to Top Right, Outside Subplots) ---
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Correct Prediction', markerfacecolor='navy', markersize=12, alpha=0.5),
    Line2D([0], [0], marker='o', color='w', label='Classification Error', markerfacecolor='crimson', markersize=12, alpha=0.9)
]

# Place legend in the upper right empty space of the corner plot grid
# The coordinates (x, y) are relative to the entire figure size (0 to 1)
figure.legend(handles=legend_elements, loc='upper right', fontsize=16, 
              bbox_to_anchor=(0.95, 0.95), frameon=True, facecolor='white', 
              edgecolor='black', framealpha=1.0)

# Remove suptitle as requested by reviewer earlier
# plt.suptitle("Scatter Corner Plot of Prediction Errors (XGBoost)", fontsize=20)

filename = os.path.join(RESULTS_DIR, 'corner_plot_errors_scatter_xgboost_fixed.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to: {filename}")
