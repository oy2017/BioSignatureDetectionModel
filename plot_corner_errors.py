import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import corner
import os
import re

# --- Configuration ---
FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILE = f"multirex_spectra_{FILL_GAS}_test.parquet"
RESULTS_DIR = "final_results/plots"
PCA_START = 2
PCA_END = 102
PARAMS_TO_PLOT = ['p_radius', 'p_mass', 's temperature', 'atm temperature']
LABELS = ['Planet Radius (R_earth)', 'Planet Mass (M_earth)', 'Star Temp (K)', 'Atmosphere Temp (K)']

# --- Load and Prepare Data ---
print("--- Loading and Preparing Data ---")
df_train = pd.read_parquet(TRAIN_FILE)
df_test = pd.read_parquet(TEST_FILE)

df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$")
cols = [c for c in df_train.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]

X_train_raw = df_train[cols].values
y_train = df_train['label'].values
X_test_raw = df_test[cols].values
y_test = df_test['label'].values

# --- Preprocessing (PCA) ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s = scaler.transform(X_test_raw)

pca = PCA()
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

X_train_final = X_train_p[:, PCA_START:PCA_END]
X_test_final = X_test_p[:, PCA_START:PCA_END]

# --- Train Best Model (XGBoost) ---
print("--- Training Best Model (XGBoost) ---")
model = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1, eval_metric='logloss')
model.fit(X_train_final, y_train)

# --- Make Predictions and Identify Errors ---
print("--- Identifying Correct vs Incorrect Predictions ---")
y_pred = model.predict(X_test_final)
correct_mask = (y_pred == y_test)
incorrect_mask = ~correct_mask

# Extract physical parameters
params_correct = df_test[correct_mask][PARAMS_TO_PLOT].values
params_incorrect = df_test[incorrect_mask][PARAMS_TO_PLOT].values

# --- Generate Corner Plot ---
print("--- Generating Corner Plot ---")
figure = plt.figure(figsize=(15, 15))

# Plot Correct Predictions (Blue)
corner.corner(params_correct, fig=figure, labels=LABELS,
              color='navy', plot_contours=True, smooth=1.0, 
              hist_kwargs={'density': True, 'color': 'navy'})

# Plot Incorrect Predictions (Red)
corner.corner(params_incorrect, fig=figure, labels=LABELS,
              color='crimson', plot_contours=True, smooth=1.0,
              hist_kwargs={'density': True, 'color': 'crimson'})

# Custom Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='navy', lw=4, label='Correct Predictions'),
                   Line2D([0], [0], color='crimson', lw=4, label='Incorrect Predictions')]
plt.legend(handles=legend_elements, loc='upper right', fontsize=14)

plt.suptitle("Corner Plot of Physical Parameters vs. Prediction Errors (XGBoost)", fontsize=20)
plt.savefig(os.path.join(RESULTS_DIR, 'corner_plot_errors.png'), dpi=300)
plt.close()

print(f"Plot saved to: {os.path.join(RESULTS_DIR, 'corner_plot_errors.png')}")
