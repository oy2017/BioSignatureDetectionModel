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
scaler_raw = StandardScaler()
X_train_s = scaler_raw.fit_transform(X_train_raw)
X_test_s = scaler_raw.transform(X_test_raw)

pca = PCA(n_components=102, random_state=42)
X_train_p_full = pca.fit_transform(X_train_s)
X_test_p_full = pca.transform(X_test_s)

X_train_final = X_train_p_full[:, PCA_START:PCA_END]
X_test_final = X_test_p_full[:, PCA_START:PCA_END]

# --- Train Best Model (XGBoost) ---
print("--- Training Final XGBoost ---")
model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
model.fit(X_train_final, y_train)

# --- Make Predictions and Identify Errors ---
print("--- Identifying Correct vs Incorrect Predictions ---")
y_pred = model.predict(X_test_final)
correct_mask = (y_pred == y_test)
incorrect_mask = ~correct_mask

def generate_corner(mask_correct, mask_incorrect, params, labels, filename):
    params_correct = df_test[mask_correct][params].values
    params_incorrect = df_test[mask_incorrect][params].values
    
    print(f"--- Generating Corner Plot: {filename} ---")
    figure = plt.figure(figsize=(12, 12))
    
    # Plot Correct Predictions (Background Contours, NO DOTS)
    corner.corner(params_correct, fig=figure, labels=labels, color='navy', 
                  plot_contours=True, plot_datapoints=False, plot_density=True, smooth=1.0, 
                  hist_kwargs={'density': True, 'color': 'navy'})
                  
    # Plot Incorrect Predictions (Foreground Contours AND DOTS)
    corner.corner(params_incorrect, fig=figure, labels=labels, color='crimson', 
                  plot_contours=True, plot_datapoints=True, plot_density=False, smooth=1.0, 
                  data_kwargs={'alpha': 0.9, 'ms': 4.0}, # Make dots highly visible
                  hist_kwargs={'density': True, 'color': 'crimson'})
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='navy', lw=4, label='Correct'), Line2D([0], [0], color='crimson', lw=4, label='Incorrect')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 1. Physical Parameters
generate_corner(correct_mask, incorrect_mask, 
                ['p_radius', 'p_mass', 's temperature', 'atm temperature'],
                ['Radius (R_earth)', 'Mass (M_earth)', 'Star Temp (K)', 'Atm Temp (K)'],
                'corner_plot_physical.png')

# 2. Chemical Parameters
generate_corner(correct_mask, incorrect_mask,
                ['atm CH4', 'atm O3'],
                ['log(CH4)', 'log(O3)'],
                'corner_plot_chemical.png')

print(f"Plots saved to: {RESULTS_DIR}")
