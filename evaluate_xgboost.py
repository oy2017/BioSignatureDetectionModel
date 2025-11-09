import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import argparse

# --- Create Results Directory ---
results_path = 'final_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# --- Add argument parser ---
parser = argparse.ArgumentParser(description="Evaluate a dataset with a specified fill gas using XGBoost.")
parser.add_argument("fill_gas", type=str, help="The fill gas for the atmosphere (e.g., H2, N2).")
parser.add_argument("--pca_start_idx", type=int, default=0, help="0-based index of the first principal component to include.")
parser.add_argument("--pca_end_idx", type=int, default=None, help="0-based index of the last principal component to include (exclusive). If None, uses all components from start_idx.")
parser.add_argument("--n_estimators", type=int, default=150, help="The number of boosting rounds.")
parser.add_argument("--max_depth", type=int, default=5, help="The maximum depth of a tree.")
parser.add_argument("--learning_rate", type=float, default=0.1, help="The learning rate.")
args = parser.parse_args()

fill_gas = args.fill_gas.upper()
pca_start_idx = args.pca_start_idx
pca_end_idx = args.pca_end_idx
n_estimators = args.n_estimators
max_depth = args.max_depth
learning_rate = args.learning_rate

# --- 1. Load and Prepare Data ---
print(f"--- Loading and Preparing {fill_gas} Dataset for XGBoost ---")
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]

X_train = df_train[spectral_cols]
y_train = df_train['label']
X_test = df_test[spectral_cols]
y_test = df_test['label']

# --- 2. Build and Fit Pipeline ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Apply PCA (all components first, then slice) ---
print(f"--- Applying PCA from component {pca_start_idx} to {pca_end_idx if pca_end_idx is not None else 'end'} ---")
pca_full = PCA() # Compute all components
X_train_pca_full = pca_full.fit_transform(X_train_scaled)
X_test_pca_full = pca_full.transform(X_test_scaled)

X_train_processed = X_train_pca_full[:, pca_start_idx:pca_end_idx]
X_test_processed = X_test_pca_full[:, pca_start_idx:pca_end_idx]

# --- 3. Train Classifier ---
print(f"--- Training XGBoost with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate} ---")
model = XGBClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_processed, y_train)

# --- 4. Make Predictions ---
y_pred = model.predict(X_test_processed)

# --- 5. Save and Display Results ---
pca_suffix = f'_pca_idx_{pca_start_idx}_{pca_end_idx if pca_end_idx is not None else "end"}'
model_suffix = f'_est_{n_estimators}_depth_{max_depth}_lr_{learning_rate}'
report_filename = os.path.join(results_path, f'{fill_gas}_xgboost{pca_suffix}{model_suffix}_report.txt')
cm_filename = os.path.join(results_path, f'{fill_gas}_xgboost{pca_suffix}{model_suffix}_confusion_matrix.png')
cm_title = f'Confusion Matrix on {fill_gas} (XGBoost, PCA {pca_start_idx}-{pca_end_idx if pca_end_idx is not None else "end"}, Est={n_estimators}, Depth={max_depth}, LR={learning_rate})'

report_str = classification_report(y_test, y_pred, target_names=['Non-Bio (0)', 'Bio (1)'])
with open(report_filename, 'w') as f:
    f.write(report_str)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(cm_title)
plt.savefig(cm_filename)
plt.close()

print(f"\n--- {fill_gas} Analysis Complete (XGBoost, PCA {pca_start_idx}-{pca_end_idx if pca_end_idx is not None else 'end'}, Est={n_estimators}, Depth={max_depth}, LR={learning_rate}) ---")
print(report_str)
