import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
parser = argparse.ArgumentParser(description="Evaluate a dataset with a specified fill gas.")
parser.add_argument("fill_gas", type=str, help="The fill gas for the atmosphere (e.g., H2, N2).")
args = parser.parse_args()

fill_gas = args.fill_gas.upper() # Convert to uppercase for consistency

# --- 1. Load and Prepare Data ---
print(f"--- Loading and Preparing {fill_gas} Dataset ---")
df = pd.read_parquet(f'multirex_spectra_{fill_gas}.parquet')
df['label'] = df['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]

X = df[spectral_cols]
y = df['label']

# --- 2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Build and Fit Pipeline ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X_train_scaled)

# --- 4. Train Classifier ---
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train_pca, y_train)

# --- 5. Apply Pipeline to Test Data ---
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# --- 6. Make Predictions ---
y_pred = model.predict(X_test_pca)

# --- 7. Save and Display Results ---
report_str = classification_report(y_test, y_pred, target_names=['Non-Bio (0)', 'Bio (1)'])
with open(os.path.join(results_path, f'{fill_gas}_classification_report.txt'), 'w') as f:
    f.write(report_str)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix on {fill_gas} Dataset')
plt.savefig(os.path.join(results_path, f'{fill_gas}_confusion_matrix.png'))
plt.close()

print(f"\n--- {fill_gas} Analysis Complete ---")
print(report_str)
