import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import re

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Analyze PCA explained variance for a given dataset.")
parser.add_argument("fill_gas", type=str, help="The fill gas for the atmosphere (e.g., H2, N2).")
args = parser.parse_args()
fill_gas = args.fill_gas.upper()

# --- 1. Load Training Data ---
print(f"--- Loading {fill_gas} Training Data for PCA Variance Analysis ---")
try:
    df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
except FileNotFoundError:
    print(f"Error: Training data file not found for {fill_gas}. Please generate it first.")
    exit()

# --- 2. Prepare Data ---
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X_train = df_train[spectral_cols]

# --- 3. Scale Data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 4. Perform PCA ---
print("--- Fitting PCA to determine explained variance ---")
pca = PCA()
pca.fit(X_train_scaled)

# --- 5. Analyze and Report Variance ---
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print("\n--- PCA Explained Variance Analysis ---")
print(f"Number of components to explain 95% of variance: {n_components_95}")

print("\nExplained variance for top components:")
for i, variance in enumerate(explained_variance[:50]):
    print(f"  PC-{i+1}: {variance:.4f} (Cumulative: {cumulative_variance[i]:.4f})")

# --- 6. Generate Cumulative Explained Variance Plot ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--')
plt.title(f'PCA Cumulative Explained Variance for {fill_gas} Dataset')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained Variance')
plt.axvline(x=n_components_95, color='g', linestyle='-', label=f'{n_components_95} Components for 95%')
plt.legend(loc='best')

plot_filename = f'{fill_gas}_pca_explained_variance.png'
plt.savefig(plot_filename)
print(f"\nCumulative explained variance plot saved to {plot_filename}")

# --- 7. Generate Traditional Scree Plot ---
plt.figure(figsize=(10, 6))
# Limiting to the first 50 components for better visualization
num_components_to_plot = 50
plt.plot(range(1, num_components_to_plot + 1), explained_variance[:num_components_to_plot], marker='o', linestyle='-')
plt.title(f'Scree Plot for {fill_gas} Dataset (First {num_components_to_plot} Components)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(0, num_components_to_plot + 1, 5))
plt.grid(True)

scree_plot_filename = f'{fill_gas}_pca_scree_plot.png'
plt.savefig(scree_plot_filename)
print(f"Scree plot saved to {scree_plot_filename}")
