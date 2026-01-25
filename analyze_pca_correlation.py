import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import re

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Analyze correlation between PCA components and the target label.")
parser.add_argument("fill_gas", type=str, help="The fill gas for the atmosphere (e.g., H2, N2).")
parser.add_argument("--n_components", type=int, default=105, help="Number of components to analyze.")
args = parser.parse_args()
fill_gas = args.fill_gas.upper()
n_components = args.n_components

# --- 1. Load Training Data ---
print(f"--- Loading {fill_gas} Training Data for Correlation Analysis ---")
try:
    df = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
except FileNotFoundError:
    print(f"Error: Training data file not found for {fill_gas}. Please generate it first.")
    exit()

# --- 2. Prepare Data ---
# Identify spectral columns
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X = df[spectral_cols]

# Identify target
y = df['biosignature']
le = LabelEncoder()
y_encoded = le.fit_transform(y) # yes=1, no=0 usually, but we will check mapping
print(f"Classes: {le.classes_}") 
# 'no' -> 0, 'yes' -> 1

# --- 3. Scale Data ---
print("--- Scaling Data ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Perform PCA ---
print(f"--- Fitting PCA with {n_components} components ---")
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# --- 5. Calculate Correlation ---
print("--- Calculating Correlations ---")
correlations = []
for i in range(n_components):
    # Calculate Pearson correlation coefficient
    # np.corrcoef returns a matrix, we want [0, 1]
    corr = np.corrcoef(X_pca[:, i], y_encoded)[0, 1]
    correlations.append(corr)

# --- 6. Plotting ---
plt.figure(figsize=(15, 6))
indices = np.arange(n_components)
colors = ['red' if i < 2 else 'blue' for i in indices]

plt.bar(indices, correlations, color=colors, alpha=0.7)
plt.title(f'Correlation between Principal Components and Biosignature Label ({fill_gas})')
plt.xlabel('Principal Component Index')
plt.ylabel('Pearson Correlation Coefficient')
plt.axhline(0, color='black', linewidth=0.8)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight specific regions
plt.axvspan(-0.5, 1.5, color='red', alpha=0.1, label='High Variance / Low Signal (PC 0-1)')
plt.legend()

plot_filename = f'{fill_gas}_pca_correlation.png'
plt.savefig(plot_filename)
print(f"\nCorrelation plot saved to {plot_filename}")

# --- 7. Print Top Correlations ---
# Sort by absolute correlation
abs_correlations = np.abs(correlations)
sorted_indices = np.argsort(abs_correlations)[::-1]

print("\n--- Top 10 Components by Absolute Correlation ---")
for i in sorted_indices[:10]:
    print(f"PC-{i}: {correlations[i]:.4f}")

print("\n--- Correlation of First 5 Components ---")
for i in range(5):
    print(f"PC-{i}: {correlations[i]:.4f}")
