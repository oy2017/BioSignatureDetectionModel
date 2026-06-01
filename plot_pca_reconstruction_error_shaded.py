import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re

# --- Configuration ---
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILE = f"multirex_spectra_{FILL_GAS}_test.parquet"
RESULTS_DIR = "final_results/plots"
PCA_START_IDX = 2
PCA_END_IDX = 102

# --- Load Data ---
print(f"--- Loading Data ---")
df_train = pd.read_parquet(TRAIN_FILE)
df_test = pd.read_parquet(TEST_FILE)

# Extract Spectral Columns and Wavelengths
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [c for c in df_train.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]
spectral_cols_sorted = sorted(spectral_cols, key=lambda x: float(x))
wavelengths = np.array([float(x) for x in spectral_cols_sorted])

X_train = df_train[spectral_cols_sorted].values
X_test = df_test[spectral_cols_sorted].values

# --- Preprocessing & PCA ---
print("--- Fitting Scaler and PCA on Training Data ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA()
pca.fit(X_train_scaled)

# --- Transform and Reconstruct Test Data ---
print("--- Reconstructing Test Spectra from PCA Components ---")
# 1. Scale the true test spectra
true_spectra_scaled = scaler.transform(X_test)

# 2. Transform into full PCA space
X_test_pca = pca.transform(true_spectra_scaled)

# 3. Create a version with only components 2-102
X_test_pca_subset = np.zeros_like(X_test_pca)
X_test_pca_subset[:, PCA_START_IDX:PCA_END_IDX] = X_test_pca[:, PCA_START_IDX:PCA_END_IDX]

# 4. Inverse transform from the subset to get the reconstruction
reconstructed_spectra_scaled = pca.inverse_transform(X_test_pca_subset)

# --- Calculate Error ---
# Using the specified formula: ((PCA - true) / true) - 1
epsilon = 1e-9
error_matrix = ((reconstructed_spectra_scaled - true_spectra_scaled) / (true_spectra_scaled + epsilon)) - 1

# --- Calculate Stats for Shading ---
mean_error = np.mean(error_matrix, axis=0)
std_error = np.std(error_matrix, axis=0)
upper_bound = mean_error + 2*std_error # 2x standard deviation
lower_bound = mean_error - 2*std_error

# --- Plotting ---
print("--- Generating Shaded Plot ---")
plt.figure(figsize=(14, 8))

# Shaded Region (Confidence Interval)
plt.fill_between(wavelengths, lower_bound, upper_bound, color='royalblue', alpha=0.15, label='Error Variance ($\pm 2\sigma$)')

# The Mean Error Line (Solid)
plt.plot(wavelengths, mean_error, color='black', linewidth=2.0, label='Mean Reconstruction Error', zorder=10)

# The Perfect Reconstruction line at -1
plt.axhline(-1, color='crimson', linestyle='-', linewidth=2.5, label='Perfect Reconstruction (y = -1)', zorder=5)

plt.title(f'PCA Reconstruction Error Dynamics (Components {PCA_START_IDX}–{PCA_END_IDX})', pad=20)
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel(r'Error $\left( \frac{X_{recon} - X_{true}}{X_{true}} - 1 \right)$')
plt.grid(True, linestyle='--', alpha=0.4)

# Focus on the relevant region
plt.ylim(-2.5, 0.5) 

plt.legend(frameon=True, loc='upper right', facecolor='white', framealpha=1.0)
plt.tight_layout()

# Save the shaded version
output_path = os.path.join(RESULTS_DIR, 'pca_reconstruction_errors_shaded.png')
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Shaded plot saved to: {output_path}")
