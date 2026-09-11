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
    'text.usetex': False,
})

FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILE = f"multirex_spectra_{FILL_GAS}_test.parquet"
RESULTS_DIR = "final_results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- 1. Load Data ---
df_train = pd.read_parquet(TRAIN_FILE)
df_test = pd.read_parquet(TEST_FILE)

# Extract Spectral Columns
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = sorted([c for c in df_train.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))], key=float)
wavelengths = np.array([float(x) for x in spectral_cols])

X_train = df_train[spectral_cols].values
X_test = df_test[spectral_cols].values

# --- 2. PCA Fitting ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA().fit(X_train_scaled)

# --- 3. Test Set Reconstruction (0-101) ---
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Reconstruct WITH components 0 and 1
subset = np.zeros_like(X_test_pca)
subset[:, 0:102] = X_test_pca[:, 0:102]
recon_scaled = pca.inverse_transform(subset)
recon = scaler.inverse_transform(recon_scaled)

# Reviewer's formula
error = ((recon - X_test) / (X_test + 1e-9)) - 1

mean_error = np.mean(error, axis=0)
std_error = np.std(error, axis=0)

# --- 4. Plotting ---
plt.figure(figsize=(12, 6))

# Shaded variance
plt.fill_between(wavelengths, mean_error - std_error, mean_error + std_error, color='mediumseagreen', alpha=0.3, label=r'Error Variance ($\pm 1\sigma$)')

# Prominent Mean Line
plt.plot(wavelengths, mean_error, color='darkgreen', linewidth=2.5, label='Mean Error (PCs 0–101)')

# Reference line at -1
plt.axhline(-1, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Perfect Reconstruction')

plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel(r'Error $\left( \frac{X_{recon} - X_{true}}{X_{true}} \right) - 1$')

plt.grid(True, linestyle='-', alpha=0.3)
plt.legend(loc='upper right')

# Let matplotlib auto-scale to show the actual wiggles
plt.margins(x=0.01)
plt.tight_layout()

output_path = os.path.join(RESULTS_DIR, 'H2_pca_test_set_reconstruction_final.png')
plt.savefig(output_path, dpi=300)
print(f"Final plot saved to: {output_path}")
