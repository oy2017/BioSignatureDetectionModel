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
    'text.usetex': False, # Keep false unless latex is installed, but use math mode string
})

FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
RESULTS_DIR = "final_results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Load Data ---
print(f"--- Loading {FILL_GAS} Data ---")
df = pd.read_parquet(TRAIN_FILE)

# Extract Spectral Columns and Wavelengths
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [c for c in df.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]
spectral_cols_sorted = sorted(spectral_cols, key=lambda x: float(x))
wavelengths = np.array([float(x) for x in spectral_cols_sorted])

X = df[spectral_cols_sorted].values

# --- Preprocessing & PCA ---
print("--- Fitting Scaler and PCA ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

# --- Reconstruct a Single Spectrum ---
SPECTRUM_IDX = 0
original_scaled = scaler.transform(X[SPECTRUM_IDX].reshape(1, -1))
pca_transformed = pca.transform(original_scaled)

# Reconstruction with components 0 to 101 (0:102)
pca_subset_0_102 = np.zeros_like(pca_transformed)
pca_subset_0_102[:, 0:102] = pca_transformed[:, 0:102]
recon_scaled_0_102 = pca.inverse_transform(pca_subset_0_102)
recon_flux_0_102 = scaler.inverse_transform(recon_scaled_0_102)[0]

true_flux = X[SPECTRUM_IDX]

# Calculate Residuals (Standard fractional residual, no -1 offset)
residuals_0_102 = (recon_flux_0_102 - true_flux) / (true_flux + 1e-9)

# --- Generate Two-Panel Plot ---
print("--- Generating Two-Panel Visualization ---")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top Panel: Original vs Reconstructed
ax1.plot(wavelengths, true_flux, color='black', linewidth=2.5, label='Original Spectrum', zorder=1)
ax1.plot(wavelengths, recon_flux_0_102, color='orange', linestyle='--', linewidth=2.5, label='PCA Recon (PCs 0–101)', zorder=2)
ax1.set_ylabel(r'Transit Depth ($\Delta F/F$)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='-', alpha=0.5)

# Bottom Panel: Residual Error (Centered at 0, zoomed to [-0.01, 0.01])
ax2.plot(wavelengths, residuals_0_102, color='royalblue', linewidth=2.0, label='Residuals (PCs 0-101)', zorder=2)
ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1) 

ax2.set_xlabel(r'Wavelength [$\mu$m]')
ax2.set_ylabel(r'Residual Error $\left( \frac{X_{recon} - X_{true}}{X_{true}} \right)$')
ax2.set_ylim(-0.010, 0.010) 
ax2.grid(True, linestyle='-', alpha=0.5)
ax2.legend(loc='upper right')

plt.tight_layout()

# Save the plot
output_path = os.path.join(RESULTS_DIR, 'H2_pca_two_panel_reconstruction_final.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Two-panel plot saved to: {output_path}")
