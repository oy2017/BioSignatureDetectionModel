import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re
import os

# --- Configuration ---
FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
RESULTS_DIR = "final_results"
SPECTRUM_IDX = 341 # Index used for the previous version

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- 1. Load and Prepare Data ---
print(f"Loading {FILL_GAS} Data...")
df = pd.read_parquet(TRAIN_FILE)
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = sorted([c for c in df.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))], key=float)
X = df[spectral_cols].values
wavelengths = np.array([float(x) for x in spectral_cols])

# --- 2. PCA Fitting ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA().fit(X_scaled)

# --- 3. Reconstruct Specific Spectrum (0-101) ---
print(f"Reconstructing spectrum index {SPECTRUM_IDX}...")
orig_scaled = scaler.transform(X[SPECTRUM_IDX].reshape(1, -1))
trans = pca.transform(orig_scaled)

# Keep all 102 components (0 to 101)
trans_recon = np.zeros_like(trans)
trans_recon[:, 0:102] = trans[:, 0:102]

recon_scaled = pca.inverse_transform(trans_recon)
recon = scaler.inverse_transform(recon_scaled)[0]
true_flux = X[SPECTRUM_IDX]

# --- 4. Calculate Residuals ---
# Formula: (Recon - True) / True
# This centers the "Perfect Score" at 0
residuals = (recon - true_flux) / (true_flux + 1e-9)

# --- 5. Generate Two-Panel Plot ---
print("Generating Plot...")
plt.figure(figsize=(10, 8))

# Top Panel: Spectra
plt.subplot(2, 1, 1)
plt.plot(wavelengths, true_flux, color='black', linewidth=1.5, label='Original Spectrum')
plt.plot(wavelengths, recon, color='darkorange', linestyle='--', linewidth=1.5, label='PCA Recon (PCs 0-101)')
plt.ylabel(r'Transit Depth ($\Delta F/F$)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='-', alpha=0.3)

# Bottom Panel: Residuals
plt.subplot(2, 1, 2)
plt.plot(wavelengths, residuals, color='royalblue', linewidth=1)
plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)
plt.ylabel(r'Residual Error $\left( \frac{X_{recon} - X_{true}}{X_{true}} \right)$')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.grid(True, linestyle='-', alpha=0.3)
plt.ylim(-0.01, 0.01) # Zoom in to show the tiny wiggles

plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, 'test_recon.png')
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Successfully recreated: {output_path}")
