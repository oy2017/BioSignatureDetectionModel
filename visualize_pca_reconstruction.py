import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import re

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Visualize PCA reconstruction of a spectrum.")
parser.add_argument("fill_gas", type=str, help="The fill gas for the atmosphere (e.g., H2, N2).")
parser.add_argument("n_components", type=int, help="The number of principal components to use for reconstruction.")
parser.add_argument("spectrum_index", type=int, help="The index of the spectrum to visualize.")
args = parser.parse_args()
fill_gas = args.fill_gas.upper()
n_components = args.n_components
spectrum_index = args.spectrum_index

# --- 1. Load Data ---
print(f"--- Loading {fill_gas} Data ---")
try:
    df = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
except FileNotFoundError:
    print(f"Error: Data file not found for {fill_gas}. Please generate it first.")
    exit()

# --- 2. Prepare Data ---
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X = df[spectral_cols]
wavelengths = X.columns.astype(float)

# Select the specific spectrum
original_spectrum = X.iloc[spectrum_index].values

# --- 3. Scale Data and Perform PCA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"--- Performing PCA with {n_components} components ---")
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# --- 4. Reconstruct the Spectrum ---
print(f"--- Reconstructing spectrum at index {spectrum_index} ---")
X_reconstructed_scaled = pca.inverse_transform(X_pca)
X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)
reconstructed_spectrum = X_reconstructed[spectrum_index]

# --- 5. Plot Original vs. Reconstructed Spectrum ---
plt.figure(figsize=(12, 6))
plt.plot(wavelengths, original_spectrum, label='Original Spectrum', color='blue', alpha=0.7)
plt.plot(wavelengths, reconstructed_spectrum, label=f'Reconstructed Spectrum ({n_components} components)', color='red', linestyle='--', alpha=0.7)
plt.title(f'Original vs. PCA Reconstructed Spectrum for {fill_gas} (Index: {spectrum_index})')
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.legend()
plt.grid(True)

plot_filename = f'{fill_gas}_pca_reconstruction_n{n_components}_idx{spectrum_index}.png'
plt.savefig(plot_filename)
print(f"\nReconstruction plot saved to {plot_filename}")
