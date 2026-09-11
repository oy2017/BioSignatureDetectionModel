import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import re

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Visualize PCA reconstruction of a spectrum from a specific component onwards.")
parser.add_argument("fill_gas", type=str, help="The fill gas for the atmosphere (e.g., H2, N2).")
parser.add_argument("n_components", type=int, help="The total number of principal components to use.")
parser.add_argument("start_component", type=int, help="The starting component for reconstruction (1-indexed).")
parser.add_argument("spectrum_index", type=int, help="The index of the spectrum to visualize.")
args = parser.parse_args()
fill_gas = args.fill_gas.upper()
n_components = args.n_components
start_component = args.start_component
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

# --- 4. Modify PCA components for reconstruction ---
if start_component > 1:
    print(f"--- Zeroing out components before component {start_component} ---")
    X_pca_modified = X_pca.copy()
    X_pca_modified[:, :start_component - 1] = 0
else:
    X_pca_modified = X_pca

# --- 5. Reconstruct the Spectrum ---
print(f"--- Reconstructing spectrum at index {spectrum_index} ---")
X_reconstructed_scaled = pca.inverse_transform(X_pca_modified)
X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)
reconstructed_spectrum = X_reconstructed[spectrum_index]

# --- 6. Plot Original vs. Reconstructed Spectrum ---
plt.figure(figsize=(12, 6))
plt.plot(wavelengths, original_spectrum, label='Original Spectrum', color='blue', alpha=0.7)
plt.plot(wavelengths, reconstructed_spectrum, label=f'Reconstructed (Components {start_component}-{n_components})', color='red', linestyle='--', alpha=0.7)
plt.title(f'Original vs. Reconstructed Spectrum (Components {start_component}-{n_components}) for {fill_gas} (Index: {spectrum_index})')
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.legend()
plt.grid(True)

plot_filename = f'{fill_gas}_pca_reconstruction_c{start_component}-{n_components}_idx{spectrum_index}.png'
plt.savefig(plot_filename)
print(f"\nReconstruction plot saved to {plot_filename}")
