import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import argparse
import re

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Measure reconstruction error (Variance/MSE) and Correlation/Covariance for different PCA component counts.")
parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
parser.add_argument("--indices", type=str, default="20,40,60,80,100", help="Comma-separated list of component counts to test.")
args = parser.parse_args()
fill_gas = args.fill_gas.upper()
component_counts = [int(x) for x in args.indices.split(',')]

# --- 1. Load Data ---
print(f"--- Loading {fill_gas} Data ---")
df = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')

# Prepare X (Spectra)
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X = df[spectral_cols].values

# Prepare y (Labels) for Correlation check
y = LabelEncoder().fit_transform(df['biosignature'])

# --- 2. Scale Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n{'Components':<12} | {'MSE (Variance of Error)':<25} | {'Correlation (Best Single Comp)':<30}")
print("-" * 75)

# --- 3. Loop through Component Counts ---
for n in component_counts:
    # A. Perform PCA
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_scaled)
    
    # B. Reconstruct to measure MSE (Variance of Error)
    X_reconstructed_scaled = pca.inverse_transform(X_pca)
    # MSE between Scaled Original and Scaled Reconstructed
    # We use scaled to be fair, otherwise intensity differences dominate
    mse = mean_squared_error(X_scaled, X_reconstructed_scaled)
    
    # C. Measure Correlation of the *newest* components added
    # (i.e., what did adding components 0-N give us in terms of signal?)
    # We look for the max correlation found in ANY of the components up to N
    correlations = [np.abs(np.corrcoef(X_pca[:, i], y)[0, 1]) for i in range(n)]
    max_corr = max(correlations)
    
    print(f"{n:<12} | {mse:<25.6f} | {max_corr:<30.6f}")

print("-" * 75)
print("Note: Lower MSE is better (Original vs Reconstruction).")
print("Note: Higher Correlation is better (Component vs Biosignature).")
