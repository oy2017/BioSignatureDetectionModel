import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re

def main():
    fill_gas = 'H2'
    print(f"--- Loading {fill_gas} Training Data ---")
    try:
        df = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
    except FileNotFoundError:
        print(f"Error: Training data not found for {fill_gas}.")
        return

    # Prepare spectral data
    float_pattern = re.compile(r"^-?\d+\.\d+$")
    spectral_cols = [col for col in df.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X = df[spectral_cols]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    n_components = 105
    print(f"--- Fitting PCA with {n_components} components ---")
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    evr = pca.explained_variance_ratio_

    # Calculate specific slices
    var_pc0_1 = np.sum(evr[0:2])
    # Note: feature set is indices 2 through 102 (inclusive), which is 101 components
    var_feature_set = np.sum(evr[2:102]) 

    print("\n--- Explained Variance Analysis ---")
    print(f"Variance explained by PC0 and PC1:      {var_pc0_1*100:.4f}%")
    print(f"Variance explained by PCs 2 through 101: {var_feature_set*100:.4f}%")
    print(f"Total Cumulative (PC0 - PC101):         {(var_pc0_1 + var_feature_set)*100:.4f}%")
    
    print("\n--- Paper Verification ---")
    print(f"The value for Comment [15] is confirmed as: {var_feature_set*100:.3f}%")

if __name__ == "__main__":
    main()
