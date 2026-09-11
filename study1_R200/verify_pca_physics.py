import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import re

def main():
    print("--- Loading H2 Training Data ---")
    try:
        df = pd.read_parquet('multirex_spectra_H2_train.parquet')
    except FileNotFoundError:
        print("Error: Training data not found.")
        return

    # Extract spectral columns
    float_pattern = re.compile(r"^-?\d+\.\d+$")
    spectral_cols = [col for col in df.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X = df[spectral_cols]

    # Preprocessing
    print("--- Scaling Data and Running PCA ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate physical proxies
    # Base transit depth is roughly (Planet Radius / Stellar Radius)^2
    # Note: ensure units match or are relative. Typically p_radius is Earth radii, s_radius is Solar radii.
    # Conversion: 1 Solar Radius ~ 109.2 Earth Radii
    transit_depth = (df['p_radius'] / (df['s radius'] * 109.2))**2
    
    # Encode target variable
    y_encoded = df['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

    print("\n--- Correlation Analysis for PC0 and PC1 ---")
    
    # 1. Correlation with Base Transit Depth
    corr_pc0_depth, _ = pearsonr(X_pca[:, 0], transit_depth)
    corr_pc1_depth, _ = pearsonr(X_pca[:, 1], transit_depth)
    print(f"PC0 vs Base Transit Depth: r = {corr_pc0_depth:.4f}")
    print(f"PC1 vs Base Transit Depth: r = {corr_pc1_depth:.4f}")
    
    # 2. Correlation with Stellar Temperature (influences overall spectral slope / continuum)
    corr_pc0_stemp, _ = pearsonr(X_pca[:, 0], df['s temperature'])
    corr_pc1_stemp, _ = pearsonr(X_pca[:, 1], df['s temperature'])
    print(f"PC0 vs Stellar Temperature: r = {corr_pc0_stemp:.4f}")
    print(f"PC1 vs Stellar Temperature: r = {corr_pc1_stemp:.4f}")
    
    # 3. Correlation with Atmospheric Temperature
    corr_pc0_atemp, _ = pearsonr(X_pca[:, 0], df['atm temperature'])
    corr_pc1_atemp, _ = pearsonr(X_pca[:, 1], df['atm temperature'])
    print(f"PC0 vs Atmospheric Temperature: r = {corr_pc0_atemp:.4f}")
    print(f"PC1 vs Atmospheric Temperature: r = {corr_pc1_atemp:.4f}")

    # 4. Correlation with Target Biosignature Label
    corr_pc0_bio, _ = pearsonr(X_pca[:, 0], y_encoded)
    corr_pc1_bio, _ = pearsonr(X_pca[:, 1], y_encoded)
    print(f"\nPC0 vs Biosignature Label: r = {corr_pc0_bio:.4f}")
    print(f"PC1 vs Biosignature Label: r = {corr_pc1_bio:.4f}")

    # 5. Correlation with individual gas abundances (log)
    corr_pc0_ch4, _ = pearsonr(X_pca[:, 0], np.log10(df['atm CH4']))
    corr_pc1_ch4, _ = pearsonr(X_pca[:, 1], np.log10(df['atm CH4']))
    corr_pc0_o3, _ = pearsonr(X_pca[:, 0], np.log10(df['atm O3']))
    corr_pc1_o3, _ = pearsonr(X_pca[:, 1], np.log10(df['atm O3']))
    print(f"\nPC0 vs Log(CH4): r = {corr_pc0_ch4:.4f}")
    print(f"PC1 vs Log(CH4): r = {corr_pc1_ch4:.4f}")
    print(f"PC0 vs Log(O3): r = {corr_pc0_o3:.4f}")
    print(f"PC1 vs Log(O3): r = {corr_pc1_o3:.4f}")
    
    print("\n--- Conclusion ---")
    if abs(corr_pc0_depth) > 0.9:
        print("-> Confirmed: PC0 strongly represents the mean transit depth (R_p / R_s)^2.")
    if abs(corr_pc0_bio) < 0.1 and abs(corr_pc1_bio) < 0.1:
        print("-> Confirmed: PC0 and PC1 have virtually zero correlation with the biosignature target.")

if __name__ == "__main__":
    main()
