import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
import os

# --- 1. Load Training Data ---
fill_gas = "H2"
print(f"Loading {fill_gas} Training Data...")
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')

# --- 2. Prepare Data ---
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X_train = df_train[spectral_cols]

# --- 3. Scale Data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 4. Perform PCA ---
print("Fitting PCA...")
pca = PCA()
pca.fit(X_train_scaled)

explained_variance = pca.explained_variance_ratio_

# --- 5. Generate Scree Plot (Log Scale, No Title) ---
print("Generating Scree Plot...")
plt.figure(figsize=(8, 5))

# Plot first 105 components to show PC0, PC1, and the 100 we kept
num_components_to_plot = 105 
components = np.arange(num_components_to_plot)

plt.plot(components, explained_variance[:num_components_to_plot], marker='o', markersize=4, linestyle='-', color='#1f77b4')

# Reviewer requested log scale on y-axis
plt.yscale('log')

# Reviewer requested no title, just clear axes
plt.xlabel('Principal Component Index', fontsize=12)
plt.ylabel('Explained Variance Ratio (Log Scale)', fontsize=12)

# Make lines/grid more prominent as requested
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xticks(np.arange(0, num_components_to_plot + 1, 10))

# Save the plot
output_dir = 'final_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plot_filename = os.path.join(output_dir, f'{fill_gas}_pca_scree_plot_final.png')
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)
print(f"Plot successfully saved to {plot_filename}")
