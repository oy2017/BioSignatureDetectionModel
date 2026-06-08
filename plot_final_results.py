import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re

# --- Configuration ---
FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
RESULTS_DIR = "final_results/plots"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Load Data ---
print(f"--- Loading Data from {TRAIN_FILE} ---")
df = pd.read_parquet(TRAIN_FILE)
df['label'] = df['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

# Extract Spectral Columns
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [c for c in df.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]
# Sort columns by wavelength (assuming they are floats in strings)
spectral_cols_sorted = sorted(spectral_cols, key=lambda x: float(x))
wavelengths = np.array([float(x) for x in spectral_cols_sorted])

X = df[spectral_cols_sorted].values
y = df['label'].values

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 1. Model Comparison Bar Chart ---
# Data from final Master 5-Set Evaluation Run (Rigorous Pipeline)
model_data = {
    'Model': ['MLP', 'CNN', 'Random Forest', 'XGBoost'],
    'Accuracy': [0.8521, 0.8098, 0.8664, 0.8815],
    'Std': [0.0050, 0.0097, 0.0102, 0.0140]
}
df_models = pd.DataFrame(model_data)

plt.figure(figsize=(10, 6))
bars = plt.bar(df_models['Model'], df_models['Accuracy'], yerr=df_models['Std'], 
               capsize=10, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'], alpha=0.9)
plt.ylim(0.75, 0.95)
plt.title(f'Model Accuracy Comparison (5 Independent Test Sets)', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(axis='y', alpha=0.3)

# Add labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    error = df_models['Std'][i]
    plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.002,
             f'{height:.2%} ± {error:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison_bar_chart.png'), dpi=300)
plt.close()
print("Saved: model_comparison_bar_chart.png")

# --- 2. PCA Scatter Plot ---
print("--- Generating PCA Plots ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['biosignature'], 
                palette={'yes': 'crimson', 'no': 'navy'}, alpha=0.6, s=15)
plt.title('PCA of Spectral Data (PC1 vs PC2)', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)', fontsize=12)
plt.legend(title='Biosignature')
plt.savefig(os.path.join(RESULTS_DIR, 'pca_scatter_plot.png'), dpi=300)
plt.close()
print("Saved: pca_scatter_plot.png")

# --- 3. Mean Spectra with Std Dev ---
print("--- Generating Spectral Plots ---")
X_bio = X[y == 1]
X_nonbio = X[y == 0]

mean_bio = np.mean(X_bio, axis=0)
std_bio = np.std(X_bio, axis=0)
mean_nonbio = np.mean(X_nonbio, axis=0)
std_nonbio = np.std(X_nonbio, axis=0)

plt.figure(figsize=(12, 6))
# Plot Non-Bio first (Background)
plt.plot(wavelengths, mean_nonbio, label='Non-Bio Mean', color='navy', linewidth=2)
plt.fill_between(wavelengths, mean_nonbio - std_nonbio, mean_nonbio + std_nonbio, color='navy', alpha=0.2)

# Plot Bio
plt.plot(wavelengths, mean_bio, label='Bio Mean', color='crimson', linewidth=2)
plt.fill_between(wavelengths, mean_bio - std_bio, mean_bio + std_bio, color='crimson', alpha=0.2)

plt.title('Mean Spectra: Biosignature vs Non-Biosignature', fontsize=16)
plt.xlabel('Wavelength (microns)', fontsize=12)
plt.ylabel('Transit Depth', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'mean_spectra_comparison.png'), dpi=300)
plt.close()
print("Saved: mean_spectra_comparison.png")

# --- 4. Difference Spectrum ---
diff_spectrum = mean_bio - mean_nonbio

plt.figure(figsize=(12, 4))
plt.plot(wavelengths, diff_spectrum, color='purple', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Difference Spectrum (Mean Bio - Mean Non-Bio)', fontsize=16)
plt.xlabel('Wavelength (microns)', fontsize=12)
plt.ylabel('Delta Transit Depth', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'difference_spectrum.png'), dpi=300)
plt.close()
print("Saved: difference_spectrum.png")

# --- 5. PCA Loadings Plot ---
# Which wavelengths contribute most to PC1 and PC2?
loadings = pca.components_

plt.figure(figsize=(12, 6))
plt.plot(wavelengths, loadings[0], label='PC1 Loading', color='blue')
plt.plot(wavelengths, loadings[1], label='PC2 Loading', color='orange', alpha=0.8)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('PCA Loadings (Feature Importance by Wavelength)', fontsize=16)
plt.xlabel('Wavelength (microns)', fontsize=12)
plt.ylabel('Loading Weight', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'pca_loadings_plot.png'), dpi=300)
plt.close()
print("Saved: pca_loadings_plot.png")

# --- 6. Spectral Heatmap ---
# Sort data by label for the heatmap
sorted_indices = np.argsort(y)
X_sorted = X_scaled[sorted_indices]
y_sorted = y[sorted_indices]

# Find the boundary index
boundary = np.searchsorted(y_sorted, 1)

plt.figure(figsize=(12, 8))
# Downsample rows for visibility if too large (e.g., take every 5th row)
step = 1 if len(X_sorted) < 1000 else len(X_sorted) // 1000 + 1
sns.heatmap(X_sorted[::step], cmap='viridis', cbar_kws={'label': 'Scaled Transit Depth'})

# Add horizontal line separating classes
plt.axhline(boundary/step, color='white', linestyle='--', linewidth=2)
plt.text(X.shape[1] + 5, boundary/step/2, 'Non-Bio', va='center', rotation=270, fontsize=14, color='navy')
plt.text(X.shape[1] + 5, (len(X_sorted) + boundary)/step/2, 'Bio', va='center', rotation=270, fontsize=14, color='crimson')

plt.title('Spectral Heatmap (Sorted by Class)', fontsize=16)
plt.xlabel('Spectral Channel Index', fontsize=12)
plt.ylabel('Sample Index', fontsize=12)
plt.yticks([]) # Hide Y ticks as individual samples don't matter
plt.savefig(os.path.join(RESULTS_DIR, 'spectral_heatmap.png'), dpi=300)
plt.close()
print("Saved: spectral_heatmap.png")

print("\n--- All Plots Generated Successfully ---")
