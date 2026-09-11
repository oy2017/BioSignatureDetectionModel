import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import os
import re

# --- Configuration ---
FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILES = [f"multirex_spectra_{FILL_GAS}_test_set_{i}.parquet" for i in range(1, 6)]
RESULTS_DIR = "final_results/physics_analysis"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Helper to load data but KEEP metadata ---
def load_data_with_meta(file_path):
    df = pd.read_parquet(file_path)
    df['label'] = df['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Identify spectral columns
    float_pattern = re.compile(r"^-?\d+\.\d+$")
    spectral_cols = [col for col in df.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    
    # Identify metadata columns (everything else)
    meta_cols = [col for col in df.columns if col not in spectral_cols and col != 'label' and col != 'biosignature']
    
    X = df[spectral_cols].values
    y = df['label'].values
    meta = df[meta_cols]
    
    return X, y, meta

# --- 1. Train Best XGBoost Model ---
print("--- Loading Training Data ---")
X_train_raw, y_train, _ = load_data_with_meta(TRAIN_FILE)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

# Feature Set: PCA 2-102
X_train_clean = X_train_pca[:, 2:102]
scaler_clean = StandardScaler()
X_train_clean = scaler_clean.fit_transform(X_train_clean)

print("--- Training Best XGBoost Model (Est=200, Depth=5, LR=0.1) ---")
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, n_jobs=-1, eval_metric='logloss', random_state=42)
model.fit(X_train_clean, y_train)

# --- 2. Load and Predict on All Test Sets ---
print("--- Predicting on 5 Test Sets ---")
all_errors = []
all_meta = []
all_y_true = []
all_y_pred = []

for f in TEST_FILES:
    X_test_raw, y_test, meta = load_data_with_meta(f)
    
    # Preprocess
    X_test_scaled = scaler.transform(X_test_raw)
    X_test_pca = pca.transform(X_test_scaled)
    X_test_clean = scaler_clean.transform(X_test_pca[:, 2:102])
    
    # Predict
    y_pred = model.predict(X_test_clean)
    
    # Calculate Error (1 if wrong, 0 if correct)
    errors = (y_pred != y_test).astype(int)
    
    all_errors.extend(errors)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_meta.append(meta)

# Combine Metadata
full_meta = pd.concat(all_meta, ignore_index=True)
full_meta['Error'] = all_errors
full_meta['True_Label'] = all_y_true
full_meta['Pred_Label'] = all_y_pred

# Filter: We care mostly about "Missed Biosignatures" (False Negatives) vs "False Alarms" (False Positives)
# But simple "Error Rate" is a good start.

print(f"Total Test Samples: {len(full_meta)}")
print(f"Total Errors: {full_meta['Error'].sum()}")
print(f"Overall Error Rate: {full_meta['Error'].mean():.4f}")

# --- 3. Analysis & Plotting ---

def plot_error_rate_by_bin(df, col, bins=10, title_suffix="", xlabel=""):
    """Creates a plot showing error rate across bins of a continuous variable."""
    # Create bins
    df['bin'] = pd.cut(df[col], bins=bins)
    
    # Group by bin and calculate mean error rate
    bin_stats = df.groupby('bin', observed=False)['Error'].mean().reset_index()
    bin_stats['bin_center'] = bin_stats['bin'].apply(lambda x: x.mid).astype(float)
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=bin_stats, x='bin_center', y='Error', marker='o', linewidth=2.5)
    plt.title(f'Error Rate vs {title_suffix}')
    plt.xlabel(xlabel)
    plt.ylabel('Error Rate (0.0 - 1.0)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(bin_stats['Error']) * 1.2) # Add some headroom
    
    filename = os.path.join(RESULTS_DIR, f'error_vs_{col.replace(" ", "_")}.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

# 3.1 Planet Radius
plot_error_rate_by_bin(full_meta, 'p_radius', bins=10, title_suffix="Planet Radius (Earth Radii)", xlabel="Planet Radius (Re)")

# 3.2 Planet Mass
plot_error_rate_by_bin(full_meta, 'p_mass', bins=10, title_suffix="Planet Mass (Earth Masses)", xlabel="Planet Mass (Me)")

# 3.3 Star Temperature
plot_error_rate_by_bin(full_meta, 's temperature', bins=10, title_suffix="Star Temperature (K)", xlabel="Star Temp (K)")

# 3.4 Atmosphere Temperature
plot_error_rate_by_bin(full_meta, 'atm temperature', bins=10, title_suffix="Atmosphere Temp (K)", xlabel="Atm Temp (K)")

# 3.5 Scatter Plot: Star Temp vs Planet Radius (Colored by Error)
plt.figure(figsize=(10, 8))
# Plot correct points first (small, grey)
correct = full_meta[full_meta['Error'] == 0]
sns.scatterplot(data=correct, x='s temperature', y='p_radius', color='lightgrey', s=20, alpha=0.3, label='Correct')

# Plot errors on top (red)
errors = full_meta[full_meta['Error'] == 1]
sns.scatterplot(data=errors, x='s temperature', y='p_radius', color='red', s=40, alpha=0.8, label='Error')

plt.title('Error Distribution: Star Temp vs Planet Radius')
plt.xlabel('Star Temperature (K)')
plt.ylabel('Planet Radius (Re)')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'scatter_error_temp_radius.png'))
plt.close()

# 3.6 Scatter Plot: CH4 vs O3 (The Biosignature Threshold)
# Biosignature is defined as CH4 > -6 AND O3 > -7.
# Let's see if errors cluster near these lines.
plt.figure(figsize=(10, 8))

# Draw threshold lines
plt.axvline(x=-6, color='black', linestyle='--', alpha=0.5, label='CH4 Threshold')
plt.axhline(y=-7, color='black', linestyle='--', alpha=0.5, label='O3 Threshold')

# Plot Correct
sns.scatterplot(data=correct, x='atm CH4', y='atm O3', color='lightgrey', s=20, alpha=0.2, label='Correct')
# Plot Errors
sns.scatterplot(data=errors, x='atm CH4', y='atm O3', color='red', s=30, alpha=0.8, label='Error')

plt.title('Error Distribution: Chemical Concentrations (Log)')
plt.xlabel('log(CH4)')
plt.ylabel('log(O3)')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'scatter_error_chemistry.png'))
plt.close()

# --- 4. Textual Summary ---
print("\n--- Error Analysis Summary ---")
# Check High Temp Stars Error Rate
high_temp_stars = full_meta[full_meta['s temperature'] > 6000]
low_temp_stars = full_meta[full_meta['s temperature'] < 4000]
print(f"Error Rate for Hot Stars (>6000K): {high_temp_stars['Error'].mean():.2%}")
print(f"Error Rate for Cool Stars (<4000K): {low_temp_stars['Error'].mean():.2%}")

# Check Small Planets
small_planets = full_meta[full_meta['p_radius'] < 8.0] # Arbitrary cutoff for 'small' in this dataset range
large_planets = full_meta[full_meta['p_radius'] > 12.0]
print(f"Error Rate for Small Planets (<8 Re): {small_planets['Error'].mean():.2%}")
print(f"Error Rate for Large Planets (>12 Re): {large_planets['Error'].mean():.2%}")

print(f"Plots saved to {RESULTS_DIR}")
