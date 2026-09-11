import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# --- Configuration ---
FILL_GAS = "H2"
SEED = 42
RESULTS_DIR = "final_results/extended_physics"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILE = f"multirex_spectra_{FILL_GAS}_test_set_1.parquet"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Constants for Physics
KB = 1.380649e-23  # Boltzmann constant
AMU = 1.660539e-27 # Atomic Mass Unit
G = 6.674e-11      # Gravitational Constant
MEarth = 5.972e24  # Mass of Earth
REarth = 6.371e6   # Radius of Earth

def calculate_scale_height(row):
    """
    Calculates atmospheric scale height (H) in kilometers.
    H = (k * T) / (mu * g)
    """
    T = row['atm temperature']
    
    # Estimate Mean Molecular Weight (mu)
    # H2-dominated: mostly H2 (2.016) + He (4.003). Approx mu ~ 2.3 amu
    if FILL_GAS == 'H2':
        mu = 2.3 * AMU
    else:
        mu = 28.0 * AMU
        
    # Calculate Gravity (g = GM/R^2)
    M = row['p_mass'] * MEarth
    R = row['p_radius'] * REarth
    g = (G * M) / (R**2)
    
    # Scale Height
    H = (KB * T) / (mu * g)
    return H / 1000.0 # Convert to km

def get_data():
    print(f"--- Loading Training Data: {TRAIN_FILE} ---")
    df_train = pd.read_parquet(TRAIN_FILE)
    print(f"--- Loading Test Data: {TEST_FILE} ---")
    df_test = pd.read_parquet(TEST_FILE)
    
    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    spectral_cols_sorted = sorted(spectral_cols, key=float)
    
    X_train_raw = df_train[spectral_cols_sorted].values
    X_test_raw = df_test[spectral_cols_sorted].values
    
    print("--- Calculating Physical Parameters ---")
    # 1. Interfering Gases
    df_test['log_H2O'] = df_test['atm H2O']
    df_test['log_CO2'] = df_test['atm CO2']
    
    # 2. Scale Height
    df_test['Scale Height (km)'] = df_test.apply(calculate_scale_height, axis=1)
    
    params = df_test[['log_H2O', 'log_CO2', 'Scale Height (km)']]
    
    return X_train_raw, y_train, X_test_raw, y_test, params

def train_model(X_train, y_train):
    print("--- Training XGBoost (Aggressive Config) ---")
    # Matching the 'Aggressive' config from generate_separate_error_plots.py
    # n_estimators=300, max_depth=5, learning_rate=0.2, subsample=1.0, eval_metric='logloss'
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    X_clean = X_pca[:, 2:102] # Feature Set: PCA 2-102
    
    model = XGBClassifier(
        n_estimators=300, 
        max_depth=5, 
        learning_rate=0.2, 
        subsample=1.0,
        n_jobs=-1,
        random_state=SEED,
        eval_metric='logloss'
    )
    model.fit(X_clean, y_train)
    
    return model, scaler, pca

def predict(model, scaler, pca, X_test):
    X_scaled = scaler.transform(X_test)
    X_pca = pca.transform(X_scaled)
    X_clean = X_pca[:, 2:102]
    return model.predict(X_clean)

def plot_error_rate(y_true, y_pred, param_values, param_name):
    # Binning
    df_plot = pd.DataFrame({'val': param_values, 'true': y_true, 'pred': y_pred})
    df_plot['error'] = (df_plot['true'] != df_plot['pred']).astype(int)
    
    bins = 10
    # Use qcut for equal-sized bins if possible
    try:
        df_plot['bin'] = pd.qcut(df_plot['val'], q=bins, duplicates='drop')
    except:
        df_plot['bin'] = pd.cut(df_plot['val'], bins=bins)
        
    bin_stats = df_plot.groupby('bin', observed=True).agg(
        mean_val=('val', 'mean'),
        error_rate=('error', 'mean'),
        count=('error', 'count')
    ).reset_index()
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=bin_stats, x='mean_val', y='error_rate', marker='o', linewidth=2.5, color='crimson')
    plt.title(f'Error Rate vs {param_name}', fontsize=14, fontweight='bold')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save
    safe_name = param_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')
    filename = os.path.join(RESULTS_DIR, f'error_vs_{safe_name}.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def main():
    X_train, y_train, X_test, y_test, params = get_data()
    
    model, scaler, pca = train_model(X_train, y_train)
    y_pred = predict(model, scaler, pca, X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    
    print("\n--- Generating Plots ---")
    for col in params.columns:
        plot_error_rate(y_test, y_pred, params[col], col)
        
if __name__ == "__main__":
    main()
