import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import argparse
import re
import os

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Grid Search Hyperparameter Tuning for Random Forest.")
parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
parser.add_argument("--pca_components", type=int, default=100, help="Number of PCA components to use (feature engineering).")
args = parser.parse_args()
fill_gas = args.fill_gas.upper()

# --- 1. Load Data ---
print(f"--- Loading {fill_gas} Data for Random Forest Grid Search ---")
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

# Labels
y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

# Features
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X_train_raw = df_train[spectral_cols]
X_test_raw = df_test[spectral_cols]

# --- 2. Preprocessing (Scaling + PCA) ---
print("--- Preprocessing: Scaling ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

print(f"--- Preprocessing: PCA (Components 2 to {args.pca_components + 2}) ---")
pca = PCA()
X_train_pca_full = pca.fit_transform(X_train_scaled)
X_test_pca_full = pca.transform(X_test_scaled)

# Slice: Skip first 2, take next N
start_idx = 2
end_idx = start_idx + args.pca_components
X_train_pca = X_train_pca_full[:, start_idx:end_idx]
X_test_pca = X_test_pca_full[:, start_idx:end_idx]

print(f"Feature Shape: {X_train_pca.shape}")

# --- 3. Define Grid ---
# Random Forest Parameters to Tune:
# n_estimators: Number of trees.
# max_depth: Max depth of tree. None = Expand until pure.
# min_samples_split: Min samples required to split an internal node.
# min_samples_leaf: Min samples required to be at a leaf node.
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# --- 4. Setup GridSearchCV ---
print("--- Starting Grid Search (108 combinations x 3 folds = 324 fits) ---")
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1
)

# --- 5. Run Search ---
grid_search.fit(X_train_pca, y_train)

# --- 6. Results ---
print("\n--- Grid Search Complete ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Val Accuracy: {grid_search.best_score_:.4f}")

# --- 7. Final Evaluation on Test Set ---
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTest Set Accuracy (Best Model): {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio']))

# Save Report
results_path = 'final_results'
if not os.path.exists(results_path):
    os.makedirs(results_path)
    
with open(os.path.join(results_path, f'{fill_gas}_rf_gridsearch_report.txt'), 'w') as f:
    f.write(f"Best Params: {grid_search.best_params_}\n")
    f.write(f"Test Accuracy: {test_acc}\n\n")
    f.write(classification_report(y_test, y_pred))
