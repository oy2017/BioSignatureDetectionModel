import os
# Set environment variable to avoid OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import argparse
import re
import json

# --- Set Random Seeds for Reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Grid Search Hyperparameter Tuning for MLP (Custom Loop).")
parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
parser.add_argument("--pca_components", type=int, default=100, help="Number of PCA components to use (feature engineering).")
parser.add_argument("--n_splits", type=int, default=3, help="Number of CV folds.")
args = parser.parse_args()
fill_gas = args.fill_gas.upper()

# --- 1. Load Data ---
print(f"--- Loading {fill_gas} Data for MLP Grid Search ---")
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

# Labels
y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values

# Features
float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X_train_raw = df_train[spectral_cols].values
X_test_raw = df_test[spectral_cols].values

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

# Rescale PCA Components (Important for Neural Nets)
print("--- Preprocessing: Rescaling PCA features ---")
scaler_pca = StandardScaler()
X_train_final = scaler_pca.fit_transform(X_train_pca)
X_test_final = scaler_pca.transform(X_test_pca)

print(f"Feature Shape: {X_train_final.shape}")

# --- 3. Define Model Building Function ---
def create_mlp_model(input_dim, hidden_layers, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    for units in hidden_layers:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 4. Define Grid ---
# Using a slightly smaller grid than typical ML libraries to save time since MLP training is slower
param_grid = {
    'hidden_layers': [
        (256, 128), 
        (256, 128, 64),          # Best architecture from previous exp
        (512, 256, 128, 64)
    ],
    'dropout_rate': [0.2, 0.4],
    'learning_rate': [0.001],    # 0.001 is usually standard good start
    'batch_size': [32, 64],
    'epochs': [30]               # Fixed epochs for search speed
}

grid = list(ParameterGrid(param_grid))
print(f"--- Starting Grid Search ({len(grid)} combinations x {args.n_splits} folds) ---")

best_score = -1
best_params = None
best_model_history = None

kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)

results = []

for i, params in enumerate(grid):
    print(f"\nEvaluating Config {i+1}/{len(grid)}: {params}")
    
    cv_scores = []
    
    # K-Fold Loop
    fold_idx = 1
    for train_index, val_index in kf.split(X_train_final, y_train):
        X_tr, X_val = X_train_final[train_index], X_train_final[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        
        # Build Model
        model = create_mlp_model(
            input_dim=X_train_final.shape[1],
            hidden_layers=params['hidden_layers'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )
        
        # Train (Silent to avoid clutter)
        model.fit(
            X_tr, y_tr,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0
        )
        
        # Evaluate
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        cv_scores.append(acc)
        # print(f"  Fold {fold_idx} Acc: {acc:.4f}")
        fold_idx += 1
        
    mean_cv_score = np.mean(cv_scores)
    print(f"  -> Mean CV Accuracy: {mean_cv_score:.4f}")
    
    results.append({
        'params': params,
        'mean_cv_accuracy': mean_cv_score
    })
    
    if mean_cv_score > best_score:
        best_score = mean_cv_score
        best_params = params

# --- 5. Retrain Best Model on Full Train Set ---
print(f"\n--- Grid Search Complete ---")
print(f"Best CV Accuracy: {best_score:.4f}")
print(f"Best Parameters: {best_params}")

print("\n--- Retraining Best Model on Full Training Set ---")
final_model = create_mlp_model(
    input_dim=X_train_final.shape[1],
    hidden_layers=best_params['hidden_layers'],
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate']
)

# Train for a bit longer on the full set to ensure convergence? 
# Or stick to the grid param 'epochs'? 
# Usually good to train full epochs or slightly more. Let's stick to grid param to be safe.
final_model.fit(
    X_train_final, y_train,
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    verbose=1
)

# --- 6. Final Evaluation ---
y_pred_prob = final_model.predict(X_test_final)
y_pred = (y_pred_prob > 0.5).astype(int)

test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy (Best Model): {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio']))

# --- 7. Save Report ---
results_path = 'final_results'
if not os.path.exists(results_path):
    os.makedirs(results_path)

report_filename = f'{fill_gas}_mlp_gridsearch_report.txt'
with open(os.path.join(results_path, report_filename), 'w') as f:
    f.write(f"Best Parameters found:\n{json.dumps(best_params, indent=2)}\n\n")
    f.write(f"Best CV Accuracy: {best_score:.4f}\n")
    f.write(f"Test Set Accuracy: {test_acc:.4f}\n\n")
    f.write("Full Grid Results:\n")
    for res in results:
        f.write(f"{res}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio']))

print(f"Report saved to {os.path.join(results_path, report_filename)}")
