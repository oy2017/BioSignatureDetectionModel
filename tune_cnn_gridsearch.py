import os
import argparse
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import re
import random

# Set environment variable to avoid OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- Configuration ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

results_path = 'final_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Grid Search for CNN Hyperparameters.")
parser.add_argument("fill_gas", type=str, default='H2', help="The fill gas (e.g., H2).")
parser.add_argument("--pca_start_idx", type=int, default=2, help="Start index for PCA components.")
parser.add_argument("--pca_end_idx", type=int, default=102, help="End index for PCA components.")
parser.add_argument("--use_pca", action="store_true", help="Enable PCA preprocessing.")
args = parser.parse_args()

fill_gas = args.fill_gas.upper()
pca_start_idx = args.pca_start_idx
pca_end_idx = args.pca_end_idx
use_pca = args.use_pca

# --- Load Data ---
print(f"--- Loading {fill_gas} Dataset for Grid Search ---")
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]

X_train = df_train[spectral_cols]
y_train = df_train['label']
X_test = df_test[spectral_cols]
y_test = df_test['label']

# --- Filter Invalid Values (> 1.0) ---
train_mask = (X_train <= 1.0).all(axis=1)
test_mask = (X_test <= 1.0).all(axis=1)
print(f"Filtering out {len(X_train) - train_mask.sum()} invalid rows from training set.")
X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test = X_test[test_mask]
y_test = y_test[test_mask]

# --- Shuffle Training Data (Critical for matching manual run) ---
X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

# --- Preprocessing ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if use_pca:
    print(f"--- Applying PCA ({pca_start_idx}-{pca_end_idx}) ---")
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    X_train_processed = X_train_pca[:, pca_start_idx:pca_end_idx]
    X_test_processed = X_test_pca[:, pca_start_idx:pca_end_idx]

    # Rescale PCA components for NN
    scaler_pca = StandardScaler()
    X_train_processed = scaler_pca.fit_transform(X_train_processed)
    X_test_processed = scaler_pca.transform(X_test_processed)
    
    report_suffix = "pca"
else:
    print(f"--- Using Raw Spectra (No PCA) ---")
    X_train_processed = X_train_scaled
    X_test_processed = X_test_scaled
    report_suffix = "raw"

# Reshape for CNN (samples, timesteps, features)
X_train_processed = X_train_processed.reshape(X_train_processed.shape[0], X_train_processed.shape[1], 1)
X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], X_test_processed.shape[1], 1)
input_shape = (X_train_processed.shape[1], 1)

# --- Grid Search Parameters ---
param_grid = {
    'filters': [32, 64],
    'kernel_size': [3, 5],
    'dropout_rate': [0.3, 0.5],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64]
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"--- Starting Grid Search with {len(combinations)} combinations ---")

best_accuracy = 0.0
best_params = {}
results = []

for i, params in enumerate(combinations):
    print(f"\n--- Training Model {i+1}/{len(combinations)}: {params} ---")
    
    model = Sequential([
        Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(params['dropout_rate']),
        
        Conv1D(filters=params['filters']*2, kernel_size=params['kernel_size'], padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(params['dropout_rate']),
        
        Flatten(),
        Dense(100),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train_processed, y_train,
        epochs=100, # Increased epochs for thorough training
        batch_size=params['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0 
    )
    
    loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    results.append({**params, 'accuracy': accuracy})
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        print(f"*** New Best Model! Accuracy: {accuracy:.4f} ***")

# --- Save Results ---
print("\n--- Grid Search Complete ---")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best Parameters: {best_params}")

report_filename = os.path.join(results_path, f'{fill_gas}_cnn_gridsearch_{report_suffix}_report.txt')
with open(report_filename, 'w') as f:
    f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
    f.write(f"Best Parameters: {best_params}\n\n")
    f.write("All Results:\n")
    for res in results:
        f.write(f"{res}\n")

print(f"Report saved to {report_filename}")
