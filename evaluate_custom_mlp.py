import os
# Set environment variable to avoid OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
import argparse
import random
from sklearn.utils import shuffle

# --- Set Random Seeds for Reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Create Results Directory ---
results_path = 'final_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Evaluate an MLP with custom layers on PCA exoplanet spectra.")
parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2, N2).")
parser.add_argument("--pca_start_idx", type=int, default=0, help="PCA start index.")
parser.add_argument("--pca_end_idx", type=int, default=100, help="PCA end index.")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--layers", type=str, default="256,128,64", help="Comma-separated list of units for each hidden layer.")
args = parser.parse_args()

fill_gas = args.fill_gas.upper()
layer_units = [int(u) for u in args.layers.split(',')]

# --- 1. Load and Prepare Data ---
print(f"--- Loading and Preparing {fill_gas} Dataset for Custom MLP ---")
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

X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

# --- 2. Preprocessing ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
print(f"--- Applying PCA ({args.pca_start_idx}-{args.pca_end_idx}) ---")
pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train_scaled)
X_test_pca_full = pca_full.transform(X_test_scaled)

X_train_processed = X_train_pca_full[:, args.pca_start_idx:args.pca_end_idx]
X_test_processed = X_test_pca_full[:, args.pca_start_idx:args.pca_end_idx]

# Re-scale PCA components
scaler_pca = StandardScaler()
X_train_processed = scaler_pca.fit_transform(X_train_processed)
X_test_processed = scaler_pca.transform(X_test_processed)

input_shape = (X_train_processed.shape[1],)

# --- 3. Build Custom MLP ---
print(f"--- Building MLP with layers: {layer_units} ---")
model = Sequential()
model.add(Flatten(input_shape=input_shape))

for units in layer_units:
    model.add(Dense(units))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. Train Model ---
print(f"--- Training MLP for {args.epochs} epochs with batch size {args.batch_size} ---")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_processed, y_train,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# --- 5. Evaluate Model ---
loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = (model.predict(X_test_processed) > 0.5).astype(int)

# --- 6. Report ---
report_str = classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio'])
print(f"\n--- Custom MLP (Layers: {args.layers}) Results ---")
print(report_str)

report_filename = os.path.join(results_path, f'{fill_gas}_mlp_custom_L{len(layer_units)}_report.txt')
with open(report_filename, 'w') as f:
    f.write(f"Layers: {args.layers}\n")
    f.write(report_str)
