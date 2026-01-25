import os
# Set environment variable to avoid OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
parser = argparse.ArgumentParser(description="Evaluate a 4-layer CNN on raw exoplanet spectra.")
parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2, N2).")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
args = parser.parse_args()

fill_gas = args.fill_gas.upper()

# --- 1. Load and Prepare Data ---
print(f"--- Loading and Preparing {fill_gas} Raw Dataset for 4-Layer CNN ---")
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

# Using raw (scaled) spectral data directly
X_train_processed = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_processed = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

input_shape = (X_train_processed.shape[1], 1)

# --- 3. Build 4-Layer CNN ---
print(f"--- Building 4-Layer CNN with input shape {input_shape} ---")
model = Sequential([
    # Layer 1
    Conv1D(filters=32, kernel_size=5, input_shape=input_shape, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    # Layer 2
    Conv1D(filters=64, kernel_size=5, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # Layer 3
    Conv1D(filters=128, kernel_size=5, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # Layer 4
    Conv1D(filters=256, kernel_size=5, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    Flatten(),
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. Train Model ---
print(f"--- Training 4-Layer CNN for {args.epochs} epochs ---")
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

# --- 6. Save and Display Results ---
report_str = classification_report(y_test, y_pred, target_names=['Non-Bio (0)', 'Bio (1)'])
print("--- 4-Layer CNN Raw Data Results ---")
print(report_str)

report_filename = os.path.join(results_path, f'{fill_gas}_cnn_4layer_raw_report.txt')
with open(report_filename, 'w') as f:
    f.write(report_str)
