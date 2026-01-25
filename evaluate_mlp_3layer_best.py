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

# --- Set Random Seeds ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

results_path = 'final_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Rigid 3-Layer MLP (Best Performer) on PCA Spectra.")
parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
parser.add_argument("--epochs", type=int, default=50, help="Epochs.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch Size.")
args = parser.parse_args()

fill_gas = args.fill_gas.upper()

# --- 1. Load Data ---
print(f"--- Loading H2 Data for Best 3-Layer MLP ---")
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

# --- 2. Preprocessing (Standard Scaling + PCA) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- Applying PCA (0-100) ---")
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Filter to Top 100 Components
X_train_processed = X_train_pca[:, 0:100]
X_test_processed = X_test_pca[:, 0:100]

# Rescale PCA Components
scaler_pca = StandardScaler()
X_train_processed = scaler_pca.fit_transform(X_train_processed)
X_test_processed = scaler_pca.transform(X_test_processed)

# --- 3. Build Rigid 3-Layer Architecture ---
# Config: 256 -> 128 -> 64
print("--- Building 3-Layer MLP [256, 128, 64] ---")
model = Sequential([
    Flatten(input_shape=(100,)),
    
    # Layer 1
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),
    
    # Layer 2
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),
    
    # Layer 3
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),
    
    # Output
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# --- 4. Train ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(
    X_train_processed, y_train,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# --- 5. Evaluate ---
loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = (model.predict(X_test_processed) > 0.5).astype(int)

# --- 6. Save Report ---
report_name = f'best_mlp_3layer_report.txt'
with open(os.path.join(results_path, report_name), 'w') as f:
    f.write(classification_report(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio']))
