import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

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

# --- Best Parameters from Raw Grid Search (Modified for depth) ---
# Accuracy Baseline: 70.02%
FILTERS = 32
KERNEL_SIZE = 5 # Larger kernel for raw data
DROPOUT_RATE = 0.3
BATCH_SIZE = 64 # Larger batch for stability
LEARNING_RATE = 0.001

# --- Load Data ---
fill_gas = 'H2'
print(f"--- Loading {fill_gas} Dataset for Optimized Raw CNN ---")
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
X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test = X_test[test_mask]
y_test = y_test[test_mask]

# --- Shuffle ---
X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

# --- Preprocessing (Raw Spectra) ---
print("--- Using Raw Spectra (Scaled) ---")
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

# Reshape for CNN
X_train_processed = X_train_processed.reshape(X_train_processed.shape[0], X_train_processed.shape[1], 1)
X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], X_test_processed.shape[1], 1)
input_shape = (X_train_processed.shape[1], 1)

# --- Build Optimized Model ---
print("--- Building Optimized Raw CNN Model ---")
print("Improvements: GaussianNoise, ReduceLROnPlateau, 3 Deep Layers")

model = Sequential([
    # Add Gaussian Noise to inputs to prevent overfitting on specific noise patterns
    GaussianNoise(0.05, input_shape=input_shape),
    
    # Layer 1
    Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(DROPOUT_RATE),
    
    # Layer 2
    Conv1D(filters=FILTERS*2, kernel_size=KERNEL_SIZE, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(DROPOUT_RATE),
    
    # Layer 3 (Deeper)
    Conv1D(filters=FILTERS*4, kernel_size=KERNEL_SIZE, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(DROPOUT_RATE),

    Flatten(),
    Dense(100),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,       
    patience=5,       
    min_lr=0.00001,   
    verbose=1
)

# --- Train ---
history = model.fit(
    X_train_processed, y_train,
    epochs=150, 
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# --- Evaluate ---
loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"\n--- Optimized Raw CNN Results ---")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_proba = model.predict(X_test_processed)
y_pred = (y_pred_proba > 0.5).astype(int)

report_str = classification_report(y_test, y_pred, target_names=['Non-Bio (0)', 'Bio (1)'])
print(report_str)

report_filename = os.path.join(results_path, f'{fill_gas}_raw_cnn_optimized_report.txt')
with open(report_filename, 'w') as f:
    f.write(f"Optimized Raw CNN Results\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Params: Filters={FILTERS}, Kernel={KERNEL_SIZE}, Batch={BATCH_SIZE}, InitLR={LEARNING_RATE}\n")
    f.write(f"Extras: GaussianNoise(0.05), ReduceLROnPlateau, 3 Conv Layers\n\n")
    f.write(report_str)

cm = confusion_matrix(y_test, y_pred)
cm_filename = os.path.join(results_path, f'{fill_gas}_raw_cnn_optimized_confusion_matrix.png')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Optimized Raw CNN Confusion Matrix (Acc={accuracy:.4f})')
plt.savefig(cm_filename)
plt.close()
print(f"Results saved to {report_filename}")
