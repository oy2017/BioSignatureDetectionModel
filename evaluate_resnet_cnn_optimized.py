import os
# Set environment variable to avoid OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import re
import argparse
import random
from sklearn.utils import shuffle

# --- Set Seeds ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

results_path = 'final_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Evaluate Optimized ResNet on Raw Spectra.")
parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.") # Smaller batch for better generalization
args = parser.parse_args()

fill_gas = args.fill_gas.upper()

# --- 1. Load Data ---
print(f"--- Loading {fill_gas} Raw Data for Optimized ResNet ---")
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

# --- Sort Columns (Ascending Wavelength) ---
float_pattern = re.compile(r"^-?\d+\.\d+$") 
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
# Sort by wavelength (numerical value)
spectral_cols = sorted(spectral_cols, key=float)
print("--- Columns Sorted Ascending ---")

X_train = df_train[spectral_cols].values
y_train = df_train['label'].values
X_test = df_test[spectral_cols].values
y_test = df_test['label'].values

X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

# --- 2. Preprocessing ---

# 2.1 Gaussian Smoothing (Noise Reduction)
def smooth_spectra(X, kernel_size=3):
    # Simple moving average / boxcar for simplicity, or gaussian
    # Boxcar: [1/3, 1/3, 1/3]
    kernel = np.ones(kernel_size) / kernel_size
    # Apply convolution along axis 1 (wavelengths)
    # Mode 'same' keeps shape
    X_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=X)
    return X_smooth

print("--- Applying Smoothing (Kernel=3) ---")
X_train = smooth_spectra(X_train)
X_test = smooth_spectra(X_test)

# 2.2 Scaling
print("--- Scaling Data ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape
X_train_processed = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_processed = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

input_shape = (X_train_processed.shape[1], 1)

# --- 3. Build Simplified ResNet ---
# Reduced filters to prevent overfitting on small dataset
def residual_block(x, filters, kernel_size=7):
    shortcut = x
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

inputs = Input(shape=input_shape)

# Initial Conv
x = Conv1D(filters=32, kernel_size=7, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Block 1 (32 Filters)
x = residual_block(x, filters=32, kernel_size=7)
x = MaxPooling1D(pool_size=2)(x)

# Block 2 (64 Filters)
x = residual_block(x, filters=64, kernel_size=7)
x = MaxPooling1D(pool_size=2)(x)

# Block 3 (128 Filters)
x = residual_block(x, filters=128, kernel_size=7)
x = GlobalAveragePooling1D()(x)

# Head
x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

# Lower LR
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. Train ---
print(f"--- Training Optimized ResNet ---")
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
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

report_str = classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio'])
print("\nClassification Report:")
print(report_str)

report_filename = os.path.join(results_path, f'{fill_gas}_resnet_optimized_report.txt')
with open(report_filename, 'w') as f:
    f.write(report_str)
