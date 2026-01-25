import os
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
parser = argparse.ArgumentParser(description="Evaluate CNN on the DERIVATIVE of raw spectra.")
parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
args = parser.parse_args()

fill_gas = args.fill_gas.upper()

# --- 1. Load Data ---
print(f"--- Loading {fill_gas} Data for Derivative CNN ---")
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]

X_train = df_train[spectral_cols].values
y_train = df_train['label'].values
X_test = df_test[spectral_cols].values
y_test = df_test['label'].values

X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

# --- 2. Calculate Derivative (Differencing) ---
print("--- Calculating 1st Derivative of Spectra ---")
# np.diff calculates the difference between adjacent elements: out[i] = a[i+1] - a[i]
# This effectively removes the constant offset and linear trend (high variance background).
X_train_diff = np.diff(X_train, axis=1)
X_test_diff = np.diff(X_test, axis=1)

# --- 3. Scale the Derivative Data ---
print("--- Scaling Derivative Data ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_diff)
X_test_scaled = scaler.transform(X_test_diff)

# Reshape for CNN (samples, timesteps, 1)
# Note: shape[1] is now (original_length - 1)
X_train_processed = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_processed = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

input_shape = (X_train_processed.shape[1], 1)

# --- 4. Build Standard 2-Layer CNN ---
print(f"--- Building CNN with input shape {input_shape} ---")
model = Sequential([
    # Layer 1
    Conv1D(filters=64, kernel_size=5, input_shape=input_shape, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    # Layer 2
    Conv1D(filters=128, kernel_size=5, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    # Head
    Flatten(),
    Dense(100),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --- 5. Train ---
print(f"--- Training Derivative CNN for {args.epochs} epochs ---")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_processed, y_train,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# --- 6. Evaluate ---
loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = (model.predict(X_test_processed) > 0.5).astype(int)

# --- 7. Report ---
report_name = f'derivative_cnn_report.txt'
with open(os.path.join(results_path, report_name), 'w') as f:
    f.write(classification_report(y_test, y_pred))

print(f"\n--- Derivative CNN Analysis Complete ---")
print(classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio']))
