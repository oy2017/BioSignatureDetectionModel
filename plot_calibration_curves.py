import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GaussianNoise, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED = 42
np.random.seed(SEED)

# Load data
fill_gas = 'H2'
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X_train_raw = df_train[spectral_cols].values
X_test_raw = df_test[spectral_cols].values

y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values

# Filter Invalid Values (> 1.0)
train_mask = (X_train_raw <= 1.0).all(axis=1)
test_mask = (X_test_raw <= 1.0).all(axis=1)
X_train_raw = X_train_raw[train_mask]
y_train = y_train[train_mask]
X_test_raw = X_test_raw[test_mask]
y_test = y_test[test_mask]

# Preprocessing
scaler_raw = StandardScaler()
X_train_scaled_raw = scaler_raw.fit_transform(X_train_raw)
X_test_scaled_raw = scaler_raw.transform(X_test_raw)

pca = PCA(n_components=102, random_state=SEED)
X_train_pca_full = pca.fit_transform(X_train_scaled_raw)
X_test_pca_full = pca.transform(X_test_scaled_raw)

# 1. Trees use PC 2-101
X_train_tree = X_train_pca_full[:, 2:102]
X_test_tree = X_test_pca_full[:, 2:102]

# 2. Neural Networks use PC 0-101
X_train_nn = X_train_pca_full[:, 0:102]
X_test_nn = X_test_pca_full[:, 0:102]
scaler_nn = StandardScaler()
X_train_nn = scaler_nn.fit_transform(X_train_nn)
X_test_nn = scaler_nn.transform(X_test_nn)

# Shuffle training data
X_train_tree, y_train_tree = shuffle(X_train_tree, y_train, random_state=SEED)
X_train_nn, y_train_nn = shuffle(X_train_nn, y_train, random_state=SEED)

probabilities = {}

# 1. XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=SEED, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train_tree, y_train_tree)
probabilities['XGBoost'] = xgb.predict_proba(X_test_tree)[:, 1]

# 2. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_depth=None, random_state=SEED, n_jobs=-1)
rf.fit(X_train_tree, y_train_tree)
probabilities['Random Forest'] = rf.predict_proba(X_test_tree)[:, 1]

# 3. MLP
print("Training MLP...")
mlp = Sequential([
    Input(shape=(102,)),
    Dense(512), BatchNormalization(), Activation('relu'), Dropout(0.4),
    Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.4),
    Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.4),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mlp.fit(X_train_nn, y_train_nn, batch_size=128, epochs=100, validation_split=0.2, callbacks=[es], verbose=0)
probabilities['MLP'] = mlp.predict(X_test_nn, verbose=0).flatten()

# 4. CNN
print("Training 1D-CNN...")
cnn = Sequential([
    Input(shape=(102, 1)),
    Conv1D(filters=64, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
    Conv1D(filters=128, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
    Flatten(),
    Dense(100), BatchNormalization(), Activation('relu'), Dropout(0.5),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
cnn.fit(X_train_nn.reshape(-1, 102, 1), y_train_nn, batch_size=64, epochs=100, validation_split=0.2, callbacks=[es], verbose=0)
probabilities['CNN'] = cnn.predict(X_test_nn.reshape(-1, 102, 1), verbose=0).flatten()

# Plotting Calibration Curves
print("Plotting Calibration Curves...")
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, probs in probabilities.items():
    display = CalibrationDisplay.from_predictions(
        y_test, probs, n_bins=10, name=model_name, ax=ax
    )

ax.set_title("Calibration Curves (Reliability Diagrams) for Biosignature Detection")
ax.grid(True, linestyle='--', alpha=0.7)

output_path = 'final_results/calibration_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Calibration curves saved to {output_path}")

# Brier Scores
print("\n--- Brier Scores (lower is better) ---")
for model_name, probs in probabilities.items():
    brier = brier_score_loss(y_test, probs)
    print(f"{model_name}: {brier:.4f}")
