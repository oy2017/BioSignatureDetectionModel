import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GaussianNoise
from tensorflow.keras.optimizers import Adam
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
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
X_train_scaled = scaler_raw.fit_transform(X_train_raw)
X_test_scaled = scaler_raw.transform(X_test_raw)

pca = PCA(n_components=102)
X_train_pca_full = pca.fit_transform(X_train_scaled)
X_test_pca_full = pca.transform(X_test_scaled)

X_train_pca = X_train_pca_full[:, 2:102]
X_test_pca = X_test_pca_full[:, 2:102]

probabilities = {}

# 1. XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.2, subsample=1.0, random_state=SEED, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train_pca, y_train)
probabilities['XGBoost'] = xgb.predict_proba(X_test_pca)[:, 1]

# 2. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, min_samples_split=5, class_weight='balanced', random_state=SEED, n_jobs=-1)
rf.fit(X_train_pca, y_train)
probabilities['Random Forest'] = rf.predict_proba(X_test_pca)[:, 1]

# 3. MLP
print("Training MLP...")
X_train_mlp = X_train_pca_full[:, 0:100]
X_test_mlp = X_test_pca_full[:, 0:100]

mlp = Sequential([
    Input(shape=(100,)),
    Dense(512), BatchNormalization(), Activation('relu'), Dropout(0.4),
    Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.4),
    Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.4),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_train_mlp, y_train, batch_size=128, epochs=60, verbose=0)
probabilities['MLP'] = mlp.predict(X_test_mlp, verbose=0).flatten()

# 4. CNN
print("Training 1D-CNN...")
X_train_cnn = X_train_scaled.reshape(-1, len(spectral_cols), 1)
X_test_cnn = X_test_scaled.reshape(-1, len(spectral_cols), 1)

cnn = Sequential([
    Input(shape=(len(spectral_cols), 1)),
    GaussianNoise(0.05),
    Conv1D(32, 5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
    Conv1D(64, 5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
    Conv1D(128, 5, padding='same'), BatchNormalization(), Activation('relu'), GlobalAveragePooling1D(), Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_cnn, y_train, batch_size=64, epochs=50, verbose=0)
probabilities['CNN'] = cnn.predict(X_test_cnn, verbose=0).flatten()

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
