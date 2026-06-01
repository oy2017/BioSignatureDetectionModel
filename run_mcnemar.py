import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GaussianNoise
from tensorflow.keras.optimizers import Adam
from scipy.stats import chi2 as chi2_dist
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

# Models
predictions = {}

# 1. XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.2, subsample=1.0, random_state=SEED, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train_pca, y_train)
predictions['XGBoost'] = xgb.predict(X_test_pca)

# 2. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, min_samples_split=5, class_weight='balanced', random_state=SEED, n_jobs=-1)
rf.fit(X_train_pca, y_train)
predictions['RandomForest'] = rf.predict(X_test_pca)

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
predictions['MLP'] = (mlp.predict(X_test_mlp, verbose=0).flatten() > 0.5).astype(int)

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
predictions['CNN'] = (cnn.predict(X_test_cnn, verbose=0).flatten() > 0.5).astype(int)

# McNemar's Test
def mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = (y_true == y_pred_a)
    correct_b = (y_true == y_pred_b)
    n12 = np.sum(correct_a & ~correct_b)
    n21 = np.sum(~correct_a & correct_b)
    if n12 + n21 == 0:
        return 0, 1.0
    chi2 = ((abs(n12 - n21) - 1)**2) / (n12 + n21)
    p = chi2_dist.sf(chi2, 1)
    return chi2, p

models = ['XGBoost', 'RandomForest', 'MLP', 'CNN']

print("\n--- McNemar's Test Pairwise Comparisons ---")
for i in range(len(models)):
    for j in range(i+1, len(models)):
        m1, m2 = models[i], models[j]
        chi2, p = mcnemar_test(y_test, predictions[m1], predictions[m2])
        significance = "Significant" if p < 0.05 else "Not Significant"
        print(f"{m1} vs {m2}: chi2={chi2:.4f}, p-value={p:.4e} ({significance})")
