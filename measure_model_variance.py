import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
import re
import random

# --- Set Seeds ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILES = [f"multirex_spectra_{FILL_GAS}_test_set_{i}.parquet" for i in range(1, 6)]

# --- Helper Functions ---
def load_data(file_path):
    df = pd.read_parquet(file_path)
    df['label'] = df['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
    
    float_pattern = re.compile(r"^-?\d+\.\d+$")
    cols = [c for c in df.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]
    
    X = df[cols].values
    y = df['label'].values
    return X, y

# --- 1. Prepare Training Data ---
print("--- Loading Training Data ---")
X_train_raw, y_train = load_data(TRAIN_FILE)
X_train_raw, y_train = shuffle(X_train_raw, y_train, random_state=SEED)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

# PCA Full
pca = PCA()
X_train_pca_full = pca.fit_transform(X_train_scaled)

# Feature Sets
# 1. MLP: PCA 0-100 (Scaled)
X_train_mlp = X_train_pca_full[:, 0:100]
scaler_mlp = StandardScaler()
X_train_mlp = scaler_mlp.fit_transform(X_train_mlp)

# 2. CNN/RF/XGB: PCA 2-102 (Scaled)
X_train_clean = X_train_pca_full[:, 2:102]
scaler_clean = StandardScaler()
X_train_clean = scaler_clean.fit_transform(X_train_clean)

# Reshape for CNN
X_train_cnn = X_train_clean.reshape(X_train_clean.shape[0], X_train_clean.shape[1], 1)

# --- 2. Train Models ---

# --- MLP ---
print("\n--- Training Best MLP ---")
model_mlp = Sequential([
    Flatten(input_shape=(100,)),
    Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.3),
    Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
    Dense(64), BatchNormalization(), Activation('relu'), Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', metrics=['accuracy'])
model_mlp.fit(X_train_mlp, y_train, epochs=100, batch_size=128, validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)

# --- CNN ---
print("--- Training Best CNN ---")
cnn_input_shape = (100, 1)
model_cnn = Sequential([
    Conv1D(64, 5, input_shape=cnn_input_shape, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
    Conv1D(128, 5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
    Flatten(),
    Dense(100), BatchNormalization(), Activation('relu'), Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train_cnn, y_train, epochs=100, batch_size=64, validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)

# --- Random Forest ---
print("--- Training Best Random Forest ---")
model_rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1, class_weight='balanced')
model_rf.fit(X_train_clean, y_train)

# --- XGBoost ---
print("--- Training Best XGBoost ---")
model_xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED, n_jobs=-1, eval_metric='logloss')
model_xgb.fit(X_train_clean, y_train)


# --- 3. Evaluate on 5 Test Sets ---
results = {'MLP': [], 'CNN': [], 'RandomForest': [], 'XGBoost': []}

print("\n--- Evaluating on 5 Test Sets ---")
for i, test_file in enumerate(TEST_FILES):
    print(f"Set {i+1}: {test_file}")
    
    # Load and Preprocess Test Set
    X_test_raw, y_test = load_data(test_file)
    X_test_scaled = scaler.transform(X_test_raw)
    X_test_pca_full = pca.transform(X_test_scaled) # Use training PCA
    
    # Feature Sets
    X_test_mlp = scaler_mlp.transform(X_test_pca_full[:, 0:100])
    X_test_clean = scaler_clean.transform(X_test_pca_full[:, 2:102])
    X_test_cnn = X_test_clean.reshape(X_test_clean.shape[0], X_test_clean.shape[1], 1)
    
    # Helper to store results
    def store_res(name, y_pred):
        results[name].append({
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0)
        })

    # Predict MLP
    store_res('MLP', (model_mlp.predict(X_test_mlp) > 0.5).astype(int))
    
    # Predict CNN
    store_res('CNN', (model_cnn.predict(X_test_cnn) > 0.5).astype(int))
    
    # Predict RF
    store_res('RandomForest', model_rf.predict(X_test_clean))
    
    # Predict XGB
    store_res('XGBoost', model_xgb.predict(X_test_clean))

# --- 4. Calculate Stats ---
print("\n--- Final Statistics ---")
stats_report = []

for model_name, metrics_list in results.items():
    print(f"\nModel: {model_name}")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        values = [m[metric] for m in metrics_list]
        print(f"  {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

# Save to file
report_path = 'final_results/model_variance_report_all.txt'
with open(report_path, 'w') as f:
    f.write("Model Variance Report (5 Independent Test Sets)\n")
    f.write("===============================================\n\n")
    for model_name, metrics_list in results.items():
        f.write(f"Model: {model_name}\n")
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
            values = [m[metric] for m in metrics_list]
            f.write(f"  {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})\n")
        f.write("\n")

print(f"\nReport saved to {report_path}")