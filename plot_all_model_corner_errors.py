import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D
import corner
import os
import re
import gc

# --- Configuration ---
FILL_GAS = "H2"
TRAIN_FILE = f"multirex_spectra_{FILL_GAS}_train.parquet"
TEST_FILE = f"multirex_spectra_{FILL_GAS}_test.parquet"
RESULTS_DIR = "final_results/plots"
PARAMS_TO_PLOT = ['p_radius', 'p_mass', 's temperature', 'atm temperature']
LABELS = ['Planet Radius (R_earth)', 'Planet Mass (M_earth)', 'Star Temp (K)', 'Atmosphere Temp (K)']

# --- Load and Prepare Data ---
print("--- Loading and Preparing Data ---")
df_train = pd.read_parquet(TRAIN_FILE)
df_test = pd.read_parquet(TEST_FILE)

df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$")
cols = [c for c in df_train.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]

X_train_raw = df_train[cols].values
y_train = df_train['label'].values
X_test_raw = df_test[cols].values
y_test = df_test['label'].values

# --- Preprocessing (PCA) ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s = scaler.transform(X_test_raw)

pca = PCA()
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

# Feature Sets
X_train_clean = X_train_p[:, 2:102] # For XGB, RF, CNN
X_test_clean = X_test_p[:, 2:102]
X_train_mlp = X_train_p[:, 0:100] # For MLP
X_test_mlp = X_test_p[:, 0:100]
X_train_cnn = X_train_clean.reshape(X_train_clean.shape[0], X_train_clean.shape[1], 1)
X_test_cnn = X_test_clean.reshape(X_test_clean.shape[0], X_test_clean.shape[1], 1)


# --- Model Training Functions ---
def train_and_predict_xgb():
    print("--- Training XGBoost ---")
    model = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1, eval_metric='logloss')
    model.fit(X_train_clean, y_train)
    return model.predict(X_test_clean)

def train_and_predict_rf():
    print("--- Training Random Forest ---")
    model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train_clean, y_train)
    return model.predict(X_test_clean)

def train_and_predict_mlp():
    print("--- Training MLP ---")
    model = Sequential([
        Flatten(input_shape=(100,)), Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train_mlp, y_train, epochs=100, batch_size=128, verbose=0)
    return (model.predict(X_test_mlp, verbose=0) > 0.5).astype(int).flatten()

def train_and_predict_cnn():
    print("--- Training CNN ---")
    model = Sequential([
        Conv1D(32, 3, input_shape=(100, 1), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
        Conv1D(64, 3, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
        Flatten(), Dense(100), BatchNormalization(), Activation('relu'), Dropout(0.5), Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, verbose=0)
    return (model.predict(X_test_cnn, verbose=0) > 0.5).astype(int).flatten()

# --- Main Loop ---
models_to_run = {
    "XGBoost": train_and_predict_xgb,
    "RandomForest": train_and_predict_rf,
    "MLP": train_and_predict_mlp,
    "CNN": train_and_predict_cnn
}

for name, train_func in models_to_run.items():
    gc.collect() # Clean up memory before each run
    y_pred = train_func()
    
    # Identify errors
    correct_mask = (y_pred == y_test)
    incorrect_mask = ~correct_mask
    params_correct = df_test[correct_mask][PARAMS_TO_PLOT].values
    params_incorrect = df_test[incorrect_mask][PARAMS_TO_PLOT].values

    # Generate Plot
    print(f"--- Generating Corner Plot for {name} ---")
    figure = plt.figure(figsize=(15, 15))
    corner.corner(params_correct, fig=figure, labels=LABELS, color='navy', plot_contours=True, smooth=1.0, hist_kwargs={'density': True, 'color': 'navy'})
    corner.corner(params_incorrect, fig=figure, labels=LABELS, color='crimson', plot_contours=True, smooth=1.0, hist_kwargs={'density': True, 'color': 'crimson'})
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='navy', lw=4, label='Correct'), Line2D([0], [0], color='crimson', lw=4, label='Incorrect')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
    plt.suptitle(f"Corner Plot of Prediction Errors ({name})", fontsize=20)
    
    filename = os.path.join(RESULTS_DIR, f'corner_plot_errors_{name.lower()}.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved to: {filename}")

print("\n--- All Corner Plots Generated ---")
