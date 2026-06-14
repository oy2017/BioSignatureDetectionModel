import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, Add, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# --- Configuration ---
FILL_GAS = "H2"
PCA_COMPONENTS = 100
SEED = 42
RESULTS_DIR = "final_results"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_data():
    print("--- Loading Data ---")
    df_train = pd.read_parquet(f'multirex_spectra_{FILL_GAS}_train.parquet')
    df_test = pd.read_parquet(f'multirex_spectra_{FILL_GAS}_test.parquet')
    
    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    spectral_cols_sorted = sorted(spectral_cols, key=float)
    
    X_train_raw = df_train[spectral_cols_sorted].values
    X_test_raw = df_test[spectral_cols_sorted].values
    
    # Extract Chemical Abundances
    params_test = pd.DataFrame({
        'CH4': df_test['atm CH4'].values,
        'O3': df_test['atm O3'].values
    })
    
    return X_train_raw, y_train, X_test_raw, y_test, params_test

def get_pca_data(X_train_raw, X_test_raw, start=0, end=102):
    print(f"--- Preparing PCA Data (Components {start}-{end-1}) ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_te_s = scaler.transform(X_test_raw)
    
    pca = PCA(n_components=102, random_state=42)
    X_tr_pca_full = pca.fit_transform(X_tr_s)
    X_te_pca_full = pca.transform(X_te_s)
    
    X_tr_pca = X_tr_pca_full[:, start:end]
    X_te_pca = X_te_pca_full[:, start:end]
    
    scaler_pca = StandardScaler()
    X_tr_final = scaler_pca.fit_transform(X_tr_pca)
    X_te_final = scaler_pca.transform(X_te_pca)
    
    return X_tr_final, X_te_final

def build_mlp():
    model = Sequential([
        Input(shape=(102,)),
        Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy')
    return model

def build_cnn():
    model = Sequential([
        Input(shape=(102, 1)),
        Conv1D(64, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
        Conv1D(128, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
        Flatten(),
        Dense(100), BatchNormalization(), Activation('relu'), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

def main():
    X_tr_raw, y_train, X_te_raw, y_test, params = get_data()
    X_tr_pca, X_te_pca = get_pca_data(X_tr_raw, X_te_raw) # Unified 0-102
    
    # Shuffle
    X_tr_pca, y_train_shuf = shuffle(X_tr_pca, y_train, random_state=SEED)
    
    predictions = {}
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 1. MLP
    print("--- Training MLP ---")
    mlp = build_mlp()
    mlp.fit(X_tr_pca, y_train_shuf, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es], verbose=0)
    predictions['MLP'] = (mlp.predict(X_te_pca, verbose=0) > 0.5).astype(int).flatten()
    
    # 2. XGBoost
    print("--- Training XGBoost ---")
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=SEED, n_jobs=-1, eval_metric='logloss')
    xgb.fit(X_tr_pca, y_train_shuf)
    predictions['XGBoost'] = xgb.predict(X_te_pca)
    
    # 3. Random Forest
    print("--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=2, max_depth=None, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_pca, y_train_shuf)
    predictions['Random Forest'] = rf.predict(X_te_pca)
    
    # 4. CNN
    print("--- Training CNN ---")
    cnn = build_cnn()
    cnn.fit(X_tr_pca.reshape(-1, 102, 1), y_train_shuf, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es], verbose=0)
    predictions['CNN'] = (cnn.predict(X_te_pca.reshape(-1, 102, 1), verbose=0) > 0.5).astype(int).flatten()

    # --- Plotting ---
    print("--- Generating Comparison Scatter Plots ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Thresholds
    CH4_THRESH = -6
    O3_THRESH = -7
    
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        
        # Identify Correct/Incorrect
        correct_mask = (y_pred == y_test)
        error_mask = (y_pred != y_test)
        
        # Plot Correct (Grey, Alpha)
        ax.scatter(params['CH4'][correct_mask], params['O3'][correct_mask], 
                   color='lightgrey', alpha=0.3, label='Correct', s=20)
        
        # Plot Error (Red)
        ax.scatter(params['CH4'][error_mask], params['O3'][error_mask], 
                   color='red', alpha=0.8, label='Error', s=30)
        
        # Draw Threshold Lines
        ax.axvline(x=CH4_THRESH, color='black', linestyle='--', alpha=0.6)
        ax.axhline(y=O3_THRESH, color='black', linestyle='--', alpha=0.6)
        
        ax.set_title(f'{model_name} Error Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('log(CH4)', fontsize=14)
        ax.set_ylabel('log(O3)', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        
    plt.tight_layout()
    output_filename = os.path.join(RESULTS_DIR, 'model_comparison_chemical_scatter.png')
    plt.savefig(output_filename, dpi=300)
    print(f"Comparison plot saved to {output_filename}")

if __name__ == "__main__":
    main()
