import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# --- Configuration ---
FILL_GAS = "H2"
SEED = 42
RESULTS_DIR = "final_results/physics_comparison"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Physics Constants
KB = 1.380649e-23
AMU = 1.660539e-27
G = 6.674e-11
MEarth = 5.972e24
REarth = 6.371e6

# --- Data Loading ---
def calculate_scale_height(row):
    T = row['atm temperature']
    mu = 2.3 * AMU if FILL_GAS == 'H2' else 28.0 * AMU
    M = row['p_mass'] * MEarth
    R = row['p_radius'] * REarth
    g = (G * M) / (R**2)
    H = (KB * T) / (mu * g)
    return H / 1000.0

def get_data(train_file, test_file):
    print(f"--- Loading Training Data: {train_file} ---")
    df_train = pd.read_parquet(train_file)
    print(f"--- Loading Test Data: {test_file} ---")
    df_test = pd.read_parquet(test_file)
    
    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    
    # Extract Spectral Features
    import re
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    cols = [c for c in df_train.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]
    cols_sorted = sorted(cols, key=float)
    
    X_train = df_train[cols_sorted].values
    X_test = df_test[cols_sorted].values
    
    # Calculate Physics Params
    print("--- Calculating Physics Parameters ---")
    df_test['Scale Height (km)'] = df_test.apply(calculate_scale_height, axis=1)
    df_test['log_H2O'] = df_test['atm H2O']
    df_test['log_CO2'] = df_test['atm CO2']
    
    # Use dictionary access for column renaming robustness
    params = pd.DataFrame()
    params['Planet Radius (Re)'] = df_test['p_radius']
    params['Scale Height (km)'] = df_test['Scale Height (km)']
    params['log(H2O)'] = df_test['log_H2O']
    params['log(CO2)'] = df_test['log_CO2']
    
    return X_train, y_train, X_test, y_test, params

# --- Model Builders ---
def train_xgboost(X_tr, y_tr):
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.2, 
                          subsample=1.0, n_jobs=-1, random_state=SEED, eval_metric='logloss')
    model.fit(X_tr, y_tr)
    return model

def train_rf(X_tr, y_tr):
    model = RandomForestClassifier(n_estimators=300, max_depth=None, 
                                   class_weight='balanced', n_jobs=-1, random_state=SEED)
    model.fit(X_tr, y_tr)
    return model

def build_mlp(input_dim):
    # Deep & Wide (Best Config)
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512), BatchNormalization(), Activation('relu'), Dropout(0.4),
        Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.4),
        Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_resnet(input_shape):
    def res_block(x, f, k=7):
        s = x
        x = Conv1D(f, k, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(f, k, padding='same')(x)
        x = BatchNormalization()(x)
        if s.shape[-1] != f: s = Conv1D(f, 1, padding='same')(s)
        x = Add()([x, s])
        x = Activation('relu')(x)
        return x
    
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = res_block(x, 32)
    x = MaxPooling1D(2)(x)
    x = res_block(x, 64)
    x = MaxPooling1D(2)(x)
    x = res_block(x, 128)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Main Logic ---
def run_analysis(dataset_name, train_file, test_file):
    print(f"\n=== Processing Dataset: {dataset_name} ===")
    X_tr_raw, y_tr, X_te_raw, y_te, params = get_data(train_file, test_file)
    
    # 1. Feature Engineering (Standard + PCA)
    print("--- Preprocessing (StandardScaler + PCA 2-102) ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_raw)
    X_te_s = scaler.transform(X_te_raw)
    
    pca = PCA()
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    
    # Feature Set 1: PCA 2-102 (For XGB, RF, CNN-proxy if needed)
    X_tr_clean = X_tr_p[:, 2:102]
    X_te_clean = X_te_p[:, 2:102]
    
    # Feature Set 2: PCA 0-100 (For MLP - it prefers 0-100)
    X_tr_mlp = X_tr_p[:, 0:100]
    X_te_mlp = X_te_p[:, 0:100]
    scaler_mlp = StandardScaler() # Re-scale PCA for MLP
    X_tr_mlp = scaler_mlp.fit_transform(X_tr_mlp)
    X_te_mlp = scaler_mlp.transform(X_te_mlp)
    
    # Feature Set 3: Raw Scaled (For CNN)
    X_tr_cnn = X_tr_s.reshape(X_tr_s.shape[0], X_tr_s.shape[1], 1)
    X_te_cnn = X_te_s.reshape(X_te_s.shape[0], X_te_s.shape[1], 1)

    # --- Training & Prediction ---
    models = {}
    
    # XGBoost
    print("--- Training XGBoost ---")
    xgb = train_xgboost(X_tr_clean, y_tr)
    models['XGBoost'] = xgb.predict(X_te_clean)
    
    # Random Forest
    print("--- Training Random Forest ---")
    rf = train_rf(X_tr_clean, y_tr)
    models['RandomForest'] = rf.predict(X_te_clean)
    
    # MLP
    print("--- Training MLP (DeepWide) ---")
    mlp = build_mlp(100)
    mlp.fit(X_tr_mlp, y_tr, epochs=50, batch_size=128, validation_split=0.2, verbose=0,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    models['MLP'] = (mlp.predict(X_te_mlp, verbose=0) > 0.5).astype(int).flatten()
    
    # CNN
    print("--- Training CNN (ResNet) ---")
    cnn = build_resnet((X_tr_cnn.shape[1], 1))
    cnn.fit(X_tr_cnn, y_tr, epochs=30, batch_size=32, validation_split=0.2, verbose=0,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    models['CNN'] = (cnn.predict(X_te_cnn, verbose=0) > 0.5).astype(int).flatten()
    
    # --- Plotting ---
    print("--- Generating Plots ---")
    for m_name, preds in models.items():
        acc = accuracy_score(y_te, preds)
        print(f"{m_name} Accuracy: {acc:.4f}")
        
        # Calculate Error
        error_mask = (preds != y_te).astype(int)
        
        for p_name in params.columns:
            vals = params[p_name]
            
            # Adaptive Binning
            df_plot = pd.DataFrame({'val': vals, 'error': error_mask})
            try:
                df_plot['bin'] = pd.qcut(df_plot['val'], q=10, duplicates='drop')
            except:
                df_plot['bin'] = pd.cut(df_plot['val'], bins=10)
                
            bin_stats = df_plot.groupby('bin', observed=True).agg(
                x=('val', 'mean'), y=('error', 'mean')
            ).reset_index()
            
            plt.figure(figsize=(8, 6))
            sns.lineplot(data=bin_stats, x='x', y='y', marker='o', linewidth=2.5, color='royalblue')
            
            plt.title(f'{dataset_name}: {m_name} Error vs {p_name}', fontsize=12, fontweight='bold')
            plt.xlabel(p_name)
            plt.ylabel('Error Rate')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0) # Standardize Y-axis
            
            safe_p = p_name.lower().replace(' ', '').replace('(', '').replace(')', '')
            fname = f"{dataset_name}_{m_name}_error_vs_{safe_p}.png"
            plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
            plt.close()

if __name__ == "__main__":
    # Run 1: Old Data (Original Train, Original Test 1)
    run_analysis("OldData", 
                 f"multirex_spectra_{FILL_GAS}_train.parquet", 
                 f"multirex_spectra_{FILL_GAS}_test_set_1.parquet")
    
    # Run 2: New Data (New Train v2, New Test v2)
    v2_train = f"multirex_spectra_{FILL_GAS}_train_v2.parquet"
    v2_test = f"multirex_spectra_{FILL_GAS}_test_v2.parquet"
    
    if os.path.exists(v2_train) and os.path.exists(v2_test):
        run_analysis("NewData_v2", v2_train, v2_test)
    else:
        print(f"\nSkipping v2 analysis: v2 files not found.")