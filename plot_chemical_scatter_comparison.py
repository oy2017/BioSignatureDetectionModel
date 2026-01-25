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
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, Add
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

def get_pca_data(X_train_raw, X_test_raw, start=2, end=102):
    print(f"--- Preparing PCA Data (Components {start}-{end}) ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_te_s = scaler.transform(X_test_raw)
    
    pca = PCA()
    X_tr_pca_full = pca.fit_transform(X_tr_s)
    X_te_pca_full = pca.transform(X_te_s)
    
    X_tr_pca = X_tr_pca_full[:, start:end]
    X_te_pca = X_te_pca_full[:, start:end]
    
    scaler_pca = StandardScaler()
    X_tr_final = scaler_pca.fit_transform(X_tr_pca)
    X_te_final = scaler_pca.transform(X_te_pca)
    
    return X_tr_final, X_te_final

def build_resnet(input_shape):
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
    x = Conv1D(filters=32, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = residual_block(x, filters=32, kernel_size=7)
    x = MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, filters=64, kernel_size=7)
    x = MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, filters=128, kernel_size=7)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    X_tr_raw, y_train, X_te_raw, y_test, params = get_data()
    X_tr_pca, X_te_pca = get_pca_data(X_tr_raw, X_te_raw) # 2-102
    
    predictions = {}

    # 1. MLP (Load Saved)
    mlp_path = os.path.join(RESULTS_DIR, f'{FILL_GAS}_best_mlp_model.keras')
    if os.path.exists(mlp_path):
        print("--- Loading Best MLP ---")
        mlp = load_model(mlp_path)
        # Recreate 0-100 pipeline for MLP
        X_tr_pca0, X_te_pca0 = get_pca_data(X_tr_raw, X_te_raw, start=0, end=100)
        predictions['MLP (DeepWide)'] = (mlp.predict(X_te_pca0, verbose=0) > 0.5).astype(int).flatten()
    
    # 2. XGBoost (Aggressive)
    print("--- Training XGBoost (Aggressive) ---")
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.2, subsample=1.0, random_state=SEED, n_jobs=-1, eval_metric='logloss')
    xgb.fit(X_tr_pca, y_train)
    predictions['XGBoost (Aggressive)'] = xgb.predict(X_te_pca)
    
    # 3. Random Forest (Tuned)
    print("--- Training Random Forest (Tuned) ---")
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1, min_samples_split=2, class_weight='balanced', random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_pca, y_train)
    predictions['Random Forest (Tuned)'] = rf.predict(X_te_pca)
    
    # 4. CNN (ResNet)
    print("--- Training CNN (ResNet) ---")
    scaler_cnn = StandardScaler()
    X_tr_cnn = scaler_cnn.fit_transform(X_tr_raw).reshape(-1, X_tr_raw.shape[1], 1)
    X_te_cnn = scaler_cnn.transform(X_te_raw).reshape(-1, X_te_raw.shape[1], 1)
    cnn = build_resnet(input_shape=(X_tr_raw.shape[1], 1))
    cnn.fit(X_tr_cnn, y_train, epochs=30, batch_size=32, validation_split=0.1, 
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
    predictions['CNN (ResNet)'] = (cnn.predict(X_te_cnn, verbose=0) > 0.5).astype(int).flatten()

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
        ax.axvline(x=CH4_THRESH, color='black', linestyle='--', alpha=0.6, label='CH4 Threshold')
        ax.axhline(y=O3_THRESH, color='black', linestyle='--', alpha=0.6, label='O3 Threshold')
        
        ax.set_title(f'{model_name} Error Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('log(CH4)', fontsize=12)
        ax.set_ylabel('log(O3)', fontsize=12)
        ax.legend(loc='upper left')
        
    plt.tight_layout()
    output_filename = os.path.join(RESULTS_DIR, 'model_comparison_chemical_scatter.png')
    plt.savefig(output_filename, dpi=300)
    print(f"Comparison plot saved to {output_filename}")

if __name__ == "__main__":
    main()
