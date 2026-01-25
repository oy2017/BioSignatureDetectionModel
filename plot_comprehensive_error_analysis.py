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
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Configuration ---
FILL_GAS = "H2"
PCA_COMPONENTS = 100
SEED = 42

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(SEED)
tf.random.set_seed(SEED)

def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print(f"{name} Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}\n")

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
    
    params_test = df_test[['atm temperature', 'p_radius', 'p_mass', 's temperature']].copy()
    params_test.columns = ['Planet Temp', 'Planet Radius', 'Planet Mass', 'Star Temp']
    
    return X_train_raw, y_train, X_test_raw, y_test, params_test

def get_pca_data(X_train_raw, X_test_raw):
    print("--- Preparing PCA Data (Components 2-102) ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_te_s = scaler.transform(X_test_raw)
    
    pca = PCA()
    X_tr_pca_full = pca.fit_transform(X_tr_s)
    X_te_pca_full = pca.transform(X_te_s)
    
    # Slice 2:102 (Effective Signal)
    X_tr_pca = X_tr_pca_full[:, 2:102]
    X_te_pca = X_te_pca_full[:, 2:102]
    
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

def calculate_error_rates(y_true, y_pred, param_values, num_bins=10):
    bins = np.linspace(param_values.min(), param_values.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    errors = []
    for i in range(num_bins):
        mask = (param_values >= bins[i]) & (param_values < bins[i+1])
        if mask.sum() > 0:
            errors.append(1 - accuracy_score(y_true[mask], y_pred[mask]))
        else:
            errors.append(np.nan)
    return bin_centers, errors

def main():
    X_tr_raw, y_train, X_te_raw, y_test, params = get_data()
    X_tr_pca, X_te_pca = get_pca_data(X_tr_raw, X_te_raw)
    
    predictions = {}

    # 1. MLP (Load Saved)
    mlp_path = 'final_results/H2_best_mlp_model.keras'
    if os.path.exists(mlp_path):
        print("--- Loading Best MLP ---")
        mlp = load_model(mlp_path)
        # Note: Must recreate the exact input pipeline for the MLP (PCA 0-100)
        scaler_mlp = StandardScaler()
        X_tr_s = scaler_mlp.fit_transform(X_tr_raw)
        X_te_s = scaler_mlp.transform(X_te_raw)
        pca_mlp = PCA(n_components=100)
        X_tr_p = pca_mlp.fit_transform(X_tr_s)
        X_te_p = pca_mlp.transform(X_te_s)
        scaler_p2 = StandardScaler()
        scaler_p2.fit(X_tr_p)
        X_te_mlp = scaler_p2.transform(X_te_p)
        predictions['MLP (DeepWide)'] = (mlp.predict(X_te_mlp, verbose=0) > 0.5).astype(int).flatten()
        print_metrics("MLP (DeepWide)", y_test, predictions['MLP (DeepWide)'])
    
    # 2. XGBoost (Original)
    print("--- Training XGBoost (Original: LR=0.1, Est=150) ---")
    xgb1 = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=SEED, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    xgb1.fit(X_tr_pca, y_train)
    predictions['XGBoost (Original)'] = xgb1.predict(X_te_pca)
    print_metrics("XGBoost (Original)", y_test, predictions['XGBoost (Original)'])
    
    # 3. XGBoost (Aggressive)
    print("--- Training XGBoost (Aggressive: LR=0.2, Est=300) ---")
    xgb2 = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.2, subsample=1.0, random_state=SEED, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    xgb2.fit(X_tr_pca, y_train)
    predictions['XGBoost (Aggressive)'] = xgb2.predict(X_te_pca)
    print_metrics("XGBoost (Aggressive)", y_test, predictions['XGBoost (Aggressive)'])
    
    # 4. Random Forest
    print("--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=150, max_depth=None, class_weight='balanced', random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_pca, y_train)
    predictions['Random Forest'] = rf.predict(X_te_pca)
    print_metrics("Random Forest", y_test, predictions['Random Forest'])
    
    # 5. CNN (ResNet)
    print("--- Training CNN (ResNet) ---")
    scaler_cnn = StandardScaler()
    X_tr_cnn = scaler_cnn.fit_transform(X_tr_raw).reshape(-1, X_tr_raw.shape[1], 1)
    X_te_cnn = scaler_cnn.transform(X_te_raw).reshape(-1, X_te_raw.shape[1], 1)
    
    cnn = build_resnet(input_shape=(X_tr_raw.shape[1], 1))
    cnn.fit(X_tr_cnn, y_train, epochs=50, batch_size=32, validation_split=0.1, 
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
    predictions['CNN (ResNet)'] = (cnn.predict(X_te_cnn, verbose=0) > 0.5).astype(int).flatten()
    print_metrics("CNN (ResNet)", y_test, predictions['CNN (ResNet)'])

    # --- Plotting ---
    print("--- Generating Plots ---")
    plot_params = ['Planet Temp', 'Planet Radius', 'Planet Mass', 'Star Temp']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, p_name in enumerate(plot_params):
        ax = axes[i]
        vals = params[p_name].values
        for m_name, preds in predictions.items():
            centers, errors = calculate_error_rates(y_test, preds, vals)
            sns.lineplot(x=centers, y=errors, ax=ax, label=m_name, marker='o', linewidth=2)
        
        ax.set_title(f'Error Rate vs {p_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel(p_name, fontsize=12)
        ax.set_ylabel('Error Rate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_filename = 'final_results/comprehensive_error_analysis_physics_v2.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Analysis complete. Plot saved to {output_filename}")

if __name__ == "__main__":
    main()
