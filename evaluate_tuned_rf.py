import os
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
    print(f"\n{name} Metrics:")
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
    
    X_train_raw = df_train[spectral_cols].values
    X_test_raw = df_test[spectral_cols].values
    
    return X_train_raw, y_train, X_test_raw, y_test

def get_pca_data(X_train_raw, X_test_raw):
    print("--- Preparing PCA Data (Components 2-102) ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_te_s = scaler.transform(X_test_raw)
    
    pca = PCA()
    X_tr_pca_full = pca.fit_transform(X_tr_s)
    X_te_pca_full = pca.transform(X_te_s)
    
    X_tr_pca = X_tr_pca_full[:, 2:102]
    X_te_pca = X_te_pca_full[:, 2:102]
    
    scaler_pca = StandardScaler()
    X_tr_final = scaler_pca.fit_transform(X_tr_pca)
    X_te_final = scaler_pca.transform(X_te_pca)
    
    return X_tr_final, X_te_final

def main():
    X_tr_raw, y_train, X_te_raw, y_test = get_data()
    X_tr_pca, X_te_pca = get_pca_data(X_tr_raw, X_te_raw)
    
    print("--- Training Tuned Random Forest (Est=300) ---")
    rf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=None, 
        min_samples_leaf=1, 
        min_samples_split=2, 
        class_weight='balanced', 
        random_state=SEED, 
        n_jobs=-1
    )
    rf.fit(X_tr_pca, y_train)
    y_pred = rf.predict(X_te_pca)
    
    print_metrics("Random Forest (Tuned)", y_test, y_pred)

if __name__ == "__main__":
    main()
