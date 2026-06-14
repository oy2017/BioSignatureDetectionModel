import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sklearn.utils import shuffle
import os
import re

# --- Configuration ---
SEED = 42
np.random.seed(SEED)

def get_data():
    print("--- Loading Data ---")
    df_train = pd.read_parquet('multirex_spectra_H2_train.parquet')
    df_test = pd.read_parquet('multirex_spectra_H2_test.parquet')
    
    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    
    X_train_raw = df_train[spectral_cols].values
    X_test_raw = df_test[spectral_cols].values
    
    return X_train_raw, y_train, X_test_raw, y_test

def get_pca_data(X_train_raw, X_test_raw):
    print(f"--- Preparing PCA Data (Components 0-101) ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_te_s = scaler.transform(X_test_raw)
    
    pca = PCA(n_components=102, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr_s)[:, 0:102]
    X_te_pca = pca.transform(X_te_s)[:, 0:102]
    
    scaler_pca = StandardScaler()
    X_tr_final = scaler_pca.fit_transform(X_tr_pca)
    X_te_final = scaler_pca.transform(X_te_pca)
    
    return X_tr_final, X_te_final

def main():
    os.makedirs('final_results', exist_ok=True)
    
    X_tr_raw, y_train, X_te_raw, y_test = get_data()
    X_train, X_test = get_pca_data(X_tr_raw, X_te_raw)
    
    # Shuffle Training Data
    X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

    print("--- Training XGBoost ---")
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=SEED, n_jobs=-1, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    # Get probabilities
    y_score = xgb.predict_proba(X_test)[:, 1]
    
    # Calculate Average Precision
    ap = average_precision_score(y_test, y_score)
    print(f"Average Precision (AP): {ap:.4f}")

    print("--- Plotting Precision-Recall Curve ---")
    plt.figure(figsize=(8, 6))
    
    display = PrecisionRecallDisplay.from_predictions(
        y_test, y_score, name=f"XGBoost (AP = {ap:.2f})", color='darkorange', linewidth=2
    )
    
    display.ax_.set_title("Precision-Recall Curve for XGBoost", fontsize=16, fontweight='bold')
    display.ax_.set_xlabel("Recall (True Positive Rate)", fontsize=14)
    display.ax_.set_ylabel("Precision (Positive Predictive Value)", fontsize=14)
    display.ax_.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for potential operational thresholds
    # E.g., showing where we can get 95% recall
    # We find the threshold closest to 95% recall
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    
    # Find index where recall is closest to 0.95
    idx_high_recall = np.argmin(np.abs(recall - 0.95))
    thresh_hr = thresholds[idx_high_recall] if idx_high_recall < len(thresholds) else 0
    prec_hr = precision[idx_high_recall]
    rec_hr = recall[idx_high_recall]
    
    plt.plot(rec_hr, prec_hr, marker='o', markersize=8, color='red', label=f'High-Recall Mode\n(Thresh={thresh_hr:.2f}: Prec={prec_hr:.2f})')
    
    # Find index where precision is closest to 0.95
    idx_high_prec = np.argmin(np.abs(precision - 0.95))
    thresh_hp = thresholds[idx_high_prec] if idx_high_prec < len(thresholds) else 0
    prec_hp = precision[idx_high_prec]
    rec_hp = recall[idx_high_prec]
    
    plt.plot(rec_hp, prec_hp, marker='o', markersize=8, color='green', label=f'High-Precision Mode\n(Thresh={thresh_hp:.2f}: Rec={rec_hp:.2f})')
    
    plt.legend(loc='lower left', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join('final_results', 'pr_curve_xgboost.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()