import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import re

def main():
    fill_gas = "H2"
    df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
    df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values

    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    
    X_train_raw = df_train[spectral_cols].values
    X_test_raw = df_test[spectral_cols].values

    # Preprocessing
    scaler_raw = StandardScaler()
    X_train_s = scaler_raw.fit_transform(X_train_raw)
    X_test_s = scaler_raw.transform(X_test_raw)

    pca = PCA(n_components=102, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_p)
    X_test_final = scaler_pca.transform(X_test_p)

    # Train XGBoost
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_final, y_train)
    
    y_pred = model.predict(X_test_final)
    errors_mask = (y_pred != y_test)
    df_errors = df_test[errors_mask]
    
    total_errors = len(df_errors)
    total_test = len(df_test)
    
    print(f"Total Errors: {total_errors} out of {total_test} samples (Error Rate: {total_errors/total_test:.1%})")
    
    # 1. Check Chemical Boundary Clustering
    # Region within 0.5 dex of BOTH thresholds
    ch4_thresh = -6.0
    o3_thresh = -7.0
    buffer = 0.5
    
    boundary_errors = df_errors[
        (df_errors['atm CH4'] >= ch4_thresh - buffer) & (df_errors['atm CH4'] <= ch4_thresh + buffer) |
        (df_errors['atm O3'] >= o3_thresh - buffer) & (df_errors['atm O3'] <= o3_thresh + buffer)
    ]
    
    print(f"\n--- Chemical Boundary Analysis ---")
    print(f"Errors within +/- {buffer} dex of boundaries: {len(boundary_errors)}")
    print(f"Percentage of total errors in boundary region: {len(boundary_errors)/total_errors:.1%}")

    # 2. Check Physical Regimes
    low_temp_errors = df_errors[df_errors['atm temperature'] < 1000]
    high_grav_errors = df_errors[df_errors['p_radius'] < 8] # Small radius = higher gravity for given mass
    
    print(f"\n--- Physical Regime Analysis ---")
    print(f"Errors at Low Temp (<1000K): {len(low_temp_errors)} ({len(low_temp_errors)/total_errors:.1%} of errors)")
    print(f"Errors at High Gravity (Radius < 8 Re): {len(high_grav_errors)} ({len(high_grav_errors)/total_errors:.1%} of errors)")

if __name__ == "__main__":
    main()
