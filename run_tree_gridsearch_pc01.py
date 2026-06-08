import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils import shuffle
import re
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def get_data(start_idx, end_idx):
    df_train = pd.read_parquet('multirex_spectra_H2_train.parquet')
    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train_raw = df_train[spectral_cols].values

    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)

    pca = PCA(n_components=102, random_state=SEED)
    X_train_pca_full = pca.fit_transform(X_train_scaled)
    X_train_pca = X_train_pca_full[:, start_idx:end_idx]

    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_pca)
    
    X_train_final, y_train = shuffle(X_train_final, y_train, random_state=SEED)
    return X_train_final, y_train

def run_tree_gridsearch_pc01():
    print("=========================================")
    print("  GRIDSEARCH: Trees WITH PC0 and PC1 ")
    print("=========================================\n")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    X_train, y_train = get_data(0, 102) # INDICES 0-101

    # 1. XGBoost
    print("--- 1. XGBoost (with PC0/1) ---")
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED, n_jobs=-1)
    xgb_grid = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    print(f"XGBoost Best Params: {xgb_grid.best_params_}")
    print(f"XGBoost Best CV Acc: {xgb_grid.best_score_:.4f}\n")

    # 2. Random Forest
    print("--- 2. Random Forest (with PC0/1) ---")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"RF Best Params: {rf_grid.best_params_}")
    print(f"RF Best CV Acc: {rf_grid.best_score_:.4f}\n")

if __name__ == "__main__":
    run_tree_gridsearch_pc01()
