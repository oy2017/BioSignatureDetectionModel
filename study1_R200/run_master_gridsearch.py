import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Input, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils import shuffle
import re
import random

# Silence TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
    
    # SHUFFLE
    X_train_final, y_train = shuffle(X_train_final, y_train, random_state=SEED)
    
    return X_train_final, y_train

def build_mlp(learning_rate=0.001, dropout_rate=0.4, hidden_layers=(512, 256, 128)):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Input(shape=(102,)))
    for units in hidden_layers:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(learning_rate=0.001, dropout_rate=0.3, filters=(64, 128)):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Input(shape=(102, 1)))
    for f in filters:
        model.add(Conv1D(filters=f, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_gridsearch():
    print("=========================================")
    print("  RUNNING UNIFIED GRIDSEARCH (Corrected) ")
    print("=========================================\n")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # ---------------------------------------------------------
    # 1. XGBOOST (Uses PC 2-101)
    # ---------------------------------------------------------
    print("--- 1. XGBoost ---")
    X_train_tree, y_train_tree = get_data(2, 102)
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED, n_jobs=-1)
    xgb_grid = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)
    xgb_grid.fit(X_train_tree, y_train_tree)
    print(f"XGBoost Best Params: {xgb_grid.best_params_}")
    print(f"XGBoost Best CV Acc: {xgb_grid.best_score_:.4f}\n")

    # ---------------------------------------------------------
    # 2. RANDOM FOREST (Uses PC 2-101)
    # ---------------------------------------------------------
    print("--- 2. Random Forest ---")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)
    rf_grid.fit(X_train_tree, y_train_tree)
    print(f"RF Best Params: {rf_grid.best_params_}")
    print(f"RF Best CV Acc: {rf_grid.best_score_:.4f}\n")

    # ---------------------------------------------------------
    # 3. MLP (Uses PC 0-101)
    # ---------------------------------------------------------
    print("--- 3. MLP ---")
    X_train_nn, y_train_nn = get_data(0, 102)
    
    mlp_model = KerasClassifier(
        model=build_mlp,
        verbose=0,
        epochs=30,
        batch_size=128
    )
    
    mlp_param_grid = {
        'model__learning_rate': [0.001, 0.0005],
        'model__dropout_rate': [0.2, 0.3, 0.4],
        'model__hidden_layers': [(256, 128, 64), (512, 256, 128)],
        'batch_size': [32, 64, 128]
    }
    
    mlp_grid = GridSearchCV(estimator=mlp_model, param_grid=mlp_param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)
    mlp_grid.fit(X_train_nn, y_train_nn)
    print(f"MLP Best Params: {mlp_grid.best_params_}")
    print(f"MLP Best CV Acc: {mlp_grid.best_score_:.4f}\n")

    # ---------------------------------------------------------
    # 4. CNN (Uses PC 0-101)
    # ---------------------------------------------------------
    print("--- 4. CNN ---")
    X_train_cnn = X_train_nn.reshape(-1, 102, 1)
    
    def build_cnn_custom(learning_rate=0.001, dropout_rate=0.3, filters=(64, 128), kernel_size=5):
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Input(shape=(102, 1)))
        for f in filters:
            model.add(Conv1D(filters=f, kernel_size=kernel_size, padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    cnn_model = KerasClassifier(
        model=build_cnn_custom,
        verbose=0,
        epochs=50,
        batch_size=64
    )
    
    cnn_param_grid = {
        'model__learning_rate': [0.001, 0.0005],
        'model__dropout_rate': [0.3, 0.5],
        'model__filters': [(32, 64), (64, 128)],
        'model__kernel_size': [3, 5],
        'batch_size': [32, 64]
    }
    
    cnn_grid = GridSearchCV(estimator=cnn_model, param_grid=cnn_param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)
    cnn_grid.fit(X_train_cnn, y_train_nn)
    print(f"CNN Best Params: {cnn_grid.best_params_}")
    print(f"CNN Best CV Acc: {cnn_grid.best_score_:.4f}\n")

if __name__ == "__main__":
    run_gridsearch()