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
from scipy.stats import chi2
from sklearn.utils import shuffle
import re
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_data():
    df_train = pd.read_parquet('multirex_spectra_H2_train.parquet')
    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train_raw = df_train[spectral_cols].values

    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)

    pca = PCA(n_components=102, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)[:, 0:102]

    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_pca)
    X_train_final, y_train = shuffle(X_train_final, y_train, random_state=SEED)
    
    # Pooled five test sets, so the paired tests and Table 4's accuracies
    # describe the same population.
    df_test = pd.concat([pd.read_parquet(f'multirex_spectra_H2_test_set_{i}.parquet')
                         for i in range(1, 6)], ignore_index=True)
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    X_test_raw = df_test[spectral_cols].values
    X_test_scaled = scaler_raw.transform(X_test_raw)
    X_test_pca = pca.transform(X_test_scaled)[:, 0:102]
    X_test_final = scaler_pca.transform(X_test_pca)
        
    return X_train_final, y_train, X_test_final, y_test

def build_mlp():
    model = Sequential([
        Input(shape=(102,)),
        Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

def build_cnn():
    model = Sequential([
        Input(shape=(102, 1)),
        Conv1D(filters=32, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(filters=64, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
        Flatten(),
        Dense(100), BatchNormalization(), Activation('relu'), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy')
    return model

def get_counts(y_true, pred1, pred2):
    """McNemar with continuity correction: returns (b, c, p).

    b and c are the discordant counts -- planets one model gets right and the
    other does not. The statistic depends on them alone, so reporting them
    makes every p-value below checkable by hand.
    """
    c1 = (pred1 == y_true)
    c2 = (pred2 == y_true)
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    return b, c, chi2.sf(stat, 1)

def main():
    X_train, y_train, X_test, y_test = get_data()
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8, eval_metric='logloss', random_state=SEED, n_jobs=-1)
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict(X_test)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=2, max_depth=None, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)

    print("Training MLP...")
    mlp = build_mlp()
    mlp.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.2, callbacks=[es], verbose=0)
    p_mlp = (mlp.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

    print("Training CNN...")
    cnn = build_cnn()
    cnn.fit(X_train.reshape(-1, 102, 1), y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es], verbose=0)
    p_cnn = (cnn.predict(X_test.reshape(-1, 102, 1), verbose=0) > 0.5).astype(int).flatten()

    print(f"\n--- Pairwise McNemar tests (pooled n = {len(y_test)}) ---")
    print(f"{'pair':<26}{'b':>5}{'c':>6}{'p':>11}")
    for name, a, b_ in [("XGBoost vs Random Forest", p_xgb, p_rf),
                        ("XGBoost vs MLP", p_xgb, p_mlp),
                        ("Random Forest vs MLP", p_rf, p_mlp),
                        ("MLP vs 1D-CNN", p_mlp, p_cnn)]:
        b, c, p = get_counts(y_test, a, b_)
        print(f"{name:<26}{b:>5}{c:>6}{p:>11.6f}")

if __name__ == "__main__":
    main()