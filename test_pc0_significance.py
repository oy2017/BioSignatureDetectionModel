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
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import shuffle
import re
import random
import sys

# Silence TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_data(start_idx, end_idx):
    df_train = pd.read_parquet('multirex_spectra_H2_train.parquet')
    df_test = pd.read_parquet('multirex_spectra_H2_test.parquet')

    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values

    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train_raw = df_train[spectral_cols].values
    X_test_raw = df_test[spectral_cols].values

    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)
    X_test_scaled = scaler_raw.transform(X_test_raw)

    pca = PCA(n_components=102, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)[:, start_idx:end_idx]
    X_test_pca = pca.transform(X_test_scaled)[:, start_idx:end_idx]

    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_pca)
    X_test_final = scaler_pca.transform(X_test_pca)
    
    # Crucial Fix: Shuffle training data so validation_split doesn't grab just one class
    X_train_final, y_train = shuffle(X_train_final, y_train, random_state=SEED)
    
    return X_train_final, y_train, X_test_final, y_test

def build_mlp(input_dim):
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512), BatchNormalization(), Activation('relu'), Dropout(0.4),
        Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.4),
        Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_dim):
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(filters=64, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(filters=128, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
        Flatten(),
        Dense(100), BatchNormalization(), Activation('relu'), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_mcnemar_test(y_true, y_pred1, y_pred2):
    # Model 1 = 0-101 (With Systematics)
    # Model 2 = 2-101 (Without Systematics)
    table = [[0, 0], [0, 0]]
    for i in range(len(y_true)):
        correct1 = (y_pred1[i] == y_true[i])
        correct2 = (y_pred2[i] == y_true[i])
        
        if correct1 and correct2:
            table[0][0] += 1
        elif correct1 and not correct2:
            table[0][1] += 1
        elif not correct1 and correct2:
            table[1][0] += 1
        elif not correct1 and not correct2:
            table[1][1] += 1
            
    result = mcnemar(table, exact=False, correction=True)
    return table, result.pvalue

def evaluate_model(name):
    print(f"\n--- Evaluating {name} ---")
    
    X_train_0, y_train, X_test_0, y_test = get_data(0, 102)
    X_train_2, _, X_test_2, _ = get_data(2, 102)
    
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    if name == 'Random Forest':
        m0 = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_depth=None, random_state=SEED, n_jobs=-1)
        m2 = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=2, max_depth=None, random_state=SEED, n_jobs=-1)
        m0.fit(X_train_0, y_train)
        m2.fit(X_train_2, y_train)
        pred0 = m0.predict(X_test_0)
        pred2 = m2.predict(X_test_2)
        
    elif name == 'XGBoost':
        m0 = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=SEED, n_jobs=-1)
        m2 = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=SEED, n_jobs=-1)
        m0.fit(X_train_0, y_train)
        m2.fit(X_train_2, y_train)
        pred0 = m0.predict(X_test_0)
        pred2 = m2.predict(X_test_2)
        
    elif name == 'MLP':
        m0 = build_mlp(102)
        m2 = build_mlp(100)
        m0.fit(X_train_0, y_train, epochs=200, batch_size=128, validation_split=0.2, callbacks=[es], verbose=0)
        m2.fit(X_train_2, y_train, epochs=200, batch_size=128, validation_split=0.2, callbacks=[es], verbose=0)
        pred0 = (m0.predict(X_test_0, verbose=0) > 0.5).astype(int).flatten()
        pred2 = (m2.predict(X_test_2, verbose=0) > 0.5).astype(int).flatten()
        
    elif name == 'CNN':
        m0 = build_cnn(102)
        m2 = build_cnn(100)
        m0.fit(X_train_0.reshape(-1, 102, 1), y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es], verbose=0)
        m2.fit(X_train_2.reshape(-1, 100, 1), y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es], verbose=0)
        pred0 = (m0.predict(X_test_0.reshape(-1, 102, 1), verbose=0) > 0.5).astype(int).flatten()
        pred2 = (m2.predict(X_test_2.reshape(-1, 100, 1), verbose=0) > 0.5).astype(int).flatten()

    acc0 = np.mean(pred0 == y_test)
    acc2 = np.mean(pred2 == y_test)
    
    # Calculate Recall for Biosignature class (class 1)
    # Recall = TP / (TP + FN)
    tp0 = np.sum((pred0 == 1) & (y_test == 1))
    fn0 = np.sum((pred0 == 0) & (y_test == 1))
    recall0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
    
    tp2 = np.sum((pred2 == 1) & (y_test == 1))
    fn2 = np.sum((pred2 == 0) & (y_test == 1))
    recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    
    table, pval = run_mcnemar_test(y_test, pred0, pred2)
    
    print(f"Accuracy with PC0/1:    {acc0:.4f}  | Recall: {recall0:.4f}")
    print(f"Accuracy without PC0/1: {acc2:.4f}  | Recall: {recall2:.4f}")
    print(f"McNemar p-value (Acc):  {pval:.4f}")
    if pval < 0.05:
        print("-> The difference is STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("-> The difference is NOT statistically significant (p >= 0.05).")

if __name__ == "__main__":
    models = ['Random Forest', 'XGBoost', 'MLP', 'CNN']
    for m in models:
        evaluate_model(m)