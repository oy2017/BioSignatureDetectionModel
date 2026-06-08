import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
    
    X_train_final, y_train = shuffle(X_train_final, y_train, random_state=SEED)
    
    # Load 5 test sets
    test_sets = []
    for i in range(1, 6):
        df_test = pd.read_parquet(f'multirex_spectra_H2_test_set_{i}.parquet')
        y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
        X_test_raw = df_test[spectral_cols].values
        
        X_test_scaled = scaler_raw.transform(X_test_raw)
        X_test_pca_full = pca.transform(X_test_scaled)
        X_test_pca = X_test_pca_full[:, start_idx:end_idx]
        X_test_final = scaler_pca.transform(X_test_pca)
        
        test_sets.append((X_test_final, y_test))
        
    return X_train_final, y_train, test_sets

def build_mlp(input_dim, hidden_layers, dropout_rate, lr):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_duel():
    print("=========================================")
    print("  FINAL MLP DUEL: Manual vs GridSearch ")
    print("=========================================\n")
    
    X_train, y_train, test_sets = get_data(0, 102)
    
    models_to_test = {
        "Manual Winner (512-256-128)": {
            "layers": (512, 256, 128),
            "dropout": 0.4,
            "lr": 0.0005
        },
        "GridSearch Winner (256-128-64)": {
            "layers": (256, 128, 64),
            "dropout": 0.3,
            "lr": 0.0005
        }
    }
    
    for name, config in models_to_test.items():
        print(f"--- Training {name} ---")
        model = build_mlp(102, config["layers"], config["dropout"], config["lr"])
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.2, callbacks=[es], verbose=0)
        
        metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        for X_test, y_test in test_sets:
            preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
            acc = accuracy_score(y_test, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
            
            metrics['acc'].append(acc)
            metrics['prec'].append(prec)
            metrics['rec'].append(rec)
            metrics['f1'].append(f1)
            
        print(f"Results across 5 sets:")
        print(f"  Accuracy:  {np.mean(metrics['acc']):.2%} (± {np.std(metrics['acc']):.2%})")
        print(f"  Precision: {np.mean(metrics['prec']):.2%} (± {np.std(metrics['prec']):.2%})")
        print(f"  Recall:    {np.mean(metrics['rec']):.2%} (± {np.std(metrics['rec']):.2%})")
        print(f"  F1-Score:  {np.mean(metrics['f1']):.2%} (± {np.std(metrics['f1']):.2%})\n")

if __name__ == "__main__":
    run_duel()