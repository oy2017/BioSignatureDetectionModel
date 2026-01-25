import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

# --- Set Environment ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_model(input_dim, seed):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Measure Training Stability with Different Seeds.")
    parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
    args = parser.parse_args()
    
    fill_gas = args.fill_gas.upper()
    pca_components = 100
    batch_size = 128
    num_runs = 5
    seeds = [42, 43, 44, 45, 46]
    
    # --- 1. Load Data ---
    print(f"--- Loading {fill_gas} Data ---")
    df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
    df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

    y_train_full = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values

    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train_raw = df_train[spectral_cols].values
    X_test_raw = df_test[spectral_cols].values

    # --- 2. Preprocessing (Fixed Pipeline) ---
    print("--- Fitting Preprocessing Pipeline ---")
    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)
    X_test_scaled = scaler_raw.transform(X_test_raw)

    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_pca)
    X_test_final = scaler_pca.transform(X_test_pca)
    
    # --- 3. Training Loop ---
    metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }
    
    print(f"\n--- Starting {num_runs} Training Runs ---")
    
    for i, seed in enumerate(seeds):
        print(f"\n[Run {i+1}/{num_runs}] Seed: {seed}")
        
        # Shuffle with current seed
        X_tr_shuffled, y_tr_shuffled = shuffle(X_train_final, y_train_full, random_state=seed)
        
        # Build Model
        model = create_model(input_dim=pca_components, seed=seed)
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True
        )
        
        history = model.fit(
            X_tr_shuffled, y_tr_shuffled,
            epochs=200, 
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0 # Silent
        )
        
        # Best Val Acc
        val_acc = max(history.history['val_accuracy'])
        print(f"  -> Best Val Acc: {val_acc:.4f} (Ep {np.argmax(history.history['val_accuracy'])+1})")
        
        # Evaluate
        y_pred_prob = model.predict(X_test_final, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["F1-Score"].append(f1)
        
        print(f"  -> Test Accuracy: {acc:.4f}")

    # --- 4. Statistics ---
    print("\n" + "="*50)
    print(f"Training Stability Results ({num_runs} Runs)")
    print("="*50)
    print(f"{ 'Metric':<15} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 41)
    
    for key in metrics:
        mean_val = np.mean(metrics[key])
        std_val = np.std(metrics[key])
        print(f"{key:<15} | {mean_val:.4f}     | {std_val:.4f}")
        
    print("="*50)
    
    # Save Report
    report_path = f'final_results/{fill_gas}_training_stability_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Training Stability Report ({fill_gas})\n")
        f.write(f"Architecture: [512, 256, 128], Batch=128, Drop=0.4, LR=0.0005\n")
        f.write("=========================================================\n")
        for i, seed in enumerate(seeds):
            f.write(f"Run {i+1} (Seed {seed}): Acc={metrics['Accuracy'][i]:.4f}, F1={metrics['F1-Score'][i]:.4f}\n")
        f.write("\nAggregate Stats:\n")
        for key in metrics:
            mean_val = np.mean(metrics[key])
            std_val = np.std(metrics[key])
            f.write(f"{key:<15} | Mean: {mean_val:.4f} | Std: {std_val:.4f}\n")
            
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
