import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import random
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Set Environment ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Set Random Seeds ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def create_model(input_dim, hidden_layers, dropout_rate, learning_rate):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    for units in hidden_layers:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Validate Top MLP Candidates with Early Stopping.")
    parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
    parser.add_argument("--pca_components", type=int, default=100, help="Number of PCA components.")
    args = parser.parse_args()

    fill_gas = args.fill_gas.upper()
    
    # --- 1. Load Data ---
    print(f"--- Loading {fill_gas} Data ---")
    try:
        df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
        df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')
    except FileNotFoundError:
        print(f"Error: Data files for {fill_gas} not found.")
        return

    # Labels
    y_train_full = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values

    # Features
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train_raw = df_train[spectral_cols].values
    X_test_raw = df_test[spectral_cols].values

    # --- 2. Preprocessing ---
    print("--- Preprocessing: Scaling -> PCA -> Scaling ---")
    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)
    X_test_scaled = scaler_raw.transform(X_test_raw)

    pca = PCA(n_components=args.pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_pca)
    X_test_final = scaler_pca.transform(X_test_pca)
    
    # --- 2.5 Shuffle Data ---
    # Critical: Keras validation_split takes the LAST 20%. 
    # If data is ordered by class, validation will be purely one class.
    from sklearn.utils import shuffle
    X_train_final, y_train_full = shuffle(X_train_final, y_train_full, random_state=SEED)
    
    # --- 3. Define Candidates ---
    candidates = [
        # Name, Layers, Batch, Drop, LR
        ("DeepWide", (512, 256, 128), 128, 0.5, 0.0005),
        ("BestCV", (256, 128, 64), 128, 0.3, 0.0005),
        ("BaselineVar", (256, 128, 64), 128, 0.4, 0.0005),
        ("Shallow", (256, 128), 64, 0.3, 0.001),
        ("DeepNarrow", (128, 128, 64, 32), 128, 0.5, 0.001)
    ]
    
    results = []
    
    print(f"--- Validating {len(candidates)} Candidates ---")
    
    for name, layers, batch, drop, lr in candidates:
        print(f"\n--- Training {name} ---")
        print(f"Config: Layers={layers}, Batch={batch}, Drop={drop}, LR={lr}")
        
        # Build
        model = create_model(
            input_dim=args.pca_components,
            hidden_layers=layers,
            dropout_rate=drop,
            learning_rate=lr
        )
        
        # Train with Early Stopping (Internal 20% split)
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_final, y_train_full,
            epochs=100, # Cap
            batch_size=batch,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0 # Silent
        )
        
        val_acc_hist = history.history['val_accuracy']
        best_val_idx = np.argmax(val_acc_hist)
        best_val_acc = val_acc_hist[best_val_idx]
        best_epoch = best_val_idx + 1
        
        print(f"  -> Stopped at Epoch {len(val_acc_hist)}")
        print(f"  -> Best Val Acc (Internal): {best_val_acc:.4f} at Epoch {best_epoch}")
        
        # Evaluate on Test
        y_pred_prob = model.predict(X_test_final)
        y_pred = (y_pred_prob > 0.5).astype(int)
        test_acc = accuracy_score(y_test, y_pred)
        
        print(f"  -> Test Accuracy: {test_acc:.4f}")
        
        results.append({
            "Name": name,
            "Layers": str(layers),
            "Batch": batch,
            "Drop": drop,
            "LR": lr,
            "Epochs": best_epoch,
            "Val_Acc": best_val_acc,
            "Test_Acc": test_acc
        })

    # --- 4. Save Report ---
    print("\n--- Validation Complete ---")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Test_Acc", ascending=False)
    
    print(results_df[["Name", "Test_Acc", "Val_Acc", "Epochs"]])
    
    report_path = f"final_results/{fill_gas}_top_models_validation.txt"
    with open(report_path, "w") as f:
        f.write(f"Top Models Validation Results ({fill_gas})\n")
        f.write("=========================================\n\n")
        f.write(results_df.to_string(index=False))
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
