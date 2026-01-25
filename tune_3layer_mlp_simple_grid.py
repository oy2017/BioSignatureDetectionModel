import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import random
import json
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
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

def create_model(input_dim, learning_rate, dropout_rate):
    """
    Creates the 3-Layer MLP architecture [256, 128, 64].
    """
    model = Sequential([
        Flatten(input_shape=(input_dim,)),
        
        # Layer 1
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        
        # Layer 2
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        
        # Layer 3
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        
        # Output
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Simple Grid Search (Single Split) with Early Stopping.")
    parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
    parser.add_argument("--quick", action="store_true", help="Run a quick test with reduced grid and data.")
    args = parser.parse_args()

    fill_gas = args.fill_gas.upper()
    
    # --- 1. Load Data ---
    print(f"--- Loading {fill_gas} Data ---")
    try:
        df_train_full = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
        df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')
    except FileNotFoundError:
        print(f"Error: Data files for {fill_gas} not found.")
        return

    # Extract Labels
    df_train_full['label'] = df_train_full['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
    df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
    y_train_full = df_train_full['label'].values
    y_test = df_test['label'].values

    # Extract Features
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train_full.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train_full = df_train_full[spectral_cols].values
    X_test = df_test[spectral_cols].values

    # Quick Mode
    if args.quick:
        print("--- QUICK MODE: Subsampling data ---")
        X_train_full = X_train_full[:500]
        y_train_full = y_train_full[:500]

    # --- 2. Preprocessing (Replicating Pipeline Manually) ---
    print("--- Preprocessing: Scaling -> PCA -> Scaling ---")
    
    # 2.1 Scale Raw
    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_full)
    X_test_scaled = scaler_raw.transform(X_test)
    
    # 2.2 PCA
    pca = PCA() # Fits on everything
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Filter 100 components
    X_train_pca_100 = X_train_pca[:, 0:100]
    X_test_pca_100 = X_test_pca[:, 0:100]
    
    # 2.3 Scale PCA
    scaler_pca = StandardScaler()
    X_train_processed = scaler_pca.fit_transform(X_train_pca_100)
    X_test_processed = scaler_pca.transform(X_test_pca_100)
    
    # --- 3. Single Validation Split ---
    # This replicates the "validation_split=0.2" logic from the original script
    # giving us ~2400 Train and ~600 Val samples.
    print("--- Creating Validation Split (80/20) ---")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_processed, y_train_full, 
        test_size=0.2, 
        random_state=SEED, 
        stratify=y_train_full
    )
    
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Validation Samples: {X_val.shape[0]}")

    # --- 4. Define Grid ---
    if args.quick:
        param_grid = {
            'epochs': [20],
            'batch_size': [32],
            'learning_rate': [0.001],
            'dropout_rate': [0.4]
        }
    else:
        # We set 'epochs' high because Early Stopping decides the actual end.
        # We can still tune 'patience' if we wanted, but let's stick to model params.
        param_grid = {
            'epochs': [100], 
            'batch_size': [64, 128],
            'learning_rate': [0.001, 0.0005],
            'dropout_rate': [0.3, 0.4, 0.5]
        }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"--- Starting Grid Search ({len(combinations)} combinations) ---")
    
    best_val_acc = -1.0
    best_params = None
    best_model_history = None
    
    results = []

    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        # Clear session
        tf.keras.backend.clear_session()
        
        model = create_model(
            input_dim=100, 
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate']
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val), # Explicit validation set
            callbacks=[early_stopping],
            verbose=0 # Silent training
        )
        
        # Get best validation accuracy from history (or evaluate manually)
        # restore_best_weights=True ensures model is at best state.
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"  -> Val Accuracy: {val_acc:.4f}")
        
        results.append({
            'params': params,
            'val_accuracy': val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params

    # --- 5. Retrain Best Model on Full Train (Optional but Recommended) ---
    # Or simply evaluate the best model found on the Test Set.
    # The original script evaluated the model trained on 80% data against the Test Set.
    # To be "comparable", we can do exactly that.
    # But usually, you'd retrain on 100% (Train+Val) before Test.
    # Let's stick to the Original Script logic: The model we just found (trained on 80%) 
    # is the one we use?
    # Actually, in the loop, we discarded the models. We need to rebuild the best one.
    
    print(f"\n--- Grid Search Complete ---")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best Parameters: {best_params}")
    
    print("\n--- Rebuilding Best Model ---")
    # We will train this exactly as the loop did (80% Train, 20% Val)
    # This replicates 'evaluate_mlp_3layer_best.py' behavior exactly.
    
    final_model = create_model(
        input_dim=100, 
        learning_rate=best_params['learning_rate'], 
        dropout_rate=best_params['dropout_rate']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    final_model.fit(
        X_train, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # --- 6. Final Evaluation ---
    print("\n--- Evaluating on Test Set ---")
    y_pred_prob = final_model.predict(X_test_processed)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {test_acc:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio'])
    print("\nClassification Report:")
    print(report)

    # --- 7. Save Report ---
    results_path = 'final_results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    report_filename = f'{fill_gas}_3layer_mlp_simple_grid_report.txt'
    with open(os.path.join(results_path, report_filename), 'w') as f:
        f.write(f"Simple Grid Search (Single Split) Results ({fill_gas})\n")
        f.write("=====================================================\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best Parameters: {json.dumps(best_params, indent=2)}\n\n")
        f.write(f"Test Set Accuracy: {test_acc:.4f}\n\n")
        f.write("Full Grid Results:\n")
        
        # Sort results
        sorted_results = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)
        for res in sorted_results:
            f.write(f"Params: {res['params']}, Val Acc: {res['val_accuracy']:.4f}\n")
            
        f.write("\nClassification Report:\n")
        f.write(report)

    print(f"Report saved to {os.path.join(results_path, report_filename)}")

if __name__ == "__main__":
    main()
