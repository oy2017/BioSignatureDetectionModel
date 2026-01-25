import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

# --- Set Environment ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Set Random Seeds ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def create_best_model(input_dim):
    """
    Creates the best performing MLP architecture found via tuning.
    Config: Deep & Wide (512, 256, 128)
    Params: Dropout 0.4, LR 0.0005
    """
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # Layer 1
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        
        # Layer 2
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        
        # Layer 3
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        
        # Output
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate the Final Best MLP Model (H2).")
    args = parser.parse_args()
    
    fill_gas = "H2"
    pca_components = 100
    batch_size = 128
    
    # --- 1. Load Data ---
    print(f"--- Loading {fill_gas} Data ---")
    df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
    df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

    # Labels
    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
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

    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_pca)
    X_test_final = scaler_pca.transform(X_test_pca)
    
    # Shuffle Training Data
    X_train_final, y_train = shuffle(X_train_final, y_train, random_state=SEED)

    # --- 3. Train Best Model ---
    print(f"--- Training Best Model (DeepWide) ---")
    model = create_best_model(input_dim=pca_components)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_final, y_train,
        epochs=200, # High cap
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # --- 4. Evaluate ---
    print("\n--- Evaluating on Test Set ---")
    loss, accuracy = model.evaluate(X_test_final, y_test, verbose=0)
    y_pred_prob = model.predict(X_test_final)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # --- 5. Save Results ---
    results_path = 'final_results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    # Report
    report_filename = os.path.join(results_path, f'{fill_gas}_final_best_mlp_report.txt')
    report_str = classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio'])
    
    with open(report_filename, 'w') as f:
        f.write(f"Final Best MLP Model (DeepWide) for {fill_gas}\n")
        f.write("==============================================\n")
        f.write(f"Architecture: [512, 256, 128]\n")
        f.write(f"Parameters: Batch={batch_size}, Drop=0.4, LR=0.0005\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report_str)
        
    print(f"Report saved to {report_filename}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Bio', 'Bio'], yticklabels=['Non-Bio', 'Bio'])
    plt.title(f'Confusion Matrix: Best MLP ({accuracy:.1%})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_filename = os.path.join(results_path, f'{fill_gas}_final_best_mlp_confusion_matrix.png')
    plt.savefig(cm_filename)
    print(f"Confusion Matrix saved to {cm_filename}")
    
    # Save Model
    model_filename = os.path.join(results_path, f'{fill_gas}_best_mlp_model.keras')
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
