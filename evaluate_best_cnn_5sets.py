import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import shuffle
import re
import random

# --- Set Seeds for Reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_best_cnn(input_shape):
    """
    Creates the best performing CNN architecture (E10).
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=128, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(100),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data(file_path):
    df = pd.read_parquet(file_path)
    df['label'] = df['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
    
    float_pattern = re.compile(r"^-?\d+\.\d+$")
    cols = [c for c in df.columns if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]
    
    X = df[cols].values
    y = df['label'].values
    return X, y

def main():
    fill_gas = "H2"
    train_file = f"multirex_spectra_{fill_gas}_train.parquet"
    test_files = [f"multirex_spectra_{fill_gas}_test_set_{i}.parquet" for i in range(1, 6)]
    
    print(f"--- Loading Training Data: {train_file} ---")
    X_train_raw, y_train = load_data(train_file)
    X_train_raw, y_train = shuffle(X_train_raw, y_train, random_state=SEED)
    
    # --- Preprocessing (E10 Config: PCA 2-102) ---
    print("--- Preprocessing: Scaling -> PCA (2-102) -> Scaling ---")
    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)
    
    pca = PCA(n_components=102) # Need up to 102
    X_train_pca_full = pca.fit_transform(X_train_scaled)
    X_train_clean = X_train_pca_full[:, 2:102]
    
    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_clean)
    
    # Reshape for CNN
    X_train_cnn = X_train_final.reshape(X_train_final.shape[0], X_train_final.shape[1], 1)
    
    # --- Train Model ---
    print("--- Training Best CNN (E10 Configuration) ---")
    model = create_best_cnn(input_shape=(100, 1))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train_cnn, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # --- Evaluate on 5 Test Sets ---
    metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }
    
    print("\n--- Evaluating on 5 Independent Test Sets ---")
    for i, test_file in enumerate(test_files):
        if not os.path.exists(test_file):
            print(f"Warning: {test_file} not found. Skipping.")
            continue
            
        print(f"Processing Set {i+1}: {test_file}")
        X_test_raw, y_test = load_data(test_file)
        
        # Apply Preprocessing
        X_test_scaled = scaler_raw.transform(X_test_raw)
        X_test_pca = pca.transform(X_test_scaled)
        X_test_clean = X_test_pca[:, 2:102]
        X_test_final = scaler_pca.transform(X_test_clean)
        X_test_cnn = X_test_final.reshape(X_test_final.shape[0], X_test_final.shape[1], 1)
        
        # Predict
        y_pred_prob = model.predict(X_test_cnn, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["F1-Score"].append(f1)
        
        print(f"  -> Accuracy: {acc:.4f}")
        
    # --- Summary ---
    print("\n" + "="*50)
    print(f"Aggregate Results for Best CNN (E10) across {len(metrics['Accuracy'])} Sets")
    print("="*50)
    print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 41)
    
    for key in metrics:
        mean_val = np.mean(metrics[key])
        std_val = np.std(metrics[key])
        print(f"{key:<15} | {mean_val:.4f}     | {std_val:.4f}")
    print("="*50)
    
    # Save Results
    results_path = 'final_results/H2_best_cnn_5sets_stats.txt'
    with open(results_path, 'w') as f:
        f.write("Best CNN Model (E10) Evaluation on 5 Sets (H2)\n")
        f.write("==============================================\n")
        for i in range(len(metrics["Accuracy"])):
            f.write(f"Set {i+1}: {metrics['Accuracy'][i]:.4f}\n")
        f.write("\n")
        f.write(f"Mean Accuracy: {np.mean(metrics['Accuracy']):.4f}\n")
        f.write(f"Std Deviation: {np.std(metrics['Accuracy']):.4f}\n")
        
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
