import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import re
import random

# --- Set Seeds for Reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def create_raw_cnn(input_shape):
    """
    Creates the optimized Raw CNN architecture.
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=64, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(128),
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    
    print("--- Training Optimized Raw CNN ---")
    model = create_raw_cnn(input_shape=(X_train_cnn.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    metrics = {
        "Accuracy": [],
        "Precision (Bio)": [],
        "Recall (Bio)": [],
        "F1-Score (Bio)": []
    }
    print("\n--- Evaluating on 5 Independent Test Sets ---")
    for i, test_file in enumerate(test_files):
        X_test_raw, y_test = load_data(test_file)
        X_test_scaled = scaler.transform(X_test_raw)
        X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        y_pred_prob = model.predict(X_test_cnn, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate Accuracy
        acc = accuracy_score(y_test, y_pred)
        
        # Calculate per-class metrics (pos_label=1 is Biosignature)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[1], average='binary')
        
        metrics["Accuracy"].append(acc)
        metrics["Precision (Bio)"].append(prec)
        metrics["Recall (Bio)"].append(rec)
        metrics["F1-Score (Bio)"].append(f1)
        
        print(f"  Set {i+1}: Acc={acc:.4f}, Bio-Recall={rec:.4f}")
        
    print("\n" + "="*50)
    print(f"Aggregate Results for Raw CNN (Biosignature Class)")
    print("="*50)
    print(f"{'Metric':<18} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 44)
    
    for key in metrics:
        mean_val = np.mean(metrics[key])
        std_val = np.std(metrics[key])
        print(f"{key:<18} | {mean_val:.4f}     | {std_val:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
