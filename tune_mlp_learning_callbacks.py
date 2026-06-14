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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
import re

# Silence TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_R550_data():
    df_train = pd.read_parquet('multirex_spectra_H2_train.parquet')
    y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train_raw = df_train[spectral_cols].values

    test_sets_raw = []
    for i in range(1, 6):
        df_test = pd.read_parquet(f'multirex_spectra_H2_test_set_{i}.parquet')
        y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
        X_test_raw = df_test[spectral_cols].values
        test_sets_raw.append((X_test_raw, y_test))

    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)

    pca = PCA(n_components=102, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)

    scaler_pca = StandardScaler()
    X_train_final = scaler_pca.fit_transform(X_train_pca)
    X_train_shuf, y_train_shuf = shuffle(X_train_final, y_train, random_state=SEED)
    
    # Pre-transform test sets
    test_sets_final = []
    for X_test_raw, y_test in test_sets_raw:
        X_test_scaled = scaler_raw.transform(X_test_raw)
        X_test_pca = pca.transform(X_test_scaled)
        X_test_final = scaler_pca.transform(X_test_pca)
        test_sets_final.append((X_test_final, y_test))
        
    return X_train_shuf, y_train_shuf, test_sets_final

def build_mlp(lr):
    tf.keras.backend.clear_session()
    # We use our locked 3-layer architecture from the swept grid search
    model = Sequential([
        Input(shape=(102,)),
        Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_learning_callback_search():
    print("==================================================================")
    print("  PHASE 2: LEARNING & CALLBACK OPTIMIZATION FOR MLP (R = 550)")
    print("==================================================================")
    
    X_train, y_train, test_sets = get_R550_data()
    
    # We sweep the learning parameters
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [32, 64, 128]
    patience_values = [5, 10, 15]
    
    best_acc = 0
    best_config = None
    
    print(f"{'LR':<8} | {'Batch':<6} | {'Patience':<8} | {'Mean Test Accuracy':<20} | {'Mean Test F1-Score':<18}")
    print("-" * 72)
    
    for lr in learning_rates:
        for batch in batch_sizes:
            for patience in patience_values:
                # Instantiate MLP with current LR
                model = build_mlp(lr)
                es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
                
                # Fit model on training set with dynamic EarlyStopping
                model.fit(X_train, y_train, epochs=150, batch_size=batch, validation_split=0.2, callbacks=[es], verbose=0)
                
                # Evaluate on the 5 independent test sets
                accuracies = []
                f1_scores = []
                for X_test, y_test in test_sets:
                    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
                    accuracies.append(accuracy_score(y_test, preds))
                    f1_scores.append(f1_score(y_test, preds, zero_division=0))
                    
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                
                print(f"{lr:<8} | {batch:<6} | {patience:<8} | {mean_acc:.2%} (± {std_acc:.2%})     | {mean_f1:.2%} (± {std_f1:.2%})")
                
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_config = (lr, batch, patience, mean_acc, mean_f1)
                    
    print("==================================================================")
    print("  PHASE 2 OPTIMIZATION COMPLETE")
    print("==================================================================")
    print(f"Optimal Learning Parameters found:")
    print(f"  * Learning Rate:     {best_config[0]}")
    print(f"  * Batch Size:        {best_config[1]}")
    print(f"  * Callback Patience: {best_config[2]}")
    print(f"  * Mean Test Accuracy: {best_config[3]:.2%}")
    print(f"  * Mean Test F1-Score: {best_config[4]:.2%}")
    print("==================================================================")

if __name__ == "__main__":
    run_learning_callback_search()
