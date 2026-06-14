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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
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

def build_mlp(input_dim):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in [256, 128, 64]:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_dim):
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(filters=32, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(filters=64, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
        Flatten(),
        Dense(100), BatchNormalization(), Activation('relu'), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_cm(y_true, y_pred, name, acc):
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate row totals to generate fractions
    row_sums = cm.sum(axis=1)
    
    # Create custom string labels for each cell
    labels = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            total = row_sums[i]
            percentage = (count / total) * 100 if total > 0 else 0
            labels[i, j] = f"{percentage:.1f}%\n({count}/{total})"
            
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=['Non-Bio (0)', 'Bio (1)'], 
                yticklabels=['Non-Bio (0)', 'Bio (1)'],
                annot_kws={"size": 14})
    # Title removed per reviewer request
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.savefig(f'final_results/CM_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate():
    os.makedirs('final_results', exist_ok=True)
    models_to_run = ['XGBoost', 'Random Forest', 'MLP', 'CNN']
    results = {}

    for name in models_to_run:
        print(f"--- Training & Evaluating {name} ---")
        # UNIFIED PIPELINE: ALL models use 0-101
        X_train, y_train, test_sets = get_data(0, 102) 
            
        # Train with Final Best Architectures
        if name == 'XGBoost':
            model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8, use_label_encoder=False, eval_metric='logloss', random_state=SEED, n_jobs=-1)
            model.fit(X_train, y_train)
        elif name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=2, max_depth=None, random_state=SEED, n_jobs=-1)
            model.fit(X_train, y_train)
        elif name == 'MLP':
            model = build_mlp(102)
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.2, callbacks=[es], verbose=0)
        elif name == 'CNN':
            model = build_cnn(102)
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train.reshape(-1, 102, 1), y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es], verbose=0)

        # Evaluate on 5 sets
        metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        for i, (X_test, y_test) in enumerate(test_sets):
            if name in ['MLP', 'CNN']:
                if name == 'CNN':
                    X_test = X_test.reshape(-1, 102, 1)
                preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
            else:
                preds = model.predict(X_test)
                
            acc = accuracy_score(y_test, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
            
            metrics['acc'].append(acc)
            metrics['prec'].append(prec)
            metrics['rec'].append(rec)
            metrics['f1'].append(f1)
            
            # Save CM for the first set only
            if i == 0:
                save_cm(y_test, preds, name, acc)

        # Calculate Means and Stds
        results[name] = {
            'acc': f"{np.mean(metrics['acc']):.2%} (± {np.std(metrics['acc']):.2%})",
            'prec': f"{np.mean(metrics['prec']):.2%} (± {np.std(metrics['prec']):.2%})",
            'rec': f"{np.mean(metrics['rec']):.2%} (± {np.std(metrics['rec']):.2%})",
            'f1': f"{np.mean(metrics['f1']):.2%} (± {np.std(metrics['f1']):.2%})"
        }
        print(f"Done. Mean Acc: {np.mean(metrics['acc']):.2%}")

    # Print Markdown Table
    print("\n\n**Table II. Model Performance Summary (Mean ± Standard Deviation over 5 Sets)**\n")
    print("| Model Architecture | Accuracy | Precision | Recall | F1-Score |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for name in models_to_run:
        r = results[name]
        print(f"| **{name}** | {r['acc']} | {r['prec']} | {r['rec']} | **{r['f1']}** |")

if __name__ == "__main__":
    evaluate()
