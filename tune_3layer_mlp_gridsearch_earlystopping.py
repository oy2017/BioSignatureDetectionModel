import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import random
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
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

class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom Scikit-Learn wrapper for the 3-layer MLP with Early Stopping.
    Architecture: [256, 128, 64].
    """
    def __init__(self, input_dim=100, epochs=100, batch_size=128, 
                 learning_rate=0.001, dropout_rate=0.4, verbose=0):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.model_ = None
    
    def fit(self, X, y):
        # Clear session
        tf.keras.backend.clear_session()
        
        # Build Model
        self.model_ = Sequential([
            Flatten(input_shape=(self.input_dim,)),
            
            # Layer 1
            Dense(256),
            BatchNormalization(),
            Activation('relu'),
            Dropout(self.dropout_rate),
            
            # Layer 2
            Dense(128),
            BatchNormalization(),
            Activation('relu'),
            Dropout(self.dropout_rate),
            
            # Layer 3
            Dense(64),
            BatchNormalization(),
            Activation('relu'),
            Dropout(self.dropout_rate),
            
            # Output
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model_.compile(optimizer=optimizer, 
                            loss='binary_crossentropy', 
                            metrics=['accuracy'])
        
        # --- Early Stopping Configuration ---
        # Mimics evaluate_mlp_3layer_best.py
        # Uses internal validation_split=0.2 so the model has a set to monitor.
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        self.model_.fit(
            X, y, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.2,  # Internal split for Early Stopping
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        return (self.model_.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        return self.model_.predict(X)

def main():
    parser = argparse.ArgumentParser(description="GridSearchCV with Early Stopping for 3-Layer MLP.")
    parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
    parser.add_argument("--quick", action="store_true", help="Run a quick test with reduced grid and data.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs (default 1).")
    args = parser.parse_args()

    fill_gas = args.fill_gas.upper()
    
    # --- 1. Load Data ---
    print(f"--- Loading {fill_gas} Data (Early Stopping Version) ---")
    try:
        df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
        df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')
    except FileNotFoundError:
        print(f"Error: Data files for {fill_gas} not found.")
        return

    # Extract Labels
    df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
    df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
    y_train = df_train['label'].values
    y_test = df_test['label'].values

    # Extract Features
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train = df_train[spectral_cols].values
    X_test = df_test[spectral_cols].values
    
    if args.quick:
        print("--- QUICK MODE: Subsampling data ---")
        X_train = X_train[:500]
        y_train = y_train[:500]

    # --- 2. Define Pipeline ---
    pipeline = Pipeline([
        ('scaler_raw', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('scaler_pca', StandardScaler()),
        ('mlp', MLPClassifier(input_dim=100, verbose=0))
    ])

    # --- 3. Define Grid ---
    if args.quick:
        param_grid = {
            'mlp__epochs': [20],
            'mlp__batch_size': [32],
            'mlp__learning_rate': [0.001],
            'mlp__dropout_rate': [0.4]
        }
    else:
        # Full Grid
        # We set epochs high (100) and rely on Early Stopping to find the best point.
        param_grid = {
            'mlp__epochs': [100], 
            'mlp__batch_size': [64, 128],
            'mlp__learning_rate': [0.001, 0.0005],
            'mlp__dropout_rate': [0.3, 0.4, 0.5]
        }

    # --- 4. Run GridSearchCV ---
    print(f"--- Starting GridSearchCV (n_jobs={args.n_jobs}) ---")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        scoring='accuracy',
        n_jobs=args.n_jobs,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    # --- 5. Results ---
    print("\n--- Grid Search Complete ---")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    print("Best Parameters:")
    print(grid_search.best_params_)

    # Retrain on full training set is done automatically by GridSearchCV refit=True
    # BUT: The refit model also uses validation_split=0.2 and EarlyStopping inside fit().
    # This means even the final model keeps 20% of the FULL data as validation.
    # This effectively mimics the original script's behavior perfectly.
    
    best_model = grid_search.best_estimator_
    
    print("\n--- Evaluating on Test Set ---")
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Set Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Non-Bio', 'Bio'])
    print(report)

    # --- 6. Save Report ---
    results_path = 'final_results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    report_filename = f'{fill_gas}_3layer_mlp_earlystopping_report.txt'
    with open(os.path.join(results_path, report_filename), 'w') as f:
        f.write(f"GridSearchCV with Early Stopping Results ({fill_gas})\n")
        f.write("=========================================================\n")
        f.write(f"Best CV Accuracy: {grid_search.best_score_:.4f}\n")
        f.write(f"Best Parameters: {json.dumps(grid_search.best_params_, indent=2)}\n\n")
        f.write(f"Test Set Accuracy: {test_acc:.4f}\n\n")
        f.write("Full Grid Results (Top 10):\n")
        
        cv_results = pd.DataFrame(grid_search.cv_results_)
        top_results = cv_results.sort_values('mean_test_score', ascending=False).head(10)
        
        for idx, row in top_results.iterrows():
            f.write(f"Params: {row['params']}, Mean Acc: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})\n")
            
        f.write("\nClassification Report:\n")
        f.write(report)

    print(f"Report saved to {os.path.join(results_path, report_filename)}")

if __name__ == "__main__":
    main()
