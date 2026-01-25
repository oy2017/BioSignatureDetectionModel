import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.models import load_model

# --- Set Environment ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    parser = argparse.ArgumentParser(description="Evaluate Best MLP on 5 Independent Test Sets.")
    parser.add_argument("fill_gas", type=str, help="The fill gas (e.g., H2).")
    args = parser.parse_args()
    
    fill_gas = args.fill_gas.upper()
    model_path = f'final_results/{fill_gas}_best_mlp_model.keras'
    
    # --- 1. Load Model ---
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
        
    print(f"--- Loading Model: {model_path} ---")
    model = load_model(model_path)
    
    # --- 2. Load Training Data (Only for Fitting Scalers/PCA) ---
    print(f"--- Loading Training Data (for Preprocessing Fit) ---")
    df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
    float_pattern = re.compile(r"^-?\d+\.\d+$") 
    spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
    X_train_raw = df_train[spectral_cols].values
    
    # Fit Preprocessing Pipeline
    print("--- Fitting Preprocessing Pipeline ---")
    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)
    
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    scaler_pca = StandardScaler()
    scaler_pca.fit(X_train_pca)
    
    # --- 3. Evaluate on 5 Test Sets ---
    metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }
    
    print(f"\n--- Evaluating on 5 Independent Test Sets (Labeled 1-5) ---")
    
    for i in range(1, 6):
        filename = f'multirex_spectra_{fill_gas}_test_set_{i}.parquet'
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue
            
        print(f"\nProcessing Set {i}: {filename}")
        df_test = pd.read_parquet(filename)
        
        y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
        X_test_raw = df_test[spectral_cols].values
        
        # Apply Preprocessing
        X_test_scaled = scaler_raw.transform(X_test_raw)
        X_test_pca = pca.transform(X_test_scaled)
        X_test_final = scaler_pca.transform(X_test_pca)
        
        # Predict
        y_pred_prob = model.predict(X_test_final, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        # Using 'weighted' average to account for potential slight class imbalances
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["F1-Score"].append(f1)
        
        print(f"  -> Accuracy:  {acc:.4f}")
        print(f"  -> Precision: {prec:.4f}")
        print(f"  -> Recall:    {rec:.4f}")
        print(f"  -> F1-Score:  {f1:.4f}")
        
    # --- 4. Statistics ---
    if metrics["Accuracy"]:
        print("\n" + "="*50)
        print(f"Aggregate Results across {len(metrics['Accuracy'])} Test Sets")
        print("="*50)
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10}")
        print("-" * 41)
        
        for key in metrics:
            mean_val = np.mean(metrics[key])
            std_val = np.std(metrics[key])
            print(f"{key:<15} | {mean_val:.4f}     | {std_val:.4f}")
            
        print("="*50)
        
        # Save Report
        report_path = f'final_results/{fill_gas}_best_mlp_5sets_full_stats.txt'
        with open(report_path, 'w') as f:
            f.write(f"Best MLP Model Full Statistics on 5 Sets ({fill_gas})\n")
            f.write("==================================================\n")
            for i in range(len(metrics["Accuracy"])):
                f.write(f"Set {i+1}:\n")
                for key in metrics:
                    f.write(f"  {key:<10}: {metrics[key][i]:.4f}\n")
                f.write("\n")
            
            f.write("Aggregate Stats:\n")
            f.write(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10}\n")
            f.write("-" * 41 + "\n")
            for key in metrics:
                mean_val = np.mean(metrics[key])
                std_val = np.std(metrics[key])
                f.write(f"{key:<15} | {mean_val:.4f}     | {std_val:.4f}\n")
            
        print(f"Full stats saved to {report_path}")

if __name__ == "__main__":
    main()
