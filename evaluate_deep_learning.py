"""
This script is designed to evaluate deep learning models (Convolutional Neural Networks - CNNs and Multi-Layer Perceptrons - MLPs) for biosignature detection from exoplanet spectra. It handles data loading, preprocessing (scaling, PCA), model building, training, and evaluation. The script allows for flexible configuration of PCA components, training hyperparameters, and model architecture.

Key functionalities include:
- Loading spectral data for a specified fill gas (e.g., H2, N2).
- Applying StandardScaler for feature scaling.
- Optionally applying Principal Component Analysis (PCA) for dimensionality reduction and feature engineering.
- Building either a 1D CNN or a Dense (MLP) neural network.
- Training the selected model with configurable epochs, batch size, and learning rate.
- Evaluating the model's performance using accuracy, classification report, and confusion matrix.
- Saving evaluation results (report and confusion matrix plot) to a specified directory.
"""
import os
# Set environment variable to avoid OpenMP runtime conflict (deadlock)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
import argparse
import random

# --- Set Random Seeds for Reproducibility ---
# Setting seeds for Python, NumPy, and TensorFlow to ensure that results are reproducible
# across different runs, which is crucial for consistent experimentation and debugging.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Create Results Directory ---
# Defines the path for saving experiment results (confusion matrices, classification reports).
# Creates the directory if it doesn't already exist.
results_path = 'final_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# --- Add argument parser ---
# --- Add argument parser ---
# Sets up command-line argument parsing for flexible script execution.
parser = argparse.ArgumentParser(description="Evaluate a dataset with a specified fill gas using a deep learning model (CNN or MLP).")
parser.add_argument("fill_gas", type=str,
                    help="The fill gas for the atmosphere (e.g., H2, N2). This determines which dataset parquet files are loaded.")
parser.add_argument("--pca_start_idx", type=int, default=0,
                    help="0-based index of the first principal component to include. Set to -1 to use raw spectra instead of PCA.")
parser.add_argument("--pca_end_idx", type=int, default=None,
                    help="0-based index of the last principal component to include (exclusive). If None, uses all components from pca_start_idx to the end. Only applicable if pca_start_idx is not -1.")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs. An epoch is one complete pass through the entire training dataset.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for training. Number of samples per gradient update. Smaller batches provide more frequent updates but noisier gradients.")
parser.add_argument("--learning_rate", type=float, default=0.001,
                    help="Learning rate for the Adam optimizer. Controls the step size at each iteration while moving towards a minimum of the loss function.")
parser.add_argument("--model_type", type=str, required=True, choices=["cnn", "mlp"],
                    help="Model architecture to use: 'cnn' (Convolutional Neural Network) or 'mlp' (Multi-Layer Perceptron/Dense Network).")
args = parser.parse_args()

fill_gas = args.fill_gas.upper()
pca_start_idx = args.pca_start_idx
pca_end_idx = args.pca_end_idx
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
model_type = args.model_type

# --- 1. Load and Prepare Data ---
# Loads training and testing datasets from parquet files based on the specified fill gas.
# Converts the 'biosignature' column into a numerical 'label' (0 for 'no', 1 for 'yes').
# Identifies spectral columns using a regex pattern to exclude non-spectral metadata.
print(f"--- Loading and Preparing {fill_gas} Dataset for {model_type.upper()} ---")
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

df_train['label'] = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)
df_test['label'] = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0)

float_pattern = re.compile(r"^-?\d+\.\d+$") # Regex to identify columns that are floats (spectral data)
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]

X_train = df_train[spectral_cols] # Features (spectral data)
y_train = df_train['label']      # Target labels
X_test = df_test[spectral_cols]
y_test = df_test['label']

# --- 2. Preprocessing ---
# Shuffles the training data to ensure randomness before splitting into training and validation sets.
# Applies StandardScaler to normalize features (mean=0, variance=1), which is crucial for neural network performance.
# Optionally applies PCA for dimensionality reduction and re-scales PCA components if used.
# Reshapes the data to a 3D format (samples, timesteps, features) required by Keras 1D Convolutional layers.
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Initialize and apply StandardScaler to the raw spectral data.
# This scales each feature (wavelength) independently.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA if pca_start_idx is not -1 (indicating raw spectra should not be used).
if pca_start_idx != -1:
    print(f"--- Applying PCA from component {pca_start_idx} to {pca_end_idx if pca_end_idx is not None else 'end'} ---")
    # Perform PCA on the scaled data to capture principal components.
    pca_full = PCA() # Compute all components to allow selection of a subset.
    X_train_pca_full = pca_full.fit_transform(X_train_scaled)
    X_test_pca_full = pca_full.transform(X_test_scaled)

    # Select a subset of PCA components based on pca_start_idx and pca_end_idx.
    X_train_processed = X_train_pca_full[:, pca_start_idx:pca_end_idx]
    X_test_processed = X_test_pca_full[:, pca_start_idx:pca_end_idx]

    # --- Re-scale PCA components for NN ---
    # Re-scaling PCA components is critical for neural networks, as PCA components
    # have varying variances, and NNs perform best with normalized inputs.
    print("--- Re-scaling PCA components ---")
    scaler_pca = StandardScaler()
    X_train_processed = scaler_pca.fit_transform(X_train_processed)
    X_test_processed = scaler_pca.transform(X_test_processed)
else:
    # If pca_start_idx is -1, use the raw (but scaled) spectral data directly.
    print("--- Using raw spectra (scaled) ---")
    X_train_processed = X_train_scaled
    X_test_processed = X_test_scaled

# Reshape data for Keras models. 1D CNNs expect input in the format (samples, timesteps, features).
# Here, each timestep is a wavelength, and there's 1 feature per timestep (the intensity).
X_train_processed = X_train_processed.reshape(X_train_processed.shape[0], X_train_processed.shape[1], 1)
X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], X_test_processed.shape[1], 1)

# Define the input shape for the Keras model, excluding the batch size.
input_shape = (X_train_processed.shape[1], 1)

# --- 3. Build Model ---
# Constructs the neural network model based on the specified 'model_type' (MLP or CNN).
# Each architecture is designed for binary classification of biosignatures.
print(f"--- Building Model with input shape {input_shape} ---")

if model_type == 'mlp':
    print("--- Using MLP (Dense) Architecture ---")
    # Multi-Layer Perceptron (MLP) architecture, suitable for tabular or PCA-transformed data.
    # It treats input features as independent, making it robust to features without spatial correlation.
    model = Sequential([
        # Flattens the 3D input (samples, timesteps, features) into 2D (samples, timesteps * features)
        # for dense layers, as MLPs expect a 1D feature vector per sample.
        Flatten(input_shape=input_shape),
        
        # First Dense layer with 256 units. 'Dense' layers are fully connected layers.
        Dense(256),
        # Batch Normalization stabilizes and accelerates training by normalizing layer inputs.
        BatchNormalization(),
        # ReLU (Rectified Linear Unit) activation function, introduces non-linearity.
        Activation('relu'),
        # Dropout layer, randomly sets a fraction (0.4) of input units to 0 at each update during training
        # to prevent overfitting.
        Dropout(0.4),
        
        # Second Dense layer with 128 units.
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        
        # Third Dense layer with 64 units.
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        
        # Output layer with 1 unit and sigmoid activation for binary classification.
        # Sigmoid outputs a probability between 0 and 1.
        Dense(1, activation='sigmoid')
    ])
else:
    print("--- Using CNN Architecture ---")
    # 1D Convolutional Neural Network (CNN) architecture, suitable for sequential data like spectra.
    # It excels at capturing local patterns (e.g., spectral features) through convolutional filters.
    model = Sequential([
        # First 1D Convolutional layer.
        # 'filters': 64 - Number of convolution filters (output dimensionality).
        # 'kernel_size': 5 - Specifies the length of the 1D convolution window.
        # 'input_shape': Defines the shape of the input data (timesteps, features).
        # 'padding': 'same' - Ensures output length is the same as input length by padding.
        Conv1D(filters=64, kernel_size=5, input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        # MaxPooling1D reduces the dimensionality of the feature maps, reducing computational cost
        # and providing a form of translation invariance. 'pool_size': 2 means taking the maximum
        # over 2 elements.
        MaxPooling1D(pool_size=2),
        # Dropout layer with a rate of 0.3.
        Dropout(0.3),
        
        # Second 1D Convolutional layer with 128 filters.
        Conv1D(filters=128, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Flattens the 3D output of the convolutional layers into a 1D vector for the dense layers.
        Flatten(),
        # Dense layer with 100 units.
        Dense(100),
        BatchNormalization(),
        Activation('relu'),
        # Dropout layer with a rate of 0.5.
        Dropout(0.5),
        # Output Dense layer for binary classification.
        Dense(1, activation='sigmoid') # Binary classification
    ])

# Configure the model for training.
# 'optimizer': Adam is an adaptive learning rate optimization algorithm.
# 'loss': Binary Crossentropy is standard for binary classification problems.
# 'metrics': Accuracy is used to monitor the model's performance during training and evaluation.
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Prints a summary of the model architecture, including layer names, output shapes, and parameter counts.
model.summary()

# --- 4. Train Model ---
# Trains the compiled model using the processed training data.
# 'epochs': The total number of times the learning algorithm will work through the entire training dataset.
# 'batch_size': The number of samples that will be propagated through the network at once.
# 'validation_split': Reserves a fraction (0.2 or 20%) of the training data for validation during training.
# 'callbacks': EarlyStopping is used to stop training when a monitored metric (validation loss) has stopped improving.
#   'patience': 10 - Number of epochs with no improvement after which training will be stopped.
#   'restore_best_weights': True - Restores model weights from the epoch with the best value of the monitored metric.
print(f"--- Training {model_type.upper()} Model for {epochs} epochs with batch size {batch_size} and learning rate {learning_rate} ---")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_processed, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2, # Use 20% of training data for validation
    callbacks=[early_stopping],
    verbose=1
)

# --- 5. Evaluate Model ---
# Evaluates the trained model's performance on the unseen test dataset.
# Calculates the loss and accuracy on the test set.
loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Generates probability predictions for the test set.
y_pred_proba = model.predict(X_test_processed)
# Converts probabilities to binary class labels (0 or 1) using a threshold of 0.5.
y_pred = (y_pred_proba > 0.5).astype(int)

# --- 6. Save and Display Results ---
# Generates and saves a classification report and a confusion matrix plot for the model's performance.
# The filenames and plot titles are dynamically generated based on model parameters for easy identification.

pca_suffix = f'_pca_idx_{pca_start_idx}_{pca_end_idx if pca_end_idx is not None else "end"}' if pca_start_idx != -1 else '_raw_spectra'
model_suffix = f'_epochs_{epochs}_batch_{batch_size}_lr_{learning_rate}'
report_filename = os.path.join(results_path, f'{fill_gas}_{model_type.lower()}{pca_suffix}{model_suffix}_report.txt')
cm_filename = os.path.join(results_path, f'{fill_gas}_{model_type.lower()}{pca_suffix}{model_suffix}_confusion_matrix.png')
cm_title = f'Confusion Matrix on {fill_gas} ({model_type.upper()}, PCA {pca_start_idx}-{pca_end_idx if pca_end_idx is not None else "end"}, Epochs={epochs}, Batch={batch_size}, LR={learning_rate})' if pca_start_idx != -1 else f'Confusion Matrix on {fill_gas} ({model_type.upper()}, Raw Spectra, Epochs={epochs}, Batch={batch_size}, LR={learning_rate})'

# Generate and print the classification report, which includes precision, recall, f1-score, and support for each class.
report_str = classification_report(y_test, y_pred, target_names=['Non-Bio (0)', 'Bio (1)'])
with open(report_filename, 'w') as f:
    f.write(report_str)

# Generate and display the confusion matrix.
# A confusion matrix shows the number of correct and incorrect predictions made by the classification model
# compared to the actual outcomes (target values).
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # Visualizes the confusion matrix using a heatmap.
plt.title(cm_title) # Sets the title of the plot.
plt.savefig(cm_filename) # Saves the plot to a file.
plt.close() # Closes the plot to free up memory.

# Final print statement summarizing the analysis.
print(f"\n--- {fill_gas} Analysis Complete ({model_type.upper()}, PCA {pca_start_idx}-{pca_end_idx if pca_end_idx is not None else 'end'}, Epochs={epochs}, Batch={batch_size}, LR={learning_rate}) ---")
print(report_str)
