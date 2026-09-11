import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GaussianNoise
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SEED = 42
np.random.seed(SEED)

fill_gas = 'H2'
df_train = pd.read_parquet(f'multirex_spectra_{fill_gas}_train.parquet')
df_test = pd.read_parquet(f'multirex_spectra_{fill_gas}_test.parquet')

float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [col for col in df_train.columns if isinstance(col, float) or (isinstance(col, str) and float_pattern.match(col))]
X_train_raw = df_train[spectral_cols].values
X_test_raw = df_test[spectral_cols].values

y_train = df_train['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values
y_test = df_test['biosignature'].apply(lambda x: 1 if x == 'yes' else 0).values

train_mask = (X_train_raw <= 1.0).all(axis=1)
test_mask = (X_test_raw <= 1.0).all(axis=1)
X_train_raw = X_train_raw[train_mask]
y_train = y_train[train_mask]
X_test_raw = X_test_raw[test_mask]

scaler_raw = StandardScaler()
X_train_scaled = scaler_raw.fit_transform(X_train_raw)
X_test_scaled = scaler_raw.transform(X_test_raw)

X_train_cnn = X_train_scaled.reshape(-1, len(spectral_cols), 1)
X_test_cnn = X_test_scaled.reshape(-1, len(spectral_cols), 1)

print("Training 1D-CNN...")
cnn = Sequential([
    Input(shape=(len(spectral_cols), 1)),
    GaussianNoise(0.05),
    Conv1D(32, 5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
    Conv1D(64, 5, padding='same'), BatchNormalization(), Activation('relu'), MaxPooling1D(2), Dropout(0.3),
    Conv1D(128, 5, padding='same'), BatchNormalization(), Activation('relu'), GlobalAveragePooling1D(), Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_cnn, y_train, batch_size=64, epochs=50, verbose=0)
cnn_probs = cnn.predict(X_test_cnn, verbose=0).flatten()

print("Plotting histogram...")
plt.figure(figsize=(8, 5))
plt.hist(cnn_probs, bins=20, edgecolor='black', alpha=0.7)
plt.title('Histogram of 1D-CNN Predicted Probabilities')
plt.xlabel('Predicted Probability of Biosignature')
plt.ylabel('Frequency')
plt.savefig('final_results/cnn_prob_histogram.png')
print("Saved histogram to final_results/cnn_prob_histogram.png")
