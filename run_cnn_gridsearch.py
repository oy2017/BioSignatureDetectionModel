import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Input, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils import shuffle
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
    return X_train_final, y_train

def build_cnn_custom(learning_rate=0.001, dropout_rate=0.3, filters=(64, 128), kernel_size=5):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Input(shape=(102, 1)))
    for f in filters:
        model.add(Conv1D(filters=f, kernel_size=kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_gridsearch_cnn():
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    X_train, y_train = get_data(0, 102)

    print("--- Running CNN GridSearch ---")
    cnn_model = KerasClassifier(model=build_cnn_custom, verbose=0, epochs=50, batch_size=64)
    cnn_param_grid = {
        'model__learning_rate': [0.001, 0.0005],
        'model__dropout_rate': [0.3, 0.5],
        'model__filters': [(32, 64), (64, 128)],
        'model__kernel_size': [3, 5],
        'batch_size': [32, 64]
    }
    cnn_grid = GridSearchCV(estimator=cnn_model, param_grid=cnn_param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)
    cnn_grid.fit(X_train.reshape(-1, 102, 1), y_train)
    print(f"CNN Best Params: {cnn_grid.best_params_}")
    print(f"CNN Best CV Acc: {cnn_grid.best_score_:.4f}\n")

if __name__ == "__main__":
    run_gridsearch_cnn()
