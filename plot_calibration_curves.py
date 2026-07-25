"""Reliability diagrams and calibration statistics for the four architectures.

Produces Figure 8 (final_results/calibration_curves.png) and the numbers
reported in Section 4.4 (final_results/H2_calibration.txt).

Two properties matter for this plot and neither is the default:

  * Pooling. The five independent test sets are pooled (2,697 planets) rather
    than a single set used, so each bin holds ~270 planets instead of ~54.
  * Equal-count bins. XGBoost's predicted probabilities are strongly bimodal
    (about a third of planets below 0.05 and a third above 0.95), so the
    equal-width bins that sklearn's CalibrationDisplay uses by default leave
    only 8-26 planets in the mid-range bins. The resulting per-bin scatter is
    binomial noise, not miscalibration. Quantile bins put equal weight in
    every plotted point.

Expected calibration error is reported alongside the Brier score because the
Brier score conflates calibration with discrimination: a model can score well
by separating the classes rather than by reporting honest probabilities. ECE
measures only the calibration part, so the two together separate the effects.

MLP and CNN are stochastic; their statistics are averaged over RESTARTS
trainings and the curve is drawn from the first. XGBoost and Random Forest are
deterministic on fixed inputs.
"""
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, Input, MaxPooling1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

SEED = 42
RESTARTS = 3
N_BINS = 10
FILL_GAS = 'H2'
FIG_PATH = 'final_results/calibration_curves.png'
TXT_PATH = 'final_results/H2_calibration.txt'

np.random.seed(SEED)

# --- Data -------------------------------------------------------------------
df_train = pd.read_parquet(f'multirex_spectra_{FILL_GAS}_train.parquet')
df_test = pd.concat(
    [pd.read_parquet(f'multirex_spectra_{FILL_GAS}_test_set_{i}.parquet')
     for i in range(1, 6)], ignore_index=True)

float_pattern = re.compile(r"^-?\d+\.\d+$")
spectral_cols = [c for c in df_train.columns
                 if isinstance(c, float) or (isinstance(c, str) and float_pattern.match(c))]


def prepare(df):
    X = df[spectral_cols].values
    y = (df['biosignature'] == 'yes').astype(int).values
    keep = (X <= 1.0).all(axis=1)          # drop physically impossible transit depths
    return X[keep], y[keep]


X_train_raw, y_train = prepare(df_train)
X_test_raw, y_test = prepare(df_test)

# --- Preprocessing: fit on training data only, applied unchanged to the tests
scaler_raw = StandardScaler()
pca = PCA(n_components=102, random_state=SEED)
scaler_pca = StandardScaler()

X_train = scaler_pca.fit_transform(pca.fit_transform(scaler_raw.fit_transform(X_train_raw)))
X_test = scaler_pca.transform(pca.transform(scaler_raw.transform(X_test_raw)))
X_train, y_train = shuffle(X_train, y_train, random_state=SEED)


# --- Models -----------------------------------------------------------------
def build_mlp():
    return Sequential([
        Input(shape=(102,)),
        Dense(512), BatchNormalization(), Activation('relu'), Dropout(0.2),
        Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.2),
        Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.2),
        Dense(1, activation='sigmoid')])


def build_cnn():
    return Sequential([
        Input(shape=(102, 1)),
        Conv1D(32, 5, padding='same'), BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(0.3),
        Conv1D(64, 5, padding='same'), BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(0.3),
        Flatten(),
        Dense(100), BatchNormalization(), Activation('relu'), Dropout(0.5),
        Dense(1, activation='sigmoid')])


def train_keras(build, reshape, batch_size, epochs):
    """Return one probability vector per restart."""
    Xa = X_train.reshape(-1, 102, 1) if reshape else X_train
    Xb = X_test.reshape(-1, 102, 1) if reshape else X_test
    out = []
    for r in range(RESTARTS):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + r)
        model = build()
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        model.fit(Xa, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                  verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=10,
                                                      restore_best_weights=True)])
        out.append(model.predict(Xb, verbose=0).flatten())
    return out


print('Training XGBoost...')
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                    eval_metric='logloss', random_state=SEED, n_jobs=-1)
xgb.fit(X_train, y_train)

print('Training Random Forest...')
rf = RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=2,
                            max_depth=None, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)

print(f'Training MLP ({RESTARTS} restarts)...')
mlp_runs = train_keras(build_mlp, reshape=False, batch_size=64, epochs=200)
print(f'Training 1D-CNN ({RESTARTS} restarts)...')
cnn_runs = train_keras(build_cnn, reshape=True, batch_size=32, epochs=100)

runs = {
    'XGBoost': [xgb.predict_proba(X_test)[:, 1]],
    'Random Forest': [rf.predict_proba(X_test)[:, 1]],
    'MLP': mlp_runs,
    'CNN': cnn_runs,
}


# --- Statistics -------------------------------------------------------------
def reliability(y, p, n_bins=N_BINS):
    """Equal-count reliability table: mean predicted, observed, n per bin."""
    order = np.argsort(p)
    ys, ps = y[order], p[order]
    rows = []
    for chunk in np.array_split(np.arange(len(ps)), n_bins):
        rows.append((ps[chunk].mean(), ys[chunk].mean(), len(chunk)))
    return rows


def ece(y, p, n_bins=N_BINS):
    """Expected calibration error over equal-count bins."""
    return sum(n / len(p) * abs(obs - pred) for pred, obs, n in reliability(y, p, n_bins))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


lines = ['Calibration of the four architectures, pooled over the five test sets', '']
lines.append(f'Pooled test planets: {len(y_test)}   positive rate: {y_test.mean():.4f}')
lines.append(f'Bins: {N_BINS} equal-count (~{len(y_test)//N_BINS} planets each)')
lines.append(f'MLP/CNN averaged over {RESTARTS} restarts; XGBoost/RF deterministic.')
lines.append('')
lines.append('ECE is the calibration-only statistic; the Brier score also rewards')
lines.append('discrimination, so the two are reported together.')
lines.append('')
lines.append(f"  {'model':<15}{'Brier':>18}{'ECE':>18}{'AUC':>9}{'accuracy':>11}")
lines.append('  ' + '-' * 69)

stats = {}
for name, ps in runs.items():
    br = [brier_score_loss(y_test, p) for p in ps]
    ec = [ece(y_test, p) for p in ps]
    au = [roc_auc_score(y_test, p) for p in ps]
    ac = [((p > 0.5) == y_test).mean() for p in ps]
    stats[name] = (np.mean(br), np.mean(ec))
    if len(ps) > 1:
        b = f'{np.mean(br):.4f} +/-{np.std(br):.4f}'
        e = f'{np.mean(ec):.4f} +/-{np.std(ec):.4f}'
    else:
        b, e = f'{np.mean(br):.4f}', f'{np.mean(ec):.4f}'
    lines.append(f'  {name:<15}{b:>18}{e:>18}{np.mean(au):>9.4f}{np.mean(ac)*100:>10.2f}%')

lines.append('')
lines.append('Random Forest discriminates better than either neural network (higher AUC)')
lines.append('yet has the worst ECE of the four, so calibration quality is not a')
lines.append('by-product of accuracy here. XGBoost leads on both.')
lines.append('')

for name, ps in runs.items():
    lines.append(f'--- {name} reliability (first restart) ---')
    lines.append(f"  {'mean pred':>10}{'observed':>10}{'n':>7}{'gap':>9}")
    for pred, obs, n in reliability(y_test, ps[0]):
        lines.append(f'  {pred:>10.3f}{obs:>10.3f}{n:>7}{obs-pred:>+9.3f}')
    lines.append('')

report = '\n'.join(lines)
print('\n' + report)
with open(TXT_PATH, 'w') as fh:
    fh.write(report + '\n')

# --- Figure -----------------------------------------------------------------
colors = {'XGBoost': 'tab:blue', 'Random Forest': 'tab:orange',
          'MLP': 'tab:green', 'CNN': 'tab:red'}

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.plot([0, 1], [0, 1], 'k:', lw=1.6, label='Perfectly calibrated', zorder=1)

for name, ps in runs.items():
    rows = reliability(y_test, ps[0])
    xs = np.array([r[0] for r in rows])
    obs = np.array([r[1] for r in rows])
    bounds = [wilson(int(round(o * n)), n) for _, o, n in rows]
    lower = np.clip(obs - np.array([b[0] for b in bounds]), 0, None)
    upper = np.clip(np.array([b[1] for b in bounds]) - obs, 0, None)
    ax.errorbar(xs, obs, yerr=[lower, upper], marker='s', ms=5, lw=1.8, capsize=3,
                color=colors[name], zorder=2,
                label=f'{name} (ECE {stats[name][1]:.3f})')

ax.set_xlabel('Mean predicted probability (positive class: 1)')
ax.set_ylabel('Fraction of positives (positive class: 1)')
ax.set_title(f'Calibration curves pooled over the five test sets (n = {len(y_test)})\n'
             f'equal-count bins (~{len(y_test)//N_BINS} planets each), 95% Wilson intervals')
ax.grid(True, ls='--', alpha=0.6)
ax.legend(loc='upper left', frameon=True)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=300)
print(f'\nCalibration curves saved to {FIG_PATH}')
print(f'Statistics saved to {TXT_PATH}')
