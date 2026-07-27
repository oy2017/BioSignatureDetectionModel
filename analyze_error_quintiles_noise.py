"""Where do injected-noise errors fall in physical-parameter space?

Section 4.3 bins the *clean* pipeline's errors by physical parameter and finds
a U-shaped dependence, with the low-feature-amplitude / low-scale-height tail
carrying the most error. Section 4.6 shows that injected noise degrades the
*aggregate* accuracy and attributes it to feature-amplitude suppression. This
script joins the two: it re-runs the §4.3 quintile analysis under injected
white and correlated noise and reports, per bin, how much error the noise
*adds* — testing whether the added errors concentrate on the already-marginal
planets (the amplitude-suppression mechanism at the planet level) or spread
evenly.

Pipeline is identical to analyze_error_quintiles.py (all 102 components,
tuned XGBoost, preprocessing fit on the training set only). Noise is injected
exactly as domain_shift_sweep.py does: amplitude m*sigma_n where
sigma_n is the successive-difference noise floor and
m = sqrt((SNR_BASE/SNR)^2 - 1); correlated noise is Gaussian-smoothed (sigma=8)
and renormalised. Bins use the clean-data quintile edges so clean and perturbed
rates are directly comparable, and each perturbed rate is averaged over five
noise realisations.
"""
import re

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
FILL_GAS = 'H2'
N_BINS = 5
SNR_BASE = 15
N_REAL = 5
OUT = 'final_results/H2_error_quintiles_noise.txt'

np.random.seed(SEED)

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
    keep = (X <= 1.0).all(axis=1)
    return X[keep], y[keep], df[keep].reset_index(drop=True)


X_train_raw, y_train, _ = prepare(df_train)
X_test_raw, y_test, meta = prepare(df_test)

scaler_raw = StandardScaler()
pca = PCA(n_components=102, random_state=SEED)
scaler_pca = StandardScaler()
scaler_raw.fit(X_train_raw)
X_train = scaler_pca.fit_transform(pca.fit_transform(scaler_raw.transform(X_train_raw)))
X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                      eval_metric='logloss', random_state=SEED, n_jobs=-1)
model.fit(X_train, y_train)


def transform(Xraw):
    return scaler_pca.transform(pca.transform(scaler_raw.transform(Xraw)))


def noise_floor(X):
    return np.diff(X, axis=1).std(axis=1, keepdims=True) / np.sqrt(2.0)


def perturb(X, family, snr, rng):
    if snr >= SNR_BASE:
        return X.copy()
    sig = noise_floor(X)
    m = np.sqrt((SNR_BASE / snr) ** 2 - 1.0)
    if family == 'white':
        return X + rng.normal(0.0, 1.0, X.shape) * sig * m
    raw = rng.normal(0.0, 1.0, X.shape)
    smooth = gaussian_filter1d(raw, sigma=8.0, axis=1)
    smooth /= smooth.std(axis=1, keepdims=True) + 1e-12
    return X + smooth * sig * m


err_clean = (model.predict(transform(X_test_raw)) != y_test).astype(float)

amplitude = X_test_raw.std(axis=1)
scale_height = (meta['atm temperature'].values.astype(float)
                * meta['p_radius'].values.astype(float) ** 2
                / meta['p_mass'].values.astype(float))
axes = [
    ('atmospheric temperature (K)', meta['atm temperature'].values.astype(float)),
    ('planet radius (R_earth)', meta['p_radius'].values.astype(float)),
    ('planet mass (M_earth)', meta['p_mass'].values.astype(float)),
    ('star temperature (K)', meta['s temperature'].values.astype(float)),
    ('scale-height proxy T R^2 / M', scale_height),
    ('feature amplitude (clean spectral scatter)', amplitude),
]

conditions = [('white', 10), ('white', 5), ('correlated', 5)]
perturbed = {}
for family, snr in conditions:
    reals = []
    for r in range(N_REAL):
        rng = np.random.default_rng(SEED + r)
        reals.append(model.predict(transform(perturb(X_test_raw, family, snr, rng))) != y_test)
    perturbed[(family, snr)] = np.mean(reals, axis=0)  # per-planet error probability

lines = ['Error rate against physical parameters under injected noise', '']
lines.append(f'Pooled test planets: {len(y_test)}   clean accuracy: {100*(1-err_clean.mean()):.2f}%')
lines.append(f'Bins: {N_BINS} clean-data quintiles. XGBoost, all 102 components.')
lines.append(f'Perturbed rates averaged over {N_REAL} noise realisations; bin edges from clean data.')
lines.append('Aggregate accuracy under each condition:')
for (family, snr), ep in perturbed.items():
    lines.append(f'   {family} noise SNR {SNR_BASE}->{snr}: {100*(1-ep.mean()):.2f}%')
lines.append('')

hdr = f"  {'quintile':>9}{'clean':>9}" + ''.join(f"{f[:3]+str(s):>10}" for f, s in conditions) \
    + ''.join(f"{'d '+f[:3]+str(s):>10}" for f, s in conditions)
for name, v in axes:
    edges = np.quantile(v, np.linspace(0, 1, N_BINS + 1))
    lines.append(f'--- {name} ---')
    lines.append(hdr)
    deltas = {c: [] for c in conditions}
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        sel = (v >= lo) & (v <= hi) if i == N_BINS - 1 else (v >= lo) & (v < hi)
        c = err_clean[sel].mean()
        row = f"  {i+1:>9}{100*c:>8.1f}%"
        for cond in conditions:
            row += f"{100*perturbed[cond][sel].mean():>9.1f}%"
        for cond in conditions:
            d = perturbed[cond][sel].mean() - c
            deltas[cond].append(d)
            row += f"{100*d:>+9.1f}"
        lines.append(row)
    # concentration: ratio of the added error in the lowest quintile to the mean added error
    for cond in conditions:
        arr = np.array(deltas[cond])
        conc = arr[0] / arr.mean() if arr.mean() > 0 else float('nan')
        lines.append(f'  added error (SNR->{cond[1]} {cond[0]}) lowest-quintile / mean = {conc:.2f}x')
    lines.append('')

lines.append('Reading:')
lines.append('  "d" columns are perturbed minus clean error rate per bin. If injected')
lines.append('  noise acted uniformly the added error would be flat across quintiles;')
lines.append('  a ratio >1 in the lowest quintile of feature amplitude / scale height')
lines.append('  means the noise concentrates its damage on the already-marginal planets,')
lines.append('  i.e. amplitude suppression operating planet-by-planet rather than evenly.')

report = '\n'.join(lines)
print(report)
with open(OUT, 'w') as fh:
    fh.write(report + '\n')
print(f'\nWritten to {OUT}')
