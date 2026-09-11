"""v3 copy of v2/common.py. One change: the label column is `label_co` (C/O > 1.0), not
`biosignature`. DATA/RESULTS/MODELS resolve relative to this file, so everything under v3/
is self-contained; noise.py, bin_spectra.py, ariel_bins.py and pipeline.py are unmodified copies.
"""
"""Shared loading, noise, feature pipelines and metrics for the v2 study."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
SEED = 42
SNR = 15.0
NOISE_SHAPE = "ariel"
TESTS = [f"test{k}" for k in range(1, 6)]

import sys
sys.path.insert(0, HERE)
from noise import add_noise  # noqa: E402


def configs():
    with open(os.path.join(HERE, "ariel_bins.json")) as f:
        return json.load(f)


def base_config(config):
    """'ariel_abs50' -> 'ariel' (binning) ; plain names pass through."""
    return config.split("_abs")[0]


def noise_spec(config):
    """(shape, level_ppm): peak-to-peak Ariel-shaped noise unless the config name
    carries an '_abs<ppm>' suffix, which selects the absolute-floor convention."""
    if "_abs" in config:
        return "ariel_abs", float(config.split("_abs")[1])
    return NOISE_SHAPE, None


def edges(config):
    return np.array(configs()[base_config(config)]["edges"])


def centres(config):
    return np.array(configs()[base_config(config)]["centres"])


def load_split(split, config, noisy=True, snr=SNR, shape=None, seed_offset=0):
    """Binned spectra and parameters for one split. Noise is drawn
    deterministically from (split, seed_offset) so every script sees the same
    realisation of a given test set."""
    X = np.load(os.path.join(DATA, f"{split}_{base_config(config)}.npy")).astype(np.float64)
    P = pd.read_parquet(os.path.join(DATA, f"{split}_params.parquet"))
    y = P["label_co"].to_numpy()
    if noisy:
        seed = (1000 if split == "train" else 2000 + int(split[-1])) + seed_offset
        shp, lvl = noise_spec(config)
        if shape is not None:
            shp = shape
        X, _ = add_noise(X, P, centres(config), snr=snr, shape=shp, seed=seed, level_ppm=lvl)
    return X, y, P


def feature_amplitude(X):
    """Per-spectrum scatter of transit depth across bins (the old manuscript's
    'feature amplitude'), on the noise-free or noisy input as given."""
    return X.std(axis=1)


class Features:
    """Feature pipelines. All are fit on the training set only.
      raw   per-bin z-score
      pca   per-bin z-score -> PCA(k)               (tree models use this)
      pcaw  per-bin z-score -> PCA(k) -> per-component z-score ('whitened'; MLP)
      norm  per-spectrum (x - mean)/std -> per-bin z-score   (per-spectrum normalization)
    """

    def __init__(self, kind, n_components=None, var_target=0.9999):
        self.kind = kind
        self.k = n_components
        self.var_target = var_target
        self.s1 = StandardScaler()
        self.pca = None
        self.s2 = None

    @staticmethod
    def _norm(X):
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        return (X - mu) / sd

    def fit(self, X):
        Z = self._norm(X) if self.kind == "norm" else X
        Z = self.s1.fit_transform(Z)
        if self.kind in ("pca", "pcaw"):
            full = PCA(random_state=SEED).fit(Z)
            if self.k is None:
                cum = np.cumsum(full.explained_variance_ratio_)
                self.k = int(np.searchsorted(cum, self.var_target) + 1)
            self.pca = PCA(n_components=self.k, random_state=SEED).fit(Z)
            Z = self.pca.transform(Z)
            if self.kind == "pcaw":
                self.s2 = StandardScaler().fit(Z)
        return self

    def transform(self, X):
        Z = self._norm(X) if self.kind == "norm" else X
        Z = self.s1.transform(Z)
        if self.pca is not None:
            Z = self.pca.transform(Z)
            if self.s2 is not None:
                Z = self.s2.transform(Z)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def ece(p, y, n_bins=10):
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    p, y = p[order], y[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    e = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        e += len(b) / len(p) * abs(p[b].mean() - y[b].mean())
    return e


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "brier": brier_score_loss(y, p),
        "ece": ece(p, y),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
    }


def summarize(rows):
    """rows: list of metric dicts (one per test set) -> mean/sd dict."""
    df = pd.DataFrame(rows)
    return {k: (df[k].mean(), df[k].std(ddof=1)) for k in df.columns}


def fmt(ms):
    return "  ".join(f"{k} {m*100:.2f}±{s*100:.2f}" if k in ("accuracy", "precision", "recall", "f1")
                     else f"{k} {m:.4f}±{s:.4f}" for k, (m, s) in ms.items())
