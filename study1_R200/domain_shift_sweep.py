"""
Phase 1 domain-shift sweep: how does the frozen pipeline degrade when the test
distribution moves away from the training distribution?

Addresses Reviewer 1's requests for robustness testing under realistic domain
shift (R1-3), and isolates whether whitening amplifies observational artifacts
(R1-8).

Design invariants - these are what make this a domain-shift test rather than an
expensive noise study:

  * Models are trained ONCE on the clean training set and never retrained.
  * The raw scaler, the PCA basis, and the post-PCA scaler are fit on clean
    training data only and applied unchanged to every perturbed test set.
  * Only the test sets are perturbed.
  * Strengths are anchored to the measured noise floor of the data, so they are
    physically interpretable rather than arbitrary multiples of spectral scatter.
  * Calibration (Brier) is reported alongside accuracy. The manuscript claims
    calibrated triage, so calibration collapse is a failure even if accuracy holds.

Three models are compared so that architecture and whitening are not confounded:

    XGBoost (unwhitened)  - the recommended pipeline; scale-invariant control
    MLP (whitened)        - the manuscript's neural pipeline
    MLP (unwhitened)      - identical architecture and components, whitening off

The MLP pair differs ONLY in whitening, so any difference in their degradation
is attributable to whitening itself. That is the direct test of R1-8.

Perturbations are applied to raw transit-depth spectra before any scaling, and
are deliberately not clipped to [0, 1] - real observations are not clipped, and
clipping would mask the failure mode.

Usage:
    python domain_shift_sweep.py                        # perturbation sweep
    python domain_shift_sweep.py --mode extrapolation   # out-of-envelope split
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
SNR_BASE = 15.0
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
TEST_FILE_FMT = "multirex_spectra_H2_test_set_{}.parquet"

# Reference palette slots 1-3 (light mode). Magenta sits below 3:1 contrast on a
# light surface, so the relief rule applies: a legend is always present and the
# full numbers are written to CSV.
COLORS = {"XGBoost (unwhitened)": "#2a78d6",
          "MLP (whitened)": "#008300",
          "MLP (unwhitened)": "#e87ba4"}
MARKERS = {"XGBoost (unwhitened)": "o", "MLP (whitened)": "s",
           "MLP (unwhitened)": "^"}
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#d8d8d4"

# Noise families are expressed as the target effective SNR after extra noise is
# added. Systematic families are multiples of the measured noise floor, so "1.0"
# means a systematic comparable in size to the noise. Resolution is a target
# resolving power, with 0 meaning no resampling at all.
SWEEPS = {
    "white noise":           [15, 12, 10, 8, 5],
    "correlated noise":      [15, 12, 10, 8, 5],
    "gain ramp":             [0.0, 0.25, 0.5, 1.0, 2.0],
    "baseline offset":       [0.0, 0.25, 0.5, 1.0, 2.0],
    "resolution loss":       [0, 200, 150, 100, 75],
    "stellar contamination": [0.0, 0.25, 0.5, 1.0, 2.0],
}


def load_raw():
    df = pd.read_parquet(TRAIN_FILE)
    fp = re.compile(r"^-?\d+\.\d+$")
    cols = [c for c in df.columns
            if isinstance(c, float) or (isinstance(c, str) and fp.match(c))]
    cols = sorted(cols, key=float)
    wl = np.array([float(c) for c in cols])
    X = df[cols].values
    y = (df["biosignature"] == "yes").astype(int).values
    tests = []
    for i in range(1, 6):
        dt = pd.read_parquet(TEST_FILE_FMT.format(i))
        tests.append((dt[cols].values,
                      (dt["biosignature"] == "yes").astype(int).values))
    return X, y, tests, wl, df, cols


def noise_floor(X):
    """Per-spectrum noise level already present, via successive differences.

    Astrophysical structure is smooth between adjacent bins while injected noise
    is not, so the scatter of first differences isolates the noise. Divided by
    sqrt(2) because differencing two independent samples doubles the variance.
    """
    return np.diff(X, axis=1).std(axis=1, keepdims=True) / np.sqrt(2.0)


def bin_to_resolution(X, wl, R):
    """Flux-conserving rebin to resolving power R, mapped back to the input grid.

    Averages within log-spaced bins rather than interpolating twice. Double
    linear interpolation acts as a low-pass filter even when the target grid
    matches the input, which silently destroys the high-frequency structure the
    classifier depends on. Averaging leaves R = 200 a near-identity operation on
    a grid that is already R = 199, giving the family a true zero point.
    """
    n_bins = max(4, int(R * np.log(wl[-1] / wl[0])))
    edges = np.logspace(np.log10(wl[0]), np.log10(wl[-1]), n_bins + 1)
    idx = np.clip(np.digitize(wl, edges) - 1, 0, n_bins - 1)
    out = np.empty_like(X)
    for b in np.unique(idx):
        m = idx == b
        out[:, m] = X[:, m].mean(axis=1, keepdims=True)
    return out


def perturb(X, wl, family, strength, rng):
    sig_n = noise_floor(X)

    if family in ("white noise", "correlated noise"):
        if strength >= SNR_BASE:
            return X.copy()
        # Independent noise of amplitude m*sig_n gives SNR_eff = SNR_BASE/sqrt(1+m^2).
        m = np.sqrt((SNR_BASE / strength) ** 2 - 1.0)
        if family == "white noise":
            return X + rng.normal(0.0, 1.0, X.shape) * sig_n * m
        raw = rng.normal(0.0, 1.0, X.shape)
        smooth = gaussian_filter1d(raw, sigma=8.0, axis=1)
        smooth /= smooth.std(axis=1, keepdims=True) + 1e-12
        return X + smooth * sig_n * m

    if family == "resolution loss":
        return X.copy() if strength == 0 else bin_to_resolution(X, wl, strength)

    if strength == 0.0:
        return X.copy()

    if family == "gain ramp":
        amp = strength * sig_n / np.abs(X).mean(axis=1, keepdims=True)
        tilt = np.linspace(-1.0, 1.0, X.shape[1])
        signs = rng.choice([-1.0, 1.0], size=(len(X), 1))
        return X * (1.0 + signs * amp * tilt)

    if family == "baseline offset":
        return X + rng.normal(0.0, 1.0, (len(X), 1)) * sig_n * strength

    if family == "stellar contamination":
        shape = 1.0 / wl
        shape = (shape - shape.mean()) / shape.std()
        amp = strength * sig_n / np.abs(X).mean(axis=1, keepdims=True)
        signs = rng.choice([-1.0, 1.0], size=(len(X), 1))
        return X * (1.0 + signs * amp * shape)

    raise ValueError(family)


def build_mlp(dim):
    import tensorflow as tf
    from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                         Dropout, Input)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    tf.keras.backend.clear_session()
    m = Sequential()
    m.add(Input(shape=(dim,)))
    for u in [256, 128, 64]:
        m.add(Dense(u)); m.add(BatchNormalization())
        m.add(Activation("relu")); m.add(Dropout(0.3))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def score(y, prob):
    pred = (prob > 0.5).astype(int)
    return (accuracy_score(y, pred), f1_score(y, pred, zero_division=0),
            brier_score_loss(y, prob))


def run_sweep():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(SEED)

    X_train, y_train, tests, wl, _, _ = load_raw()

    scaler_raw = StandardScaler().fit(X_train)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(
        scaler_raw.transform(X_train))
    P_train = pca.transform(scaler_raw.transform(X_train))
    scaler_pca = StandardScaler().fit(P_train)

    def fit_mlp(P):
        Xs, ys = shuffle(P, y_train, random_state=SEED)
        m = build_mlp(N_COMPONENTS)
        m.fit(Xs, ys, epochs=200, batch_size=128, validation_split=0.2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)], verbose=0)
        return m

    Xs, ys = shuffle(P_train, y_train, random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                        subsample=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(Xs, ys)

    models = {
        "XGBoost (unwhitened)": (xgb, lambda P: P, False),
        "MLP (whitened)":       (fit_mlp(scaler_pca.transform(P_train)),
                                 lambda P: scaler_pca.transform(P), True),
        "MLP (unwhitened)":     (fit_mlp(P_train), lambda P: P, True),
    }

    rows = []
    for family, strengths in SWEEPS.items():
        for s in strengths:
            rng = np.random.default_rng(SEED)
            acc = {k: [] for k in models}
            bri = {k: [] for k in models}
            f1s = {k: [] for k in models}
            for X_raw, y in tests:
                P = pca.transform(scaler_raw.transform(
                    perturb(X_raw, wl, family, s, rng)))      # frozen pipeline
                for name, (mdl, prep, is_nn) in models.items():
                    Z = prep(P)
                    p = (mdl.predict(Z, verbose=0).ravel() if is_nn
                         else mdl.predict_proba(Z)[:, 1])
                    a, f, b = score(y, p)
                    acc[name].append(a); f1s[name].append(f); bri[name].append(b)
            row = dict(family=family, strength=s)
            for name in models:
                tag = name.split()[0].lower() + ("_w" if "(whitened)" in name else
                                                 "_u" if "MLP" in name else "")
                row[f"{tag}_acc"] = np.mean(acc[name])
                row[f"{tag}_f1"] = np.mean(f1s[name])
                row[f"{tag}_brier"] = np.mean(bri[name])
            rows.append(row)
            print(f"{family:22s} {s:>5} | " + " | ".join(
                f"{n.split('(')[0].strip()[:3]}{'W' if '(whitened)' in n else 'U' if 'MLP' in n else ''} "
                f"{np.mean(acc[n]):.2%} (B {np.mean(bri[n]):.3f})" for n in models))

    df = pd.DataFrame(rows)
    os.makedirs("final_results", exist_ok=True)
    df.to_csv("final_results/H2_domain_shift_sweep.csv", index=False)
    make_figures(df)
    summarise(df)
    return df


COLS = {"XGBoost (unwhitened)": "xgboost", "MLP (whitened)": "mlp_w",
        "MLP (unwhitened)": "mlp_u"}


# Panel x-axis units differ by family: effective SNR, multiples of the measured
# noise floor, or resolving power. Unlabelled axes made the sweep unreadable.
XLABELS = {
    "white noise":           "effective SNR",
    "correlated noise":      "effective SNR",
    "gain ramp":             "amplitude (x noise floor)",
    "baseline offset":       "amplitude (x noise floor)",
    "resolution loss":       "resolving power R (0 = unbinned R=200)",
    "stellar contamination": "amplitude (x noise floor)",
}

# This panel is the parametric 1/lambda proxy, not the physical TLSE model whose
# result the manuscript quotes; the titles collided without the qualifier.
PANEL_TITLES = {"stellar contamination": "stellar contamination (1/\u03bb proxy)"}


def make_figures(df):
    for metric, label, fname, low_better in [
        ("acc", "Accuracy", "domain_shift_accuracy.png", False),
        ("brier", "Brier score", "domain_shift_calibration.png", True),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), dpi=300)
        for ax, fam in zip(axes.ravel(), SWEEPS):
            d = df[df.family == fam]
            x = np.arange(len(d))
            for name, col in COLS.items():
                ax.plot(x, d[f"{col}_{metric}"], color=COLORS[name], linewidth=2,
                        marker=MARKERS[name], markersize=5, label=name)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" for v in d.strength], fontsize=8)
            # The three families use different units, so each panel states its own.
            ax.set_xlabel(XLABELS[fam], fontsize=8, color=C_MUTED)
            ax.set_title(PANEL_TITLES.get(fam, fam), fontsize=10, color=C_TEXT)
            ax.grid(True, color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)
            ax.tick_params(labelsize=8, colors=C_MUTED)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"): ax.spines[sp].set_color(C_GRID)
            if not low_better:
                ax.axhline(0.5, color=C_MUTED, linestyle="--", linewidth=1, alpha=0.6)
        for r in (0, 1):
            axes[r, 0].set_ylabel(label, fontsize=10, color=C_TEXT)
        axes[0, 0].legend(frameon=False, fontsize=7.5, labelcolor=C_TEXT)
        fig.suptitle(f"{label} under increasing perturbation strength"
                     f"{'  (lower is better)' if low_better else ''}",
                     fontsize=11, color=C_TEXT)
        fig.tight_layout()
        fig.savefig(f"final_results/{fname}", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def summarise(df):
    L = ["Domain-shift sweep: degradation from the clean baseline", ""]
    for fam in SWEEPS:
        d = df[df.family == fam]
        b, w = d.iloc[0], d.iloc[-1]
        L.append(f"{fam}")
        for name, col in COLS.items():
            L.append(f"    {name:22s} {b[col+'_acc']:.2%} -> {w[col+'_acc']:.2%} "
                     f"({w[col+'_acc']-b[col+'_acc']:+.2%})   "
                     f"Brier {b[col+'_brier']:.4f} -> {w[col+'_brier']:.4f}")
    L += ["", "=" * 72,
          "R1-8: does whitening itself reduce robustness?",
          "The two MLP rows share architecture and components and differ only in",
          "whitening, so their gap isolates whitening. Degradation is also given as",
          "a fraction of each model's headroom above chance, since models with a",
          "lower clean baseline have less room to fall.", ""]
    for fam in SWEEPS:
        d = df[df.family == fam]
        b, w = d.iloc[0], d.iloc[-1]
        parts = []
        for name, col in COLS.items():
            drop = b[col+"_acc"] - w[col+"_acc"]
            head = max(b[col+"_acc"] - 0.5, 1e-9)
            parts.append(f"{name.split('(')[0].strip()[:3]}"
                         f"{'W' if '(whitened)' in name else 'U' if 'MLP' in name else ''}"
                         f" {drop:+.1%} ({drop/head:.0%} of headroom)")
        gap = ((b["mlp_w_acc"] - w["mlp_w_acc"]) / max(b["mlp_w_acc"]-0.5, 1e-9)
               - (b["mlp_u_acc"] - w["mlp_u_acc"]) / max(b["mlp_u_acc"]-0.5, 1e-9))
        verdict = "whitening WORSE" if gap > 0.02 else (
            "whitening BETTER" if gap < -0.02 else "no clear difference")
        L.append(f"  {fam:22s} " + "  ".join(parts))
        L.append(f"  {'':22s} -> {verdict} (headroom gap {gap:+.0%})")
    out = "\n".join(L)
    print("\n" + out)
    with open("final_results/H2_domain_shift_sweep.txt", "w") as fh:
        fh.write(out + "\n")


def run_extrapolation():
    """Out-of-envelope split, with a matched-sample-size control.

    Training on a restricted radius range also means training on fewer planets,
    so a naive comparison confounds distribution shift with sample size. The
    control trains on a random subset of the same size drawn from the full
    distribution and tests in-distribution; the difference between the two
    isolates the extrapolation penalty.
    """
    X, y, _, wl, df, _ = load_raw()
    r = df["p_radius"].values
    cut = 15.0
    tr, te = r <= cut, r > cut
    n_tr, n_te = int(tr.sum()), int(te.sum())

    def fit_eval(Xtr, ytr, Xte, yte):
        sc = StandardScaler().fit(Xtr)
        pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(sc.transform(Xtr))
        Ptr, Pte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
        Xs, ys = shuffle(Ptr, ytr, random_state=SEED)
        m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                          subsample=0.8, eval_metric="logloss",
                          random_state=SEED, n_jobs=-1).fit(Xs, ys)
        return score(yte, m.predict_proba(Pte)[:, 1])

    a_ext, f_ext, b_ext = fit_eval(X[tr], y[tr], X[te], y[te])

    rng = np.random.default_rng(SEED)
    ctrl = []
    for rep in range(5):
        perm = rng.permutation(len(y))
        s_tr, s_te = perm[:n_tr], perm[n_tr:n_tr + n_te]
        ctrl.append(fit_eval(X[s_tr], y[s_tr], X[s_te], y[s_te]))
    a_ctl = np.mean([c[0] for c in ctrl]); b_ctl = np.mean([c[2] for c in ctrl])

    out = (
        f"Out-of-envelope generalisation, radius split at {cut} R_earth\n"
        f"  train n={n_tr}, test n={n_te}\n"
        f"  positive rate: in-envelope {y[tr].mean():.1%}, "
        f"out-of-envelope {y[te].mean():.1%}\n\n"
        f"  EXTRAPOLATION  train R<={cut}, test R>{cut} : "
        f"acc {a_ext:.2%}  F1 {f_ext:.2%}  Brier {b_ext:.4f}\n"
        f"  CONTROL        random split, same n={n_tr}   : "
        f"acc {a_ctl:.2%}  Brier {b_ctl:.4f}   (mean of 5 draws)\n\n"
        f"  Extrapolation penalty net of sample size: "
        f"{a_ext - a_ctl:+.2%} accuracy, {b_ext - b_ctl:+.4f} Brier\n\n"
        f"The control holds training-set size fixed and only removes the\n"
        f"distribution shift, so the difference between the two rows is the\n"
        f"cost of predicting outside the training envelope."
    )
    print("\n" + out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_extrapolation_split.txt", "w") as fh:
        fh.write(out + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sweep", "extrapolation"], default="sweep")
    a = p.parse_args()
    run_extrapolation() if a.mode == "extrapolation" else run_sweep()
