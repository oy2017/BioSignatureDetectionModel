"""Bin native-resolution spectra onto an observing configuration.

bin_native(X, wl_native, edges): flux-average of the native points falling in
each bin (the same operation the old domain_shift_sweep.py used for its
resolution family, and what ArielRad does when it bins pixels to the delivered
R). Native points are R ~ 1000, so even the R = 100 AIRS-CH0 bins hold ~10
points; the coarsest Ariel bins (R = 30) hold ~30.

Usage:
    python bin_spectra.py            # writes {split}_{config}.npy for every split and config
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SPLITS = ["train"] + [f"test{k}" for k in range(1, 6)]


def bin_native(X, wl_native, edges):
    """Mean of the piecewise-linear native spectrum over each bin:
    (1/dl) * integral y(l) dl between the bin edges. Exact for the linear
    interpolant, so it does not depend on where the native points fall
    relative to the edges (a plain mean of the points inside a bin does, and
    differed from MultiREx's FluxBinner by up to 1e-2 at sharp features)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    wl = np.asarray(wl_native, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges[0] < wl[0] - 1e-9 or edges[-1] > wl[-1] + 1e-9:
        # clamp the outermost edges to the native range (flat extrapolation)
        edges = edges.copy()
        edges[0] = max(edges[0], wl[0])
        edges[-1] = min(edges[-1], wl[-1])
    # cumulative trapezoid integral of each spectrum along wl
    dw = np.diff(wl)
    cum = np.zeros_like(X)
    cum[:, 1:] = np.cumsum(0.5 * (X[:, 1:] + X[:, :-1]) * dw, axis=1)
    # integral up to an arbitrary point = cum at the left native point + partial trapezoid
    idx = np.clip(np.searchsorted(wl, edges, side="right") - 1, 0, len(wl) - 2)
    frac = (edges - wl[idx]) / dw[idx]
    y_edge = X[:, idx] + (X[:, idx + 1] - X[:, idx]) * frac
    cum_edge = cum[:, idx] + 0.5 * (X[:, idx] + y_edge) * (edges - wl[idx])
    width = np.diff(edges)
    if np.any(width <= 0):
        raise ValueError("bin edges must be strictly increasing")
    return (np.diff(cum_edge, axis=1) / width).astype(np.float32)


def load_configs():
    with open(os.path.join(HERE, "ariel_bins.json")) as f:
        return json.load(f)


def main():
    cfg = load_configs()
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    for split in SPLITS:
        p = os.path.join(DATA, f"{split}_native.npy")
        if not os.path.exists(p):
            continue
        X = np.load(p)
        for name, c in cfg.items():
            Y = bin_native(X, wl, np.array(c["edges"]))
            np.save(os.path.join(DATA, f"{split}_{name}.npy"), Y)
        print(f"{split}: {X.shape[0]} spectra -> " +
              ", ".join(f"{k} {v['n_bins']}" for k, v in cfg.items()))


if __name__ == "__main__":
    main()
