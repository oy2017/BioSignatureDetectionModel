"""Observational noise for binned, noise-free spectra.

Convention (kept from the old study so the two are comparable): MultiREx's
generate_df_SNR_noise defines the per-spectrum noise as

    sigma = (max - min of the noise-free spectrum on the observation grid) / SNR

i.e. SNR is referenced to the spectrum's peak-to-peak feature amplitude, not to
the mean transit depth. The old grid used SNR = 15 with white noise.

New here: the wavelength dependence of Ariel's noise. The per-bin sigma is
scaled by the ExoRad2 noise-to-signal shape for the star's temperature
(final_results/ariel_nsr_curves.npz, reconstructed Ariel payload; see
ariel_noise_model/build_ariel_nsr.py), normalised to unit median over the bins,
so the MEDIAN per-bin sigma equals (max - min)/SNR exactly as before and only the
colouring differs. shape="white" reproduces the old convention.

    add_noise(X, params, centres, snr=15, shape="ariel", seed=0) -> (Xnoisy, sigma)

X: (n, nb) noise-free binned depths; params: DataFrame with 's temperature';
centres: bin centres in microns. Deterministic for a given seed.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NSR_NPZ = os.path.join(os.path.dirname(HERE), "ariel_noise_model", "ariel_nsr_curves.npz")

_cache = {}


def nsr_shapes(centres):
    """Unit-median NSR shape per ExoRad Teff node, interpolated onto centres."""
    key = tuple(np.round(centres, 6))
    if key in _cache:
        return _cache[key]
    z = np.load(NSR_NPZ)
    teffs = z["teffs"]
    shapes = {}
    for T in teffs:
        wl, nsr = z[f"wl_{T}"], z[f"nsr_{T}"]
        o = np.argsort(wl)
        s = np.interp(centres, wl[o], nsr[o])          # flat extrapolation at the ends
        shapes[int(T)] = s / np.median(s)
    _cache[key] = (teffs, shapes)
    return teffs, shapes


def sigma_matrix(X, tstar, centres, snr=15.0, shape="ariel", level_ppm=None):
    """Per-bin noise sigma. shape: 'white' (flat, peak-to-peak/snr), 'ariel'
    (ExoRad-shaped, median = peak-to-peak/snr), 'ariel_abs' (ExoRad-shaped with
    an ABSOLUTE median level of level_ppm parts per million of transit depth,
    independent of the planet's feature amplitude: the realistic convention,
    under which small-amplitude planets are genuinely harder)."""
    X = np.asarray(X, dtype=np.float64)
    if shape == "ariel_abs":
        base = np.full(X.shape[0], float(level_ppm) * 1e-6)
    else:
        base = (X.max(axis=1) - X.min(axis=1)) / snr          # (n,)
    if shape == "white":
        return np.repeat(base[:, None], X.shape[1], axis=1)
    teffs, shapes = nsr_shapes(np.asarray(centres, dtype=float))
    node = teffs[np.argmin(np.abs(teffs[None, :] - np.asarray(tstar)[:, None]), axis=1)]
    S = np.vstack([shapes[int(t)] for t in node])
    return base[:, None] * S


def add_noise(X, params, centres, snr=15.0, shape="ariel", seed=0, level_ppm=None):
    sig = sigma_matrix(X, params["s temperature"].to_numpy(), centres, snr, shape, level_ppm)
    rng = np.random.default_rng(seed)
    return np.asarray(X, dtype=np.float64) + rng.normal(0.0, 1.0, sig.shape) * sig, sig


def add_correlated_noise(X, params, centres, snr=15.0, seed=0, rho=0.9):
    """Time-correlated systematics as in the old sweep: AR(1) across adjacent
    bins with coefficient rho, scaled to the same per-spectrum sigma."""
    sig = sigma_matrix(X, params["s temperature"].to_numpy(), centres, snr, "white")
    rng = np.random.default_rng(seed)
    n, nb = sig.shape
    e = rng.normal(0.0, 1.0, (n, nb))
    z = np.empty_like(e)
    z[:, 0] = e[:, 0]
    for i in range(1, nb):
        z[:, i] = rho * z[:, i - 1] + np.sqrt(1 - rho**2) * e[:, i]
    return np.asarray(X, dtype=np.float64) + z * sig, sig
