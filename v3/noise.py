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
# v3: the same curve structure from the Ariel consortium's ExoSim2 on the reconstructed
# Ariel-like payload (v3/exosim_noise_grid.py); selected with shape="exosim"
EXOSIM_NPZ = os.path.join(HERE, "results", "exosim_nsr_curves.npz")

_cache = {}


PHOT_EDGES = [(0.50, 0.60), (0.60, 0.80), (0.80, 1.10)]      # VISPhot, FGS1, FGS2
SPEC_BOUNDS = [1.10, 1.95, 3.90, 7.80]                          # NIRSpec | AIRS-CH0 | AIRS-CH1


def _instrument(x):
    """0,1,2 = photometers; 3,4,5 = NIRSpec, AIRS-CH0, AIRS-CH1 (by wavelength)."""
    x = np.asarray(x, float)
    return np.where(x < 0.60, 0, np.where(x < 0.80, 1, np.where(x < 1.10, 2,
                    np.where(x < 1.95, 3, np.where(x < 3.90, 4, 5)))))


def nsr_binned(edges, npz=NSR_NPZ):
    """Absolute 1-hour NSR per target bin for every Teff node.

    Each noise-curve bin i (photometers: their band; spectrometers: width from the centre spacing
    within the instrument) carries a variance density v_i = nsr_i^2 * width_i. A target bin
    collects information 1/sigma^2 = integral of d(lambda) / v(lambda) over its extent, with v
    piecewise constant on the nearest noise-curve bin of the same instrument (photon-limited
    scaling). Returns (teffs, {T: nsr per target bin})."""
    edges = np.asarray(edges, float); key = ("binned", npz, tuple(np.round(edges, 6)))
    if key in _cache:
        return _cache[key]
    z = np.load(npz); teffs = z["teffs"]; out = {}
    fine = np.exp(np.linspace(np.log(edges[0]), np.log(edges[-1]), 60001))
    fine = 0.5 * (fine[1:] + fine[:-1]); dl = np.diff(np.exp(np.linspace(np.log(edges[0]), np.log(edges[-1]), 60001)))
    inst_fine = _instrument(fine)
    which = np.clip(np.searchsorted(edges, fine, side="right") - 1, 0, edges.size - 2)
    for T in teffs:
        T = int(T); wl, nsr = z[f"wl_{T}"], z[f"nsr_{T}"]; o = np.argsort(wl); wl, nsr = wl[o], nsr[o]
        inst = _instrument(wl); width = np.empty(wl.size)
        for k in range(3):
            width[inst == k] = PHOT_EDGES[k][1] - PHOT_EDGES[k][0]
        for k in range(3, 6):
            j = np.where(inst == k)[0]
            width[j] = np.gradient(wl[j]) if j.size > 1 else (SPEC_BOUNDS[k - 2] - SPEC_BOUNDS[k - 3])
        v = nsr ** 2 * width
        info = np.zeros(edges.size - 1)
        for k in range(6):
            j = np.where(inst == k)[0]; m = inst_fine == k
            if not m.any():
                continue
            lj = np.log(wl[j]); pos = np.clip(np.searchsorted(lj, np.log(fine[m])), 1, max(j.size - 1, 1)) if j.size > 1 else np.zeros(m.sum(), int)
            if j.size > 1:
                left = pos - 1; nearest = np.where(np.abs(np.log(fine[m]) - lj[left]) <= np.abs(np.log(fine[m]) - lj[pos]), left, pos)
            else:
                nearest = pos
            info += np.bincount(which[m], weights=dl[m] / v[j][nearest], minlength=edges.size - 1)
        out[T] = 1.0 / np.sqrt(info)
    _cache[key] = (teffs, out)
    return _cache[key]


MCS_CSV = os.path.join(HERE, "data", "mcs", "Ariel_MCS_Known_2026-05-11.csv")
MCS_PARAMS = os.path.join(HERE, "data", "mcs", "mcs_params.parquet")


def _mcs_hosts():
    import pandas as pd
    d = pd.read_csv(MCS_CSV); d["key"] = d["Planet Name"].astype(str).str.replace(" ", "")
    M = pd.read_parquet(MCS_PARAMS); M["key"] = M.name.astype(str).str.replace(" ", "")
    return M.merge(d[["key", "Star Distance [pc]"]].drop_duplicates("key"), on="key", how="left")


def radiometric_sigma(P, edges, transits="tier1_transits", k2=None, npz=None):
    """Absolute per-bin transit-depth sigma for known Ariel targets from the payload noise model.

    ExoSim 2's 1-hour NSR for the reference host (1.18 R_sun at 47.5 pc) at each Teff node, binned
    by nsr_binned, scaled photon-limited to the target (distance / 47.5 pc x 1.18 R_sun / R_star),
    to one transit with equal in- and out-of-transit time (sqrt(2 / T14[h])), and divided by
    sqrt(N) for the catalogue's integer number of transits of the named tier. The global level k2
    is calibrated once so that SNR 7 on the 5-scale-height modulation (median Tier-3 bin) reproduces
    the catalogue's Tier-3 transit counts (targets with N3 >= 5; 0.07 dex scatter).
    P needs a 'name' column matching the MCS; returns (sigma (n, nbins), k2)."""
    npz = npz or EXOSIM_NPZ; M = _mcs_hosts()

    def raw(Q, e):
        teffs_b, absn = nsr_binned(np.asarray(e, float), npz)
        node = np.argmin(np.abs(teffs_b[None, :] - Q["s temperature"].to_numpy()[:, None]), axis=1)
        out = np.vstack([absn[int(teffs_b[i])] for i in node])
        scale = (Q["Star Distance [pc]"].to_numpy() / 47.5) * (1.18 / Q["s radius"].to_numpy()) * np.sqrt(2.0 / (Q.t14_s.to_numpy() / 3600))
        return out * scale[:, None]
    if k2 is None:
        from common import configs
        e3 = np.asarray(configs()["ariel"]["edges"], float); s = raw(M, e3); N3 = M.tier3_transits.to_numpy(float); ok = N3 >= 5
        k2 = float(10 ** np.median(np.log10(N3[ok] / (7 * np.median(s[ok], 1) / M.modulation_5H.to_numpy()[ok]) ** 2)))
    import pandas as pd
    key = pd.DataFrame({"key": P["name"].astype(str).str.replace(" ", "").to_numpy()})
    Q = key.merge(M[["key", "Star Distance [pc]", "s radius", "s temperature", "t14_s", transits]].drop_duplicates("key"), on="key", how="left")
    if Q[transits].isna().any() or Q["Star Distance [pc]"].isna().any():
        raise ValueError("radiometric_sigma: targets missing from the MCS catalogue")
    return np.sqrt(k2) * raw(Q, edges) / np.sqrt(Q[transits].to_numpy(float))[:, None], k2


def _edges_for(centres):
    from common import configs
    for name, cfg in configs().items():
        c = np.asarray(cfg["centres"], float)
        if c.shape == centres.shape and np.allclose(c, centres, rtol=1e-6):
            return np.asarray(cfg["edges"], float)
    return None


def nsr_shapes(centres, npz=NSR_NPZ, edges=None):
    """Unit-median NSR shape per Teff node on the given bins.

    With bin edges (passed, or found by matching the centres to a configuration in ariel_bins.json)
    each bin combines the noise-curve bins it covers by inverse variance, so a wide Tier-1 point is
    as much quieter than a Tier-3 bin as its width implies. Before 2026-09-12 the shape was
    interpolated at the bin centres regardless of width, which made the Tier-1 spectrometer points
    3-4x too noisy relative to the photometers. Centres that match no configuration fall back to
    point interpolation (appropriate only for bins no wider than the noise-curve bins)."""
    centres = np.asarray(centres, float)
    if edges is None:
        edges = _edges_for(centres)
    key = (npz, tuple(np.round(centres, 6)), None if edges is None else tuple(np.round(edges, 6)))
    if key in _cache:
        return _cache[key]
    if edges is not None:
        teffs, absn = nsr_binned(edges, npz)
        shapes = {int(T): absn[int(T)] / np.median(absn[int(T)]) for T in teffs}
    else:
        z = np.load(npz); teffs = z["teffs"]; shapes = {}
        for T in teffs:
            T = int(T); wl, nsr = z[f"wl_{T}"], z[f"nsr_{T}"]; o = np.argsort(wl)
            s = np.interp(centres, wl[o], nsr[o]); shapes[T] = s / np.median(s)
    _cache[key] = (teffs, shapes)
    return teffs, shapes


def sigma_matrix(X, tstar, centres, snr=15.0, shape="ariel", level_ppm=None):
    """Per-bin noise sigma. shape: 'white' (flat, peak-to-peak/snr), 'exosim' (ExoSim2-shaped, v3), 'ariel'
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
    teffs, shapes = nsr_shapes(np.asarray(centres, dtype=float), EXOSIM_NPZ if shape == "exosim" else NSR_NPZ)
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
