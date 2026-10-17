"""Readers for published JWST transmission-spectrum products, each returning a segment dict
{wl (um), depth (fractional), err (fractional), width (um or None)} for jwst_to_ariel.combine.

Every archive has its own layout; a reader is added per archive as it is inspected (the file
formats are recorded in REVISION_PLAN.md's data section as they are met). Asymmetric errors are
symmetrized as the mean of the two sides. Depths are converted to fractional (Rp/Rs)^2 from ppm or
percent where the product uses those units.
"""
import numpy as np


def _sym(lo, hi):
    return 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))


def ers_h5(path):
    """ERS-style HDF5 (NIRCam record 7101283 and siblings): central_wavelength, bin_half_width,
    transit_depth, transit_depth_error(_up/_down), all fractional and in microns."""
    import h5py
    with h5py.File(path) as h:
        wl = np.asarray(h["central_wavelength"][()], float)
        d = np.asarray(h["transit_depth"][()], float)
        if "transit_depth_error" in h:
            e = np.asarray(h["transit_depth_error"][()], float)
        else:
            e = _sym(h["transit_depth_error_down"][()], h["transit_depth_error_up"][()])
        hw = np.asarray(h["bin_half_width"][()], float)
    n = len(wl)
    hw = hw[:n] if len(hw) >= n else None            # one product carries an off-by-one half-width array
    return dict(wl=wl, depth=d, err=e, width=None if hw is None else 2 * hw)


def table(path, wl_col=0, depth_col=1, err_col=2, err2_col=None, width_col=None, depth_unit=1.0, skip=0, delimiter=None):
    """A plain text/CSV table. depth_unit: 1 for fractional, 1e-6 for ppm, 1e-2 for percent."""
    a = np.genfromtxt(path, skip_header=skip, delimiter=delimiter, comments="#")
    a = a[np.isfinite(a[:, wl_col])]
    e = a[:, err_col] if err2_col is None else _sym(a[:, err_col], a[:, err2_col])
    return dict(wl=a[:, wl_col], depth=a[:, depth_col] * depth_unit, err=e * depth_unit,
                width=None if width_col is None else a[:, width_col])


def ers_niriss_h5(path, order, drop_flagged=True):
    """NIRISS SOSS products of record 7709712: wavelength_order_N (um), wavelength_err_order_N (half-width,
    um), transit_depth_order_N and transit_depth_err_order_N (ppm), quality_flags_order_N (nonzero = bad)."""
    import h5py
    with h5py.File(path) as h:
        g = lambda k: np.asarray(h[f"{k}_order_{order}"][()], float)
        wl, hw, d, e = g("wavelength"), g("wavelength_err"), g("transit_depth"), g("transit_depth_err")
        q = np.asarray(h[f"quality_flags_order_{order}"][()]) if f"quality_flags_order_{order}" in h else np.zeros(len(wl))
    keep = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if drop_flagged:
        keep &= (q == 0)
    return dict(wl=wl[keep], depth=d[keep] * 1e-6, err=e[keep] * 1e-6, width=2 * hw[keep])


def ers_nc(path):
    """NIRSpec G395H products of record 7185300 (HDF5-backed netCDF): central_wavelength, bin_half_width,
    transit_depth, transit_depth_error or _error_neg/_error_pos, all fractional and in microns."""
    import h5py
    with h5py.File(path) as h:
        wl = np.asarray(h["central_wavelength"][()], float); d = np.asarray(h["transit_depth"][()], float)
        e = np.asarray(h["transit_depth_error"][()], float) if "transit_depth_error" in h else _sym(h["transit_depth_error_neg"][()], h["transit_depth_error_pos"][()])
        hw = np.asarray(h["bin_half_width"][()], float)
    ok = np.isfinite(d) & np.isfinite(e) & (e > 0)
    width = 2 * hw if np.all(hw < 0.1) else None     # one reduction stored wavelengths in the half-width field
    return dict(wl=wl[ok], depth=d[ok], err=e[ok], width=None if width is None else width[ok])


def prism_h5(path, drop=(0.634, 2.115)):
    """NIRSpec PRISM products of record 7388032 (Rustamkulov et al. 2023): wavelength (um), transit_depth
    and transit_depth_uncertainty (fractional). `drop` removes the saturated region, by default the
    range Carter & May et al. 2024 flag (satflag) in their re-reduction of the same data; None keeps it."""
    import h5py
    with h5py.File(path) as h:
        wl = np.asarray(h["wavelength"][()], float); d = np.asarray(h["transit_depth"][()], float)
        e = np.asarray(h["transit_depth_uncertainty"][()], float)
    keep = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if drop is not None:
        keep &= ~((wl >= drop[0]) & (wl <= drop[1]))
    o = np.argsort(wl[keep])
    return dict(wl=wl[keep][o], depth=d[keep][o], err=e[keep][o], width=None)


def miri_h5(path):
    """MIRI LRS products of record 10055845 (Powell et al. 2024): wavelength (um), dppm, dppm_error (ppm),
    wavelength_error (half-width, scalar or per point). Points are returned sorted by wavelength."""
    import h5py
    with h5py.File(path) as h:
        wl = np.asarray(h["wavelength"][()], float); d = np.asarray(h["dppm"][()], float); e = np.asarray(h["dppm_error"][()], float)
        hw = np.broadcast_to(np.asarray(h["wavelength_error"][()], float), wl.shape)
    o = np.argsort(wl); keep = np.isfinite(d[o]) & np.isfinite(e[o]) & (e[o] > 0)
    return dict(wl=wl[o][keep], depth=d[o][keep] * 1e-6, err=e[o][keep] * 1e-6, width=2 * hw[o][keep])


def carter_csv(path, drop_saturated=True):
    """Transmission spectra of record 10161743 (Carter & May et al. 2024 benchmark): CSV with wave,
    wave_low, wave_hig (um), rp/rs with asymmetric errors, and for PRISM a satflag column (nonzero = in
    the saturated region) and a dilution factor already applied. Depth = (rp/rs)^2, error propagated."""
    import pandas as pd
    t = pd.read_csv(path)
    rp = t["rp/rs"].to_numpy(float); e = _sym(t["rp/rs_err_low"], t["rp/rs_err_hih"])
    keep = np.isfinite(rp) & np.isfinite(e) & (e > 0)
    if drop_saturated and "satflag" in t:
        keep &= t["satflag"].to_numpy() == 0
    wl = t["wave"].to_numpy(float); width = (t["wave_hig"] - t["wave_low"]).to_numpy(float)
    o = np.argsort(wl[keep])
    return dict(wl=wl[keep][o], depth=(rp ** 2)[keep][o], err=(2 * rp * e)[keep][o], width=width[keep][o])


def carter_model(path):
    """The benchmark's best-fit model (5_MODEL_SPECTRUM/BestFitModel.txt): wavelength (um), depth (fractional),
    0.5-12.5 um. Returns (wl, depth) sorted."""
    a = np.loadtxt(path); o = np.argsort(a[:, 0])
    return a[o, 0], a[o, 1]
