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
