"""Stellar contamination (transit light source effect) on the native grid.

Rackham et al. (2018): unocculted spots and faculae make the transit chord's
stellar spectrum unrepresentative of the disk-integrated one, multiplying the
transit depth by

    eps(l) = 1 / [1 - f_spot (1 - F_spot(l)/F_phot(l)) - f_fac (1 - F_fac(l)/F_phot(l))]

with F the stellar surface fluxes from BT-Settl/PHOENIX (solar metallicity,
STScI phoenixm00 atlas in v2/phoenix/), binned onto the native grid. Adopted
contrasts as in the old study: T_spot = 0.85 T_eff, T_fac = T_eff + 100 K;
log g from the stellar mass and radius, clamped to the grid ceiling.

Applied at native resolution (before binning and noise), so every observing
configuration sees the same physical contamination. Writes
    v2/data/{split}_native_tlse_{case}.npy
for cases spots02, spots05, spots10, spots20, fac05, fac10, mixed (10% spots + 5% faculae).

Usage: python shift_tlse.py [--splits ...] [--limit N]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PHX = os.path.join(HERE, "phoenix")
T_SPOT_FRAC, T_FAC_DELTA = 0.85, 100.0
CASES = {"spots02": (0.02, 0.0), "spots05": (0.05, 0.0), "spots10": (0.10, 0.0),
         "spots20": (0.20, 0.0), "fac05": (0.0, 0.05), "fac10": (0.0, 0.10),
         "mixed": (0.10, 0.05)}
G_SUN = 274.0  # m/s^2

GRID_TEFF = np.array(sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                            for p in glob.glob(f"{PHX}/phoenixm00_*.fits")))
_edges = None
_cache = {}


def teff_node(t):
    return int(GRID_TEFF[np.argmin(np.abs(GRID_TEFF - t))])


def logg_col(logg):
    g = int(round(logg / 0.5)) * 5
    return f"g{min(50, max(0, g)):02d}"


def set_grid(wl_um):
    global _edges
    c = np.asarray(wl_um, float) * 1e4
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0]); e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    _edges = e


def phoenix_binned(tnode, gcol):
    key = (tnode, gcol)
    if key in _cache:
        return _cache[key]
    with fits.open(f"{PHX}/phoenixm00_{tnode}.fits") as f:
        w = np.asarray(f[1].data["WAVELENGTH"], float)
        cols = f[1].columns.names
        if gcol not in cols:      # some nodes lack low-gravity columns
            avail = sorted(c for c in cols if c.startswith("g"))
            gcol = min(avail, key=lambda c: abs(int(c[1:]) - int(gcol[1:])))
        flux = np.asarray(f[1].data[gcol], float)
    idx = np.searchsorted(_edges, w) - 1
    m = (idx >= 0) & (idx < _edges.size - 1)
    nb = _edges.size - 1
    s = np.bincount(idx[m], weights=flux[m], minlength=nb)
    n = np.bincount(idx[m], minlength=nb)
    b = s / np.maximum(n, 1)
    if (n == 0).any():            # fill bins with no PHOENIX sample
        good = n > 0
        b = np.interp(np.arange(nb), np.arange(nb)[good], b[good])
    _cache[key] = b
    return b


def contamination(Tstar, logg, f_spot, f_fac):
    n = Tstar.size
    eps = np.ones((n, _edges.size - 1))
    for i in range(n):
        g = logg_col(logg[i])
        Fp = phoenix_binned(teff_node(Tstar[i]), g)
        term = np.zeros_like(Fp)
        if f_spot > 0:
            term += f_spot * (1.0 - phoenix_binned(teff_node(T_SPOT_FRAC * Tstar[i]), g) / Fp)
        if f_fac > 0:
            term += f_fac * (1.0 - phoenix_binned(teff_node(Tstar[i] + T_FAC_DELTA), g) / Fp)
        eps[i] = 1.0 / (1.0 - term)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=[f"test{k}" for k in range(1, 6)])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    set_grid(wl)
    print(f"PHOENIX nodes: {GRID_TEFF.min()}-{GRID_TEFF.max()} K ({len(GRID_TEFF)} files)")
    for split in a.splits:
        P = pd.read_parquet(os.path.join(DATA, f"{split}_params.parquet"))
        X = np.load(os.path.join(DATA, f"{split}_native.npy")).astype(np.float64)
        if a.limit:
            P, X = P.head(a.limit), X[:a.limit]
        T = P["s temperature"].to_numpy()
        logg = np.log10(G_SUN * P["s mass"].to_numpy() / P["s radius"].to_numpy() ** 2 * 100)  # cgs
        for case, (fs, ff) in CASES.items():
            eps = contamination(T, logg, fs, ff)
            Xc = (X * eps).astype(np.float32)
            np.save(os.path.join(DATA, f"{split}_native_tlse_{case}.npy"), Xc)
            amp = np.median(Xc.std(1) / X.std(1))
            print(f"{split} {case}: median amplitude ratio {amp:.3f}, median eps {np.median(eps):.4f}", flush=True)


if __name__ == "__main__":
    main()
