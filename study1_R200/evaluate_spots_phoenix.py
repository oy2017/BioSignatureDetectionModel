"""Axis 4 (R1-1) -- stellar contamination with REAL PHOENIX spectra.

Rigorous replacement for both the manuscript's 1/lambda proxy and the earlier
blackbody TLSE: the Transit Light Source Effect (Rackham et al. 2018) computed
from genuine BT-Settl PHOENIX model spectra (Allard et al. 2012; the STScI grid
at ssb.stsci.edu/trds/grid/phoenix -- BT-Settl, not the Husser+2013 ACES grid),
so the spot/faculae spectra carry their own molecular bands (TiO, water) -- the
contamination a Planck curve cannot produce and the effect most able to mimic a
planetary feature.

Unocculted spots/faculae make the transit chord's stellar spectrum
unrepresentative of the disk-integrated one, multiplying the measured transit
depth by a wavelength-dependent stellar contamination factor

    eps(l) = 1 / [ 1 - f_spot (1 - F_spot(l)/F_phot(l))
                     - f_fac  (1 - F_fac(l)/F_phot(l)) ]

with F the stellar surface fluxes (PHOENIX, binned to the observation grid).
Adopted contrasts (stated as caveats): cool spots T_spot = 0.85 T_eff, hot
faculae T_fac = T_eff + 100 K (Rackham et al. 2018, 2019). PHOENIX solar
metallicity; log g clamped to the grid ceiling (5.0) for the few compact stars.

Frozen H2 pipeline (StandardScaler -> PCA(102) -> XGBoost on the scores), the
same pipeline used for the aerosol/opacity results. Paired: every contaminated
spectrum has a clean spectrum for the identical planet, so the change is the
contamination and nothing else, and a McNemar-style flip breakdown is reported.

Usage: python evaluate_spots_phoenix.py
"""
import os
import re
import glob

import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
PHX_DIR = ("/tmp/claude-1000/-mnt-c-Users-owenh-behaviorbench-eval/"
           "041ec91d-7980-4915-ae40-daa5bc4abb3f/scratchpad/phoenix")
TRAIN = "multirex_spectra_H2_train.parquet"
TESTS = [f"multirex_spectra_H2_test_set_{i}.parquet" for i in range(1, 6)]
OUT = "final_results/H2_spots_phoenix.txt"

T_SPOT_FRAC = 0.85     # cool spots: T_spot = 0.85 T_eff
T_FAC_DELTA = 100.0    # hot faculae: T_fac = T_eff + 100 K

fp = re.compile(r"^-?\d+\.\d+$")


def scols(df):
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


# ---- PHOENIX grid access ------------------------------------------------
GRID_TEFF = np.array(sorted(
    int(os.path.basename(p).split("_")[1].split(".")[0])
    for p in glob.glob(f"{PHX_DIR}/phoenixm00_*.fits")))


def teff_node(t):
    return int(GRID_TEFF[np.argmin(np.abs(GRID_TEFF - t))])


def logg_col(logg):
    g = int(round(logg / 0.5)) * 5     # 0.0->00, 0.5->05, ... 5.0->50
    g = min(50, max(0, g))
    return f"g{g:02d}"


_edges = None
_bincache = {}


def set_grid(wl_um):
    """Bin edges (Angstrom) for the 550-point observation grid."""
    global _edges
    c = np.asarray(wl_um, float) * 1e4          # um -> Angstrom
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    _edges = e


def phoenix_binned(tnode, gcol):
    """PHOENIX surface flux at (tnode, gcol) averaged onto the 550-point grid."""
    key = (tnode, gcol)
    if key in _bincache:
        return _bincache[key]
    with fits.open(f"{PHX_DIR}/phoenixm00_{tnode}.fits") as f:
        w = np.asarray(f[1].data["WAVELENGTH"], float)     # Angstrom
        flux = np.asarray(f[1].data[gcol], float)          # FLAM
    idx = np.searchsorted(_edges, w) - 1
    m = (idx >= 0) & (idx < _edges.size - 1)
    nb = _edges.size - 1
    s = np.bincount(idx[m], weights=flux[m], minlength=nb)
    n = np.bincount(idx[m], minlength=nb)
    binned = s / np.maximum(n, 1)
    _bincache[key] = binned
    return binned


def contamination(Tstar, logg, f_spot, f_fac):
    """eps(l) per planet: (n, 550)."""
    n = Tstar.size
    eps = np.ones((n, _edges.size - 1))
    # cache the ratio spectra per (teff_node, gcol) triple
    for i in range(n):
        g = logg_col(logg[i])
        Fp = phoenix_binned(teff_node(Tstar[i]), g)
        term = np.zeros_like(Fp)
        if f_spot > 0:
            Fs = phoenix_binned(teff_node(T_SPOT_FRAC * Tstar[i]), g)
            term += f_spot * (1.0 - Fs / Fp)
        if f_fac > 0:
            Ff = phoenix_binned(teff_node(Tstar[i] + T_FAC_DELTA), g)
            term += f_fac * (1.0 - Ff / Fp)
        eps[i] = 1.0 / (1.0 - term)
    return eps


def main():
    dtr = pd.read_parquet(TRAIN)
    sc = scols(dtr)
    wl_um = np.array([float(c) for c in sc])
    set_grid(wl_um)

    Xtr = dtr[sc].values
    ytr = (dtr["biosignature"] == "yes").astype(int).values
    raw = StandardScaler().fit(Xtr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(raw.transform(Xtr))
    Xs, ys = shuffle(pca.transform(raw.transform(Xtr)), ytr, random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                        subsample=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(Xs, ys)

    tests = []
    for tf in TESTS:
        d = pd.read_parquet(tf)
        X = d[sc].values
        y = (d["biosignature"] == "yes").astype(int).values
        T = d["s temperature"].values.astype(float)
        logg = 4.438 + np.log10(d["s mass"].values) - 2 * np.log10(d["s radius"].values)
        tests.append((X, y, T, logg))

    def predict(X):
        return xgb.predict(pca.transform(raw.transform(X)))

    # clean baseline (paired reference)
    base_pred = [predict(X) for (X, y, T, g) in tests]
    base_corr = sum(int((p == y).sum()) for p, (X, y, T, g) in zip(base_pred, tests))
    tot = sum(len(y) for (X, y, T, g) in tests)
    base = 100 * base_corr / tot
    assert 87.5 < base < 90.0, f"baseline {base:.2f} off -- pipeline changed"

    L = ["Stellar contamination with real PHOENIX spectra -- Transit Light Source Effect",
         "",
         f"Frozen H2 pipeline, pooled over {tot} test planets. Clean baseline: {base:.2f}%.",
         "Cool spots T_spot=0.85 T_eff; hot faculae T_fac=T_eff+100 K; PHOENIX solar Z.",
         "",
         f"  {'case':<30}{'accuracy':>9}{'change':>9}{'flip c->i':>11}{'feat.amp':>10}"]

    def run(f_spot, f_fac, label):
        corr = 0
        c2i = 0
        amps = []
        for (X, y, T, g), bp in zip(tests, base_pred):
            eps = contamination(T, g, f_spot, f_fac)
            Xc = X * eps
            pred = predict(Xc)
            corr += int((pred == y).sum())
            c2i += int(((bp == y) & (pred != y)).sum())
            amps.append(np.median(Xc.std(1) / np.abs(Xc.mean(1))))
        acc = 100 * corr / tot
        amp = float(np.median(amps))
        L.append(f"  {label:<30}{acc:>8.2f}%{acc-base:>+9.2f}{c2i:>11d}{amp:>10.5f}")
        return acc

    L.append(f"  {'clean (reference)':<30}{base:>8.2f}%{0.0:>+9.2f}{0:>11d}"
             f"{float(np.median([np.median(X.std(1)/np.abs(X.mean(1))) for X,_,_,_ in tests])):>10.5f}")
    for f in (0.02, 0.05, 0.10, 0.20):
        run(f, 0.0, f"cool spots f={int(f*100)}%")
    for f in (0.05, 0.10):
        run(0.0, f, f"faculae f={int(f*100)}%")
    run(0.10, 0.05, "spots 10% + faculae 5%")

    txt = "\n".join(L)
    print(txt)
    os.makedirs("final_results", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(txt + "\n")
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
