"""Axis 6 (R1-1) -- instrumental systematics from a REAL radiometric model.

Rigorous replacement for the parametric white/correlated noise: the
wavelength-dependent noise-to-signal ratio (NSR) is computed by ExoRad2
(Mugnai et al. 2020), the open radiometric engine underneath ArielRad, driven
by an Ariel payload reconstructed from PUBLISHED parameters (Tinetti et al.
2018 Ariel Definition Study Report; Mugnai et al. 2020; Edwards et al. 2019).
See scratchpad/exorad_run/ariel_payload.xml for every adopted value.

Caveats stated plainly:
  * The payload is reconstructed from the literature, NOT the consortium's
    official ArielRad instrument file (that is not public).
  * ExoRad computes flux from distance/radius/Teff (it ignores K-magnitude), so
    each Teff SED is evaluated at a common operating point (R=1 Rsun, 20 pc,
    zodiacal + telescope thermal, no atmosphere -- a space telescope at L2).
    The absolute level is therefore swept; the ExoRad product used here is the
    per-wavelength NSR *shape* (its coloring), which is what distinguishes
    realistic instrument noise from white noise.
  * Planck SED per target Teff (standard for a radiometric noise budget).

The NSR shape is interpolated onto the 550-bin observation grid per star (by
nearest Teff), normalised to unit median, and used as the relative noise
sigma(l). Colored (ExoRad) noise is compared to white noise AT THE SAME MEDIAN
sigma, so the only difference is the wavelength coloring. Frozen H2 pipeline
(StandardScaler -> PCA(102) -> XGBoost), baseline 88.9%.

Usage: python evaluate_ariel_noise.py
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
NSR_NPZ = "final_results/ariel_nsr_curves.npz"   # produced by ariel_noise_model/build_ariel_nsr.py
TRAIN = "multirex_spectra_H2_train.parquet"
TESTS = [f"multirex_spectra_H2_test_set_{i}.parquet" for i in range(1, 6)]
OUT = "final_results/H2_ariel_noise.txt"
DRAWS = 5
LEVELS = [(5e-5, "SNR~30"), (1e-4, "SNR~15"), (2e-4, "SNR~7")]

fp = re.compile(r"^-?\d+\.\d+$")


def scols(df):
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def main():
    rng = np.random.default_rng(SEED)

    dtr = pd.read_parquet(TRAIN)
    sc = scols(dtr)
    wl_um = np.array([float(c) for c in sc])

    # ---- ExoRad NSR shapes per Teff node, interpolated onto the 550 grid ----
    z = np.load(NSR_NPZ)
    teff_nodes = z["teffs"]
    shape_by_node = {}
    for T in teff_nodes:
        wl_c = z[f"wl_{T}"]
        nsr_c = z[f"nsr_{T}"]
        o = np.argsort(wl_c)
        shp = np.interp(wl_um, wl_c[o], nsr_c[o])   # linear, flat-extrapolated
        shape_by_node[int(T)] = shp / np.median(shp)  # unit median -> pure color
    tn = np.array(sorted(shape_by_node))

    def star_shapes(Tstar):
        idx = np.abs(Tstar[:, None] - tn[None, :]).argmin(1)
        return np.array([shape_by_node[int(tn[j])] for j in idx])   # (n, 550)

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
        tests.append((X, y, T, star_shapes(T)))

    def predict(X):
        return xgb.predict(pca.transform(raw.transform(X)))

    tot = sum(len(y) for (X, y, T, s) in tests)
    base = 100 * sum(int((predict(X) == y).sum()) for (X, y, T, s) in tests) / tot
    assert 87.5 < base < 90.0, f"baseline {base:.2f} off -- pipeline changed"

    def acc(level, colored):
        accs = []
        for _ in range(DRAWS):
            corr = 0
            for (X, y, T, shp) in tests:
                sig = level * shp if colored else level * np.ones_like(X)
                Xn = X + rng.standard_normal(X.shape) * sig
                corr += int((predict(Xn) == y).sum())
            accs.append(100 * corr / tot)
        return float(np.mean(accs)), float(np.std(accs))

    L = ["Instrumental systematics from ExoRad2 with a published-parameter Ariel payload",
         "",
         f"Frozen H2 pipeline, pooled over {tot} test planets. Clean baseline: {base:.2f}%.",
         "ExoRad2 (Mugnai+2020) NSR shape; Ariel payload reconstructed from Tinetti+2018 / "
         "Mugnai+2020 (not the official ArielRad file). Colored vs white AT MATCHED median sigma.",
         "",
         f"  {'median sigma':<16}{'white acc':>12}{'Ariel-colored':>15}{'colored-white':>15}"]
    for level, lab in LEVELS:
        aw, sw = acc(level, colored=False)
        ac, scol = acc(level, colored=True)
        L.append(f"  {lab+' '+format(level,'.0e'):<16}"
                 f"{aw:>10.2f}%{'':>1}{ac:>13.2f}%{ac-aw:>+14.2f}")
    # report the NSR coloring actually applied
    span = np.median([shape_by_node[int(T)].max() / shape_by_node[int(T)].min()
                      for T in tn])
    L += ["",
          f"Median NSR color span across 0.5-7.8 um (max/min): {span:.1f}x.",
          "Noise is concentrated in the AIRS bands (1.95-7.8 um) that carry CH4/CO2/H2O/O3."]

    txt = "\n".join(L)
    print(txt)
    os.makedirs("final_results", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(txt + "\n")
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
