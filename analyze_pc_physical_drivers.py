"""
What each principal component represents: loading-vector analysis and variance
decomposition by physical parameter.

Two of Reviewer 1's requested analyses (R1-3):
  * loading-vector analysis -- the shape of each PC's loading in wavelength space
  * variance decomposition by physical parameter -- how much of each PC score is
    explained by each generating parameter (radius, temperature, pressures, and
    the log abundances of the four spectrally active molecules).

Same preprocessing as analyze_pc_discriminative_power.py: StandardScaler on the
raw 550-bin training spectra (unsorted columns) -> PCA(102).
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 42
N_COMPONENTS = 102
TRAIN_FILE = "multirex_spectra_H2_train.parquet"

# Generating parameters to decompose the PC scores against.
PARAMS = ["p_radius", "p_mass", "atm temperature", "atm base_pressure",
          "atm top_pressure", "atm H2O", "atm CH4", "atm CO2", "atm O3",
          "atm CO", "atm NH3", "s temperature"]
ACTIVE = ["atm H2O", "atm CH4", "atm CO2", "atm O3"]


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return [c for c in df.columns
            if isinstance(c, float) or (isinstance(c, str) and fp.match(c))]


def univar_r2(y, x):
    """Fraction of var(y) linearly explained by x."""
    x = (x - x.mean()) / (x.std() + 1e-30)
    y = y - y.mean()
    b = (x @ y) / (x @ x)
    yhat = b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum(y ** 2) + 1e-30
    return 1.0 - ss_res / ss_tot


def partial_template(Xs, Pz, idx):
    P = np.column_stack([np.ones(len(Pz))] + [Pz[:, j] for j in range(Pz.shape[1])])
    coef, *_ = np.linalg.lstsq(P, Xs, rcond=None)
    v = coef[1 + idx]
    return v / (np.linalg.norm(v) + 1e-30)


def main():
    os.makedirs("final_results", exist_ok=True)
    df = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df)
    wl = np.array([float(c) for c in cols])
    order_wl = np.argsort(wl)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[cols].values)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(Xs)
    S = pca.transform(Xs)                    # scores n x 102
    comps = pca.components_                  # 102 x 550
    var = pca.explained_variance_ratio_

    # Templates (scaled space) for the active molecules, to name loading shapes.
    Praw = df[ACTIVE].values.astype(float)
    Pz_active = (Praw - Praw.mean(0)) / Praw.std(0)
    templ = {m: partial_template(Xs, Pz_active, i) for i, m in enumerate(ACTIVE)}
    dc = np.ones(len(wl)) / np.sqrt(len(wl))                 # flat / mean-depth
    slope = (wl - wl.mean()); slope = slope / np.linalg.norm(slope)  # linear tilt

    # ---- variance decomposition by parameter (univariate R^2 of each PC score) ----
    Pall = df[PARAMS].values.astype(float)
    r2 = np.zeros((N_COMPONENTS, len(PARAMS)))
    for k in range(N_COMPONENTS):
        for j in range(len(PARAMS)):
            r2[k, j] = univar_r2(S[:, k], Pall[:, j])

    # ---- report ----
    L = ["Principal components: loading shape and physical driver",
         "",
         "Preprocessing: StandardScaler -> PCA(102) on the training spectra.",
         "Loading shape: correlation (|r|) of each PC's loading vector with a flat",
         "mean-depth vector, a linear wavelength tilt, and the partial spectral",
         "imprint of each active molecule. Driver: the single generating parameter",
         "with the highest univariate R^2 against the PC score (that R^2 in %).",
         "",
         f"{'PC':>4} {'var%':>8}  {'loading best-match (|r|)':<28} {'top driver (R^2%)':<22} {'2nd driver (R^2%)'}",
         "-" * 96]
    shape_names = {"flat/mean-depth": dc, "wavelength-tilt": slope,
                   "H2O": templ["atm H2O"], "CH4": templ["atm CH4"],
                   "CO2": templ["atm CO2"], "O3": templ["atm O3"]}
    rows = []
    for k in range(N_COMPONENTS):
        matches = {name: abs(float(comps[k] @ v)) for name, v in shape_names.items()}
        best = max(matches, key=matches.get)
        j_order = np.argsort(r2[k])[::-1]
        d1, d2 = j_order[0], j_order[1]
        rows.append((k, var[k], best, matches[best], PARAMS[d1], r2[k, d1],
                     PARAMS[d2], r2[k, d2]))
        if k <= 9 or k in (9,):
            L.append(f"{k:>4} {100*var[k]:>7.3f}  {best+' ('+format(matches[best],'.2f')+')':<28} "
                     f"{PARAMS[d1]+' ('+format(100*r2[k,d1],'.0f')+')':<22} "
                     f"{PARAMS[d2]+' ('+format(100*r2[k,d2],'.0f')+')'}")
    # also list the most discriminative components' drivers (PC3, PC4, PC9)
    L += ["", "Selected low-variance components of interest:"]
    for k in (2, 3, 4, 5, 9):
        j_order = np.argsort(r2[k])[::-1]
        d1 = j_order[0]
        L.append(f"  PC{k:<3d} var {100*var[k]:.3f}%  driver {PARAMS[d1]} (R^2 {100*r2[k,d1]:.0f}%)  "
                 f"loading ~ {max(shape_names, key=lambda n: abs(comps[k]@shape_names[n]))}")

    # How concentrated is each active molecule's *driver* footprint across PCs?
    L += ["", "Where each active molecule drives the variance (sum of R^2 over PCs,",
          "and how many PCs hold 90% of that):"]
    for m in ACTIVE:
        j = PARAMS.index(m)
        col = r2[:, j]
        share = col / (col.sum() + 1e-30)
        o = np.argsort(share)[::-1]
        c90 = int(np.searchsorted(np.cumsum(share[o]), 0.90) + 1)
        top = ", ".join(f"PC{int(i)}({100*share[i]:.0f}%)" for i in o[:4])
        L.append(f"  {m:<9s}: 90% of its driver footprint in {c90} PCs; top: {top}")

    txt = "\n".join(L)
    print(txt)
    with open("final_results/H2_pc_drivers.txt", "w") as f:
        f.write(txt + "\n")

    pd.DataFrame(rows, columns=["component", "explained_variance_ratio",
                "loading_match", "loading_match_r", "driver1", "driver1_r2",
                "driver2", "driver2_r2"]).to_csv(
                "final_results/H2_pc_drivers.csv", index=False)

    # ---- figure: loading vectors of the first 6 PCs vs wavelength ----
    fig, axes = plt.subplots(6, 1, figsize=(7.2, 8.4), sharex=True, dpi=300)
    for k, ax in enumerate(axes):
        ax.plot(wl[order_wl], comps[k][order_wl], color="#2a78d6", linewidth=1.2)
        ax.axhline(0, color="#b0b0ad", linewidth=0.6)
        ax.set_ylabel(f"PC{k}\n({100*var[k]:.2f}%)", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("Wavelength (µm)", fontsize=10)
    fig.suptitle("PCA loading vectors (first six components)", fontsize=11)
    fig.tight_layout()
    fig.savefig("final_results/pc_loadings.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote final_results/H2_pc_drivers.{txt,csv} and pc_loadings.png")


if __name__ == "__main__":
    main()
