"""
How CH4 and O3 absorption features project onto the principal components.

Reviewer 1 (R1-5 / R1-3) asked for "explicit quantification of how methane and
ozone absorption features project onto individual principal components." This is
the direct test of whether chemical information is isolated in a few components
or distributed across many.

Method
------
1. Preprocess exactly as analyze_pc_discriminative_power.py: StandardScaler on the
   raw 550-bin spectra (training set, unsorted columns) -> PCA(102). The PCA
   loading vectors pca.components_ are an orthonormal basis of the scaled spectrum
   space.
2. Isolate the *partial* spectral response to each molecule: regress the scaled
   spectrum on the log abundances of the four spectrally active molecules
   (H2O, CH4, CO2, O3) plus the main structural parameters (planet radius,
   temperature, base/top pressure). The CH4 (O3) regression-coefficient vector is
   the spectral imprint of CH4 (O3) holding the others fixed -- i.e. "the methane
   (ozone) absorption features." Normalise to a unit vector.
3. Project that unit response vector onto every principal component. Because the
   components are orthonormal, the squared projection coefficients say what
   fraction of the CH4 (O3) imprint each component captures.

Reported: how concentrated that projection is (participation ratio, number of
components to reach 50%/90%, share in the two highest-variance components), which
is the quantity that distinguishes "isolated in a few components" from
"distributed across many".
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

# Spectrally active molecules (only these have opacity data) + structural params
# that shape the continuum, used as controls so the CH4/O3 vectors are partial.
ACTIVE = ["atm H2O", "atm CH4", "atm CO2", "atm O3"]
STRUCT = ["p_radius", "atm temperature", "atm base_pressure", "atm top_pressure"]


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return [c for c in df.columns
            if isinstance(c, float) or (isinstance(c, str) and fp.match(c))]


def partial_response(Xs, predictors, target_idx):
    """OLS of every scaled bin on all predictors; return the target's coefficient
    vector across bins (the partial spectral imprint of that predictor)."""
    P = np.column_stack([np.ones(len(predictors))] + [predictors[:, j]
                        for j in range(predictors.shape[1])])
    coef, *_ = np.linalg.lstsq(P, Xs, rcond=None)   # (1+p) x n_bins
    return coef[1 + target_idx]                     # +1 skips the intercept row


def concentration(proj_sq):
    """proj_sq: squared projection onto each PC (unnormalised)."""
    q = proj_sq / proj_sq.sum()                     # share per PC
    pr = 1.0 / np.sum(q ** 2)                        # participation ratio
    order = np.argsort(q)[::-1]
    csum = np.cumsum(q[order])
    n50 = int(np.searchsorted(csum, 0.50) + 1)
    n90 = int(np.searchsorted(csum, 0.90) + 1)
    return q, pr, n50, n90, order


def main():
    os.makedirs("final_results", exist_ok=True)
    df = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df)
    wl = np.array([float(c) for c in cols])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[cols].values)      # scaled spectra
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(Xs)
    comps = pca.components_                          # 102 x 550, orthonormal
    var = pca.explained_variance_ratio_

    # Standardise predictors so coefficients are comparable.
    pred_names = ACTIVE + STRUCT
    Praw = df[pred_names].values.astype(float)
    Pz = (Praw - Praw.mean(0)) / Praw.std(0)

    out = {}
    for mol, idx in [("CH4", ACTIVE.index("atm CH4")),
                     ("O3", ACTIVE.index("atm O3"))]:
        resp = partial_response(Xs, Pz, idx)        # imprint across bins (scaled space)
        resp_unit = resp / np.linalg.norm(resp)
        proj = comps @ resp_unit                    # projection onto each PC
        captured = float(np.sum(proj ** 2))         # fraction of imprint inside the 102-dim basis
        q, pr, n50, n90, order = concentration(proj ** 2)
        out[mol] = dict(q=q, pr=pr, n50=n50, n90=n90, order=order,
                        captured=captured, proj=proj)

    # Report
    L = ["How CH4 and O3 absorption features project onto the 102 principal components",
         "",
         "Method: partial spectral response to each molecule (scaled space, controlling",
         "for the other active molecules and planet radius/temperature/pressure),",
         "projected onto the orthonormal PCA loading vectors. A response isolated in one",
         "component would give participation ratio ~1 and 100% share in that component;",
         "a response spread evenly over the tail gives a large participation ratio.",
         "",
         f"PC0 + PC1 carry {100*var[:2].sum():.2f}% of the spectral variance.",
         ""]
    for mol in ("CH4", "O3"):
        d = out[mol]
        q, order = d["q"], d["order"]
        L += [f"--- {mol} ---",
              f"  fraction of the {mol} imprint captured by the 102-component basis: {100*d['captured']:.1f}%",
              f"  participation ratio (effective # of components the imprint spreads over): {d['pr']:.1f}",
              f"  components needed to capture 50% of the imprint: {d['n50']}",
              f"  components needed to capture 90% of the imprint: {d['n90']}",
              f"  share in the two highest-variance components (PC0, PC1): {100*(q[0]+q[1]):.1f}%",
              f"  share in the low-variance tail (PCs 2-101):             {100*q[2:].sum():.1f}%",
              "  top 8 components by share of the imprint (PC: share%, variance-rank%):",
              ]
        for k in order[:8]:
            L.append(f"     PC{k:<3d}  {100*q[k]:5.1f}%   (variance {100*var[k]:.3f}%)")
        L.append("")
    verdict = [
        "Reading:",
        "  Each imprint concentrates in a few components (participation ratio ~2), not",
        "  spread evenly across the tail. But the original manuscript attribution -- PC0/PC1",
        "  purely physical, chemistry isolated in PCs 2-101 -- is refuted from both sides:",
        "  the O3 imprint sits mostly in PC0 (the highest-variance 'physical' component), and",
        "  the CH4 imprint sits mostly in PC2/PC3, not spread through all of 2-101.",
        "  Concentration of the imprint is NOT discriminative power: the components carrying",
        "  it (PC0 for O3 especially) are entangled with the dominant continuum/radius",
        "  variance, which is why no single component separates the threshold label",
        "  (max single-feature AUC 0.66; PC0 AUC 0.51) -- consistent with the per-component",
        "  AUC and ablation results.",
    ]
    L += verdict + [""]
    txt = "\n".join(L)
    print(txt)
    with open("final_results/H2_chem_projection.txt", "w") as f:
        f.write(txt + "\n")

    # CSV
    pd.DataFrame({
        "component": np.arange(N_COMPONENTS),
        "explained_variance_ratio": var,
        "ch4_share": out["CH4"]["q"],
        "o3_share": out["O3"]["q"],
    }).to_csv("final_results/H2_chem_projection.csv", index=False)

    # Figure: share of each molecule's imprint per component, with PC0/PC1 shaded.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, dpi=300)
    for ax, mol, color in zip(axes, ("CH4", "O3"), ("#2a78d6", "#008300")):
        q = out[mol]["q"]
        ax.axvspan(-0.5, 1.5, color="#52514e", alpha=0.12, linewidth=0)
        ax.bar(np.arange(N_COMPONENTS), 100 * q, color=color, width=0.9)
        ax.set_ylabel(f"{mol} imprint\nshare per PC (%)", fontsize=10)
        ax.set_xlim(-1, N_COMPONENTS)
        ax.text(0.99, 0.9, f"participation ratio {out[mol]['pr']:.0f} components; "
                f"90% needs {out[mol]['n90']} PCs",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_xlabel("Principal component index", fontsize=10)
    fig.tight_layout()
    fig.savefig("final_results/chem_projection.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote final_results/H2_chem_projection.{txt,csv} and chem_projection.png")


if __name__ == "__main__":
    main()
