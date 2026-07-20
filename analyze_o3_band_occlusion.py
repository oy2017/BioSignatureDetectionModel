"""
Which wavelength band carries the O3 signal the classifier uses?

The haze result showed the 0.5-1 um band being crushed to 0.05x amplitude,
and the natural reading is that this destroys the O3 Chappuis-band
information - but whether the classifier actually relies on the Chappuis
band (0.6 um) versus the 4.74 um O3 feature was never verified (the
strongest O3 band, 9.6 um, lies outside the 0.5-7.8 um window entirely).

Occlusion test on the frozen pipeline: mask one octave band at a time in
the clear test spectra (replace the band's bins with a linear bridge over
log-wavelength through the surrounding spectrum; at constant resolving
power each octave holds the same number of bins, so the masks are
comparable). Then measure accuracy on the label-diagnostic subsets:

  * O3 subset:  rows with CH4 >= -6, where the label depends only on O3
  * CH4 subset: rows with O3 >= -7, where the label depends only on CH4

If masking 0.5-1 um collapses the O3 subset toward chance while the other
masks do not, the short-wavelength region is where the classifier reads
O3, and the haze-mechanism wording can say so directly.

Caveat: masked spectra are out of distribution for the frozen scaler and
PCA, like any perturbation test; comparing equal-width masks controls for
generic out-of-distribution damage.

Usage:
    python analyze_o3_band_occlusion.py
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
CLEAN_TEST_FMT = "multirex_spectra_H2_test_set_{}.parquet"
BIO_CH4, BIO_O3 = -6.0, -7.0
BANDS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (3.9, 7.8)]


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def mask_band(X, wl, lo, hi):
    """Replace the band's bins with a linear bridge (in log-wavelength)
    through the rest of the spectrum."""
    band = (wl >= lo) & (wl < hi)
    keep = ~band
    out = X.copy()
    logwl = np.log(wl)
    for i in range(len(X)):
        out[i, band] = np.interp(logwl[band], logwl[keep], X[i, keep])
    return out, band.sum()


def main():
    df_tr = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df_tr)
    wl = np.array([float(c) for c in cols])
    X_tr = df_tr[cols].values
    y_tr = (df_tr["biosignature"] == "yes").astype(int).values

    scaler_raw = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(
        scaler_raw.transform(X_tr))
    Xs, ys = shuffle(pca.transform(scaler_raw.transform(X_tr)), y_tr,
                     random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                        subsample=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(Xs, ys)

    tests = pd.concat([pd.read_parquet(CLEAN_TEST_FMT.format(i))
                       for i in range(1, 6)], ignore_index=True)
    X_te = tests[spectral_cols(tests)].values
    y_te = (tests["biosignature"] == "yes").astype(int).values
    ch4 = tests["atm CH4"].astype(float).values
    o3 = tests["atm O3"].astype(float).values
    o3_subset = ch4 >= BIO_CH4    # label decided by O3 alone
    ch4_subset = o3 >= BIO_O3     # label decided by CH4 alone

    def evaluate(X):
        p = xgb.predict_proba(pca.transform(scaler_raw.transform(X)))[:, 1]
        pred = (p > 0.5).astype(int)
        return (np.mean(pred == y_te),
                np.mean(pred[o3_subset] == y_te[o3_subset]),
                np.mean(pred[ch4_subset] == y_te[ch4_subset]))

    lines = ["Octave-band occlusion on the frozen pipeline (XGBoost, clear "
             "test sets)", "",
             f"O3 subset: n={o3_subset.sum()} (CH4 >= {BIO_CH4}, label = O3 "
             f">= {BIO_O3});  CH4 subset: n={ch4_subset.sum()}", "",
             f"{'condition':<18} {'bins':>5} {'overall':>9} {'O3 subset':>10} "
             f"{'CH4 subset':>11}", "-" * 58]
    a, ao, ac = evaluate(X_te)
    lines.append(f"{'unmasked':<18} {'-':>5} {a:>9.2%} {ao:>10.2%} {ac:>11.2%}")
    rows = [("unmasked", 0, a, ao, ac)]
    for lo, hi in BANDS:
        Xm, nbins = mask_band(X_te, wl, lo, hi)
        a, ao, ac = evaluate(Xm)
        name = f"mask {lo:.1f}-{hi:.1f} um"
        lines.append(f"{name:<18} {nbins:>5} {a:>9.2%} {ao:>10.2%} {ac:>11.2%}")
        rows.append((name, nbins, a, ao, ac))

    out = "\n".join(lines)
    print(out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_o3_band_occlusion.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["condition", "bins_masked", "overall_acc",
                                "o3_subset_acc", "ch4_subset_acc"]
                 ).to_csv("final_results/H2_o3_band_occlusion.csv", index=False)
    print("\nWrote final_results/H2_o3_band_occlusion.{txt,csv}")


if __name__ == "__main__":
    main()
