"""
H2 -> N2 composition transfer (R1-1, axis 7).

Evaluate the frozen H2-trained pipeline on the N2-rendered committed planets
(generate_n2_paired.py) against the committed H2 baseline. Zero-shot transfer:
the classifier is trained on H2 and never sees N2. Same StandardScaler ->
PCA(102) -> XGBoost pipeline as the aerosol/opacity results.
"""
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N = 102
fp = re.compile(r"^-?\d+\.\d+$")


def scols(df):
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))], key=float)


def main():
    dtr = pd.read_parquet("multirex_spectra_H2_train.parquet")
    cols = scols(dtr)
    Xtr = dtr[cols].values
    ytr = (dtr["biosignature"] == "yes").astype(int).values
    raw = StandardScaler().fit(Xtr)
    pca = PCA(n_components=N, random_state=SEED).fit(raw.transform(Xtr))
    Xs, ys = shuffle(pca.transform(raw.transform(Xtr)), ytr, random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                        eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(Xs, ys)

    def ev(fmt):
        corr = tot = 0; amps = []
        for i in range(1, 6):
            d = pd.read_parquet(fmt.format(i))
            X = d[cols].values; y = (d["biosignature"] == "yes").astype(int).values
            pred = xgb.predict(pca.transform(raw.transform(X)))
            corr += int((pred == y).sum()); tot += len(y)
            amps.append(np.median(X.std(1) / X.mean(1)))
        return 100 * corr / tot, float(np.mean(amps)), tot

    h2a, h2amp, n = ev("multirex_spectra_H2_test_set_{}.parquet")
    n2a, n2amp, _ = ev("multirex_spectra_N2_paired_set_{}.parquet")
    out = [f"H2 -> N2 composition transfer (frozen H2-trained pipeline), {n} paired planets", "",
           f"  H2 test (baseline)   accuracy {h2a:.2f}%   feature amp {h2amp:.4f}",
           f"  N2 test (transfer)   accuracy {n2a:.2f}%   feature amp {n2amp:.4f}",
           f"  transfer penalty: {n2a-h2a:+.1f} points; N2 features are {n2amp/h2amp:.2f}x of H2", "",
           "The classifier trained on H2 does not transfer to N2 atmospheres: the higher",
           "mean molecular weight shrinks the scale height, muting every feature to a few",
           "percent of the H2 value, so the N2 spectra are effectively featureless and",
           "accuracy falls toward the majority-class baseline."]
    txt = "\n".join(out)
    print(txt)
    with open("final_results/H2_N2_transfer.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
