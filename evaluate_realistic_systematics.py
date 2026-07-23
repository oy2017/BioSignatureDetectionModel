"""
Two physically-grounded upgrades to R1-1's parametric proxies:

  axis 4 -- stellar contamination: the Transit Light Source Effect (Rackham
    et al. 2018). Unocculted starspots/faculae make the transit chord's stellar
    spectrum unrepresentative of the disk-integrated one, multiplying the
    measured depth by a wavelength-dependent factor
        eps(l) = 1 / [1 - f (1 - B(l,T_spot)/B(l,T_phot))]
    with B the Planck function. This replaces the earlier 1/l proxy with a real
    (blackbody) spot model. NOTE: blackbody, not PHOENIX stellar spectra.

  axis 6 -- instrumental systematics: a photon-noise-limited model with a
    systematic floor, wavelength dependence set by each star's SED
    (sigma(l) ~ 1/sqrt(B(l,T_star) * l) + floor), instead of flat white noise.
    This is an approximation to Ariel's radiometric model, NOT a run of ArielRad.

Both are applied to the committed test spectra and pushed through the frozen
H2-trained pipeline (StandardScaler -> PCA(102) -> XGBoost on the scores), the
same pipeline used for the aerosol/opacity results. Baseline 88.9%.
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
H, C, KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
fp = re.compile(r"^-?\d+\.\d+$")


def scols(df):
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))], key=float)


def planck(wl_m, T):
    # T may be (n,1); wl_m (1,L). Returns spectral radiance (constants keep ratios/shape correct).
    x = H * C / (wl_m * KB * T)
    return 1.0 / (wl_m ** 5 * (np.expm1(x)))


def main():
    rng = np.random.default_rng(SEED)
    dtr = pd.read_parquet("multirex_spectra_H2_train.parquet")
    sc = scols(dtr)
    wl_um = np.array([float(c) for c in sc]); wl_m = (wl_um * 1e-6)[None, :]
    Xtr = dtr[sc].values
    ytr = (dtr["biosignature"] == "yes").astype(int).values
    raw = StandardScaler().fit(Xtr)
    pca = PCA(n_components=N, random_state=SEED).fit(raw.transform(Xtr))
    Xs, ys = shuffle(pca.transform(raw.transform(Xtr)), ytr, random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8,
                        eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(Xs, ys)

    tests = []
    for i in range(1, 6):
        d = pd.read_parquet(f"multirex_spectra_H2_test_set_{i}.parquet")
        tests.append((d[sc].values, (d["biosignature"] == "yes").astype(int).values,
                      d["s temperature"].values[:, None]))

    def accuracy(fn):
        corr = tot = 0
        for X, y, Tstar in tests:
            pred = xgb.predict(pca.transform(raw.transform(fn(X, Tstar))))
            corr += int((pred == y).sum()); tot += len(y)
        return 100 * corr / tot

    def amp(X, Tstar, fn):
        Xp = fn(X, Tstar)
        return float(np.median(Xp.std(1) / Xp.mean(1)))

    base = accuracy(lambda X, T: X)
    L = ["Physically-grounded stellar contamination (TLSE) and Ariel-like noise",
         f"", f"Frozen H2 pipeline, pooled over 2697 test planets. Clean baseline: {base:.2f}%.",
         ""]

    # ---- axis 4: TLSE stellar spots / faculae ----
    L += ["Stellar contamination -- Transit Light Source Effect (blackbody spots):",
          f"  {'case':<26}{'accuracy':>10}{'change':>9}{'feat.amp':>10}"]
    def tlse(X, Tstar, f, tcontrast):
        Tspot = Tstar * tcontrast
        eps = 1.0 / (1.0 - f * (1.0 - planck(wl_m, Tspot) / planck(wl_m, Tstar)))
        return X * eps
    ampc = tests[0][0]  # for amp we use set 1
    for f in (0.02, 0.05, 0.10, 0.20):
        fn = lambda X, T, f=f: tlse(X, T, f, 0.85)         # cool spots, T_spot = 0.85 T_phot
        a = accuracy(fn)
        am = amp(tests[0][0], tests[0][2], fn)
        L.append(f"  spots f={int(f*100)}% (T_sp=0.85T)   {a:>9.2f}%{a-base:>+9.2f}{am:>10.5f}")
    for f in (0.05, 0.10):
        fn = lambda X, T, f=f: tlse(X, T, f, 1.10)         # hot faculae, T_fac = 1.10 T_phot
        a = accuracy(fn)
        L.append(f"  faculae f={int(f*100)}% (T_fa=1.10T){a:>9.2f}%{a-base:>+9.2f}")

    # ---- axis 6: Ariel-like photon noise + floor ----
    L += ["", "Instrumental systematics -- photon-noise (SED-shaped) + floor:",
          f"  {'case':<26}{'accuracy':>10}{'change':>9}"]
    def photon_shape(Tstar):
        # relative per-bin noise ~ 1/sqrt(photon rate); photon rate ~ B(l,T)*l
        s = 1.0 / np.sqrt(planck(wl_m, Tstar) * wl_m)
        return s / np.median(s, axis=1, keepdims=True)      # median-normalised shape per planet
    FLOOR = 3e-5     # ~30 ppm systematic floor in transit depth
    def noisy(X, Tstar, med_sigma, colored, draws=3):
        shape = photon_shape(Tstar) if colored else np.ones_like(X)
        sig = np.sqrt((med_sigma * shape) ** 2 + FLOOR ** 2)
        return X, sig
    def acc_noise(med_sigma, colored, draws=3):
        accs = []
        for _ in range(draws):
            corr = tot = 0
            for X, y, Tstar in tests:
                _, sig = noisy(X, Tstar, med_sigma, colored)
                Xp = X + rng.normal(0, 1, X.shape) * sig
                pred = xgb.predict(pca.transform(raw.transform(Xp)))
                corr += int((pred == y).sum()); tot += len(y)
            accs.append(100 * corr / tot)
        return np.mean(accs), np.std(accs)
    for med, lab in [(5e-5, "SNR~30"), (1e-4, "SNR~15"), (2e-4, "SNR~7")]:
        ac, sc_ = acc_noise(med, colored=True)
        aw, sw = acc_noise(med, colored=False)
        L.append(f"  {lab} colored {med:.0e}     {ac:>9.2f}% {ac-base:>+8.2f}  (white {aw:.2f}%)")

    txt = "\n".join(L)
    print(txt)
    with open("final_results/H2_realistic_systematics.txt", "w") as fh:
        fh.write(txt + "\n")
    print("Wrote final_results/H2_realistic_systematics.txt")


if __name__ == "__main__":
    main()
