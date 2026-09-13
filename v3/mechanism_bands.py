"""Where the omitted-species loss comes from: band restoration and transplant (manuscript Section 3.2).

On the pooled five test sets, with the clean-trained Tier-3 screen and the test sets' own noise
(sigma from the clean spectrum, seeds 2001-2005, exactly as augment.shifted_test):
  restore    absorber spectra with the bins inside a window put back to their clean values
  transplant clean spectra with only the bins inside a window taken from the absorber spectra
Windows: 2.75-3.05 um (the water-methane gap the HCN/C2H2 3-um bands fill) and 2.75-4.3 um.

Usage: python mechanism_bands.py [--case absorbers]
Writes results/ariel_mechanism_bands.txt / .csv
"""
import argparse, json, os, sys
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SNR, TESTS, centres, configs, load_split, metrics, noise_spec, base_config  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
from noise import sigma_matrix  # noqa: E402

WINDOWS = [(2.75, 3.05), (2.75, 4.30)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="absorbers"); ap.add_argument("--config", default="ariel"); a = ap.parse_args()
    cfg = a.config; cen = centres(cfg); edges = np.array(configs()[cfg]["edges"]); wl = np.load(os.path.join(DATA, "native_wl.npy"))
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]; fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
    f, m = fr["features"], fr["model"]; nshape, nlevel = noise_spec(cfg)
    clean, shifted, sigs, epss, ys = [], [], [], [], []
    for k, t in enumerate(TESTS):
        _, y, P = load_split(t, cfg, noisy=False)
        cb = np.load(os.path.join(DATA, f"{t}_{base_config(cfg)}.npy")).astype(np.float64)
        sb = bin_native(np.load(os.path.join(DATA, f"{t}_native_{a.case}.npy")).astype(np.float64), wl, edges)
        sb = np.where(np.isfinite(sb), sb, cb)
        sig = sigma_matrix(cb, P["s temperature"].to_numpy(), cen, SNR, nshape, nlevel)
        eps = np.random.default_rng(2000 + k + 1).normal(0.0, 1.0, sig.shape)
        clean.append(cb); shifted.append(sb); sigs.append(sig); epss.append(eps); ys.append(y)
    C, S, SIG, EPS, y = (np.vstack(clean), np.vstack(shifted), np.vstack(sigs), np.vstack(epss), np.concatenate(ys))
    acc = lambda Xb: metrics(y, m.predict_proba(f.transform(Xb + EPS * SIG))[:, 1])["accuracy"]
    base_clean, base_shift = acc(C), acc(S)
    rows = [dict(variant="clean", window="", accuracy=base_clean), dict(variant=a.case, window="", accuracy=base_shift)]
    L = [f"Mechanism of the {a.case} loss, {cfg} ({len(cen)} bins), {best}: clean {base_clean*100:.2f} %, {a.case} {base_shift*100:.2f} % (loss {100*(base_clean-base_shift):.1f} points)", ""]
    for lo, hi in WINDOWS:
        w = (cen >= lo) & (cen <= hi)
        R = S.copy(); R[:, w] = C[:, w]; T = C.copy(); T[:, w] = S[:, w]
        ar, at = acc(R), acc(T)
        rows += [dict(variant="restore", window=f"{lo}-{hi}", accuracy=ar), dict(variant="transplant", window=f"{lo}-{hi}", accuracy=at)]
        L.append(f"window {lo}-{hi} um ({w.sum()} bins): restoring it recovers {100*(ar-base_shift):.1f} of {100*(base_clean-base_shift):.1f} points; "
                 f"transplanting only it into clean spectra costs {100*(base_clean-at):.1f}")
    open(os.path.join(RESULTS, f"{cfg}_mechanism_bands.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"{cfg}_mechanism_bands.csv"), index=False); print("\n".join(L))


if __name__ == "__main__":
    main()
