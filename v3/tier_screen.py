"""The screen at the binning Ariel's tiers actually deliver.

Triage happens on the reconnaissance survey (Tier 1: R ~ 1 / 3 / 1 in NIRSpec / CH0 / CH1, seven
bins in all, SNR >= 7 on the assumed modulation) and the chemical survey (Tier 2: R ~ 10 / 50 /
10, 51 bins). The 102-bin 'ariel' layout used so far is Tier 3's native binning. The same
pipeline (per-spectrum normalized XGBoost, the Tier-3 hyper-parameters) is trained at each tier
and scored clean at SNR 15 and at the tier requirement SNR 7, then on the main mismatch cases.

Usage: python tier_screen.py
Writes results/tier_screen.txt / .csv and models/{tier1,tier2}_norm_xgb.joblib
"""
import json, os, sys
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SNR, TESTS, Features, load_split, metrics  # noqa: E402
from pipeline import make_xgb  # noqa: E402
from augment import shifted_test  # noqa: E402

CASES = ["cloud_1e4Pa", "cloud_1e3Pa", "haze_3p0e7", "tlse_spots10", "tlse_spots20", "exotransmit", "exomol", "quenched",
         "compound_spots20_haze3e7"]


def main():
    params = json.load(open(os.path.join(RESULTS, "ariel_best.json")))
    best = params["best"]; hp = joblib.load(os.path.join(MODELS, f"ariel_{best}.joblib"))["params"]
    rows, L = [], ["The screen at each tier's binning (norm_xgb, Tier-3 hyper-parameters, trained at SNR 15)", "",
                   f"{'tier':<8}{'bins':>5}{'clean SNR15':>12}{'clean SNR7':>11}" + "".join(f"{c[:12]:>13}" for c in CASES)]
    for cfg, nb in (("ariel", 102), ("tier2", 51), ("tier1", 7)):
        Xtr, ytr, _ = load_split("train", cfg)
        if cfg == "ariel":
            fr = joblib.load(os.path.join(MODELS, f"ariel_{best}.joblib")); f, m = fr["features"], fr["model"]
        else:
            f = Features("norm").fit(Xtr); m = make_xgb(hp).fit(f.transform(Xtr), ytr)
            joblib.dump({"features": f, "model": m, "config": cfg, "params": hp}, os.path.join(MODELS, f"{cfg}_norm_xgb.joblib"))
        acc = lambda X, y: metrics(y, m.predict_proba(f.transform(X))[:, 1])["accuracy"]
        Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
        X7 = np.vstack([load_split(t, cfg, snr=7.0, seed_offset=5)[0] for t in TESTS])
        r = dict(tier=cfg, bins=nb, clean_snr15=acc(Xc, yc), clean_snr7=acc(X7, yc))
        for c in CASES:
            if all(os.path.exists(os.path.join(DATA, f"{t}_native_{c}.npy")) for t in TESTS):
                Xs, ys = shifted_test(c, cfg); r[c] = acc(Xs, ys)
            else:
                r[c] = np.nan
        rows.append(r)
        L.append(f"{cfg:<8}{nb:>5}{r['clean_snr15']*100:11.2f}%{r['clean_snr7']*100:10.2f}%" + "".join(f"{r[c]*100:12.2f}%" for c in CASES))
        print(L[-1], flush=True)
    L += ["", "Tier 1 is what the triage decision would actually be made on; Tier 3 ('ariel') is what the study has used so far."]
    open(os.path.join(RESULTS, "tier_screen.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tier_screen.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
