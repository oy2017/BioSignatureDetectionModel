"""Is 'quenching helps the C/O screen' robust to the eddy-diffusion coefficient?

The Axis 8 result used one K_zz (1e9 cm^2/s). generate_grid.py --mode quenched --kzz <v>
re-renders the test splits at other values (tagged _quenched_kzz<log10>). This scores the
frozen screen on each and reports, per temperature band, the accuracy and the label
separation in log10 H2O that drives it. Also the reverse-direction cost (a screen trained on
quenched spectra deployed on equilibrium) is NOT repeated here; it needs a training render.

Usage: python kzz_eval.py --config ariel
Writes results/ariel_kzz_sweep.txt / .csv
"""
import argparse, glob, json, os, re, sys
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, TESTS, load_split, metrics  # noqa: E402
from augment import shifted_test  # noqa: E402

T_BANDS = [(500, 1000), (1000, 1500), (1500, 2500)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel"); a = ap.parse_args(); cfg = a.config
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib")); ff, fm = fr["features"], fr["model"]
    Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    Tp = Pc["atm temperature"].to_numpy(); lab = Pc["label_co"].to_numpy()
    correct = lambda X, y: ((fm.predict_proba(ff.transform(X))[:, 1] >= 0.5).astype(int) == y)
    c0 = correct(Xc, yc)

    tags = ["quenched_kzz7", "quenched_kzz8", "quenched", "quenched_kzz10", "quenched_kzz11"]
    kzz_of = {"quenched": 9}
    rows, L = [], [f"Quenching vs K_zz, frozen screen {best}, configuration {cfg}; clean accuracy {c0.mean()*100:.2f}%", "",
                   f"{'K_zz':>6}{'all':>9}" + "".join(f"{f'T {lo}-{hi}':>16}" for lo, hi in T_BANDS) + f"{'dH2O cool (dex)':>17}{'dCH4 cool':>11}"]
    for lo, hi in T_BANDS:
        m = (Tp >= lo) & (Tp < hi)
    for tag in tags:
        if not all(os.path.exists(os.path.join(DATA, f"{t}_native_{tag}.npy")) for t in TESTS):
            print(f"  {tag}: not rendered yet"); continue
        k = kzz_of.get(tag) or int(re.search(r"kzz(\d+)", tag).group(1))
        Xs, ys = shifted_test(tag, cfg); cr = correct(Xs, ys)
        Q = pd.concat([pd.read_parquet(os.path.join(DATA, f"{t}_params_{tag}.parquet")) for t in TESTS], ignore_index=True)
        cool = (Tp >= 500) & (Tp < 1000)
        dH2O = Q.loc[cool & (lab == 1), "atm H2O"].mean() - Q.loc[cool & (lab == 0), "atm H2O"].mean()
        dCH4 = Q.loc[cool & (lab == 1), "atm CH4"].mean() - Q.loc[cool & (lab == 0), "atm CH4"].mean()
        r = dict(kzz=10.0 ** k, accuracy=cr.mean(), dH2O_cool=dH2O, dCH4_cool=dCH4)
        cells = ""
        for lo, hi in T_BANDS:
            m = (Tp >= lo) & (Tp < hi); r[f"acc_{lo}_{hi}"] = cr[m].mean(); r[f"clean_{lo}_{hi}"] = c0[m].mean()
            cells += f"{cr[m].mean()*100:8.2f} ({(cr[m].mean()-c0[m].mean())*100:+5.2f})"
        rows.append(r)
        L.append(f"{f'1e{k}':>6}{cr.mean()*100:8.2f}%" + cells + f"{dH2O:17.2f}{dCH4:11.2f}")
    E = Pc
    cool = (Tp >= 500) & (Tp < 1000)
    L += ["", f"equilibrium reference, cool planets: dH2O {E.loc[cool & (lab==1), 'atm H2O'].mean() - E.loc[cool & (lab==0), 'atm H2O'].mean():.2f} dex,"
          f" dCH4 {E.loc[cool & (lab==1), 'atm CH4'].mean() - E.loc[cool & (lab==0), 'atm CH4'].mean():.2f} dex",
          "", "Numbers in parentheses: change from the clean accuracy of the same temperature band.",
          "The claim 'quenching helps' holds at a given K_zz if the all-planet change is positive and the cool band gains most."]
    open(os.path.join(RESULTS, f"{cfg}_kzz_sweep.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"{cfg}_kzz_sweep.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
