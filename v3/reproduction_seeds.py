"""Seed-to-seed scatter of the reproduction (Appendix A): the faithful rebuild refit with other noise draws and learner
initialisations (alfnoor_faithful.py --fit --noise radiometric --seed N), against the published accuracies.

Usage: python reproduction_seeds.py [--seeds 1 2 3 4 5]   -> results/reproduction_seeds.{txt,csv}
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402

# Mugnai et al. 2021, Table 6 (accuracy %, POP-III train, POP-I test), as transcribed in the manuscript's Table A1
PUBLISHED = {("KNN", "CH4"): (79, 83, 85), ("KNN", "H2O"): (64, 71, 82), ("KNN", "CO2"): (77, 79, 82), ("KNN", "NH3"): (75, 82, 84),
             ("MLP", "CH4"): (78, 85, 87), ("MLP", "H2O"): (70, 76, 84), ("MLP", "CO2"): (77, 81, 83), ("MLP", "NH3"): (80, 86, 87),
             ("RFC", "CH4"): (77, 82, 87), ("RFC", "H2O"): (69, 74, 82), ("RFC", "CO2"): (76, 79, 83), ("RFC", "NH3"): (78, 85, 87),
             ("SVC", "CH4"): (79, 86, 89), ("SVC", "H2O"): (69, 78, 84), ("SVC", "CO2"): (79, 83, 84), ("SVC", "NH3"): (81, 87, 87)}
TH = ["1e-5", "1e-4", "1e-3"]; BASE = "alfnoor_faithful_faithful_tier3_r20_radiometric"


def load(tag):
    df = pd.read_csv(os.path.join(RESULTS, f"{BASE}{tag}.csv")); df = df[~df.model.str.startswith("vote")]
    lab = lambda t: {5: "1e-5", 4: "1e-4", 3: "1e-3"}[int(round(-np.log10(float(t))))]
    return {(r.model, r.molecule, lab(r.threshold)): 100 * r.clean for r in df.itertuples()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5]); a = ap.parse_args()
    runs = {0: load("")}; runs.update({s: load(f"_seed{s}") for s in a.seeds if os.path.exists(os.path.join(RESULTS, f"{BASE}_seed{s}.csv"))})
    rows = []
    for (m, mol), pub in PUBLISHED.items():
        for j, th in enumerate(TH):
            v = np.array([runs[s][(m, mol, th)] for s in runs]); v0 = runs[0][(m, mol, th)]
            rows.append(dict(model=m, molecule=mol, threshold=th, published=pub[j], committed=v0, mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                             lo=v.min(), hi=v.max(), n=len(v), inside=(pub[j] >= v.min() - 0.5) & (pub[j] <= v.max() + 0.5), dev_committed=v0 - pub[j], dev_mean=v.mean() - pub[j],
                             sd_test=100 * np.sqrt(v0 / 100 * (1 - v0 / 100) / 954)))   # binomial scatter of one 954-planet test set
    df = pd.DataFrame(rows); df["sd_combined"] = np.sqrt(df.sd ** 2 + df.sd_test ** 2 + 0.5 ** 2 / 3)   # seed + test-set sampling + integer rounding of the published value
    df.to_csv(os.path.join(RESULTS, "reproduction_seeds.csv"), index=False)
    L = [f"Reproduction seed scatter: {len(runs)} fits (committed seed + {len(runs)-1} others), 48 published accuracies (rounded to integers, so a published value within 0.5 of the seed range counts as inside)", ""]
    L.append(f"per-cell sd across seeds: median {df.sd.median():.2f}, max {df.sd.max():.2f} points; seed range (max - min): median {(df.hi-df.lo).median():.1f}, max {(df.hi-df.lo).max():.1f}")
    L.append(f"published inside the seed range: {int(df.inside.sum())} of 48; |committed - published| mean {df.dev_committed.abs().mean():.2f}, |seed mean - published| mean {df.dev_mean.abs().mean():.2f}")
    L.append(f"cells where |committed - published| exceeds 2 seed sd: {int((df.dev_committed.abs() > 2 * df.sd).sum())} of 48 (the seed sd omits the test-set sampling scatter: the published run is one draw of the population)")
    L.append(f"combined scatter per cell (seed sd, binomial sd of a 954-planet test set, rounding to integers): median {df.sd_combined.median():.2f} points; "
             f"cells beyond 2 combined sd: {int((df.dev_committed.abs() > 2 * df.sd_combined).sum())} of 48 (2.3 expected by chance); beyond 3: {int((df.dev_committed.abs() > 3 * df.sd_combined).sum())}; "
             f"chi2 of the 48 deviations against the combined scatter: {float((df.dev_committed ** 2 / df.sd_combined ** 2).sum()):.0f} for 48 cells")
    L.append(f"mean signed deviation (committed - published) by molecule: " + ", ".join(f"{m} {df[df.molecule == m].dev_committed.mean():+.2f}" for m in ["CH4", "H2O", "CO2", "NH3"]) + "; by threshold: " + ", ".join(f"{t} {df[df.threshold == t].dev_committed.mean():+.2f}" for t in TH))
    # the monotone-with-threshold pattern per seed
    mono = []
    for s in runs:
        mono.append(sum(all(runs[s][(m, mol, TH[j])] <= runs[s][(m, mol, TH[j+1])] for j in range(2)) for (m, mol) in PUBLISHED))
    L.append(f"classifier-molecule pairs with accuracy non-decreasing in the threshold, per seed (16 published): {mono}")
    L.append(""); L.append(f"{'model':<5}{'mol':<5}{'thr':<6}{'publ':>6}{'commit':>8}{'mean':>7}{'sd':>6}{'min':>7}{'max':>7}  inside")
    for r in df.itertuples():
        L.append(f"{r.model:<5}{r.molecule:<5}{r.threshold:<6}{r.published:6.0f}{r.committed:8.1f}{r.mean:7.1f}{r.sd:6.2f}{r.lo:7.1f}{r.hi:7.1f}  {'yes' if r.inside else 'NO'}")
    txt = "\n".join(L); print(txt); open(os.path.join(RESULTS, "reproduction_seeds.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
