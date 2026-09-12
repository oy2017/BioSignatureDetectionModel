"""Does trust depend on the host star? Frozen screen accuracy per host-Teff band, per mismatch.

M-dwarf hosts are where the ExoSim2 and ExoRad noise shapes disagree (Spearman 0.80 at
2500 K vs 0.985 at 7500 K) and where stellar contamination is strongest. If the loss under
each mismatch concentrates on cool hosts, the reliability envelope should be stated per host
type, which is directly usable for target selection.

Usage: python host_dependence.py --config ariel
Writes results/ariel_host_dependence.txt / .csv
"""
import argparse, json, os, sys
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, TESTS, centres, load_split, metrics  # noqa: E402
from augment import shifted_test, corr_noise  # noqa: E402
from noise import add_noise  # noqa: E402

BANDS = [("M (<4000 K)", 0, 4000), ("K (4000-5300)", 4000, 5300), ("G (5300-6000)", 5300, 6000), ("F+ (>6000)", 6000, 1e9)]
CASES = ["cloud_1e4Pa", "cloud_1e3Pa", "haze_3p0e7", "haze_2p4e8", "tlse_spots05", "tlse_spots10", "tlse_spots20", "tlse_mixed",
         "exotransmit", "exomol", "quenched", "compound_spots20_haze3e7"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel"); a = ap.parse_args(); cfg = a.config
    cen = centres(cfg)
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib")); ff, fm = fr["features"], fr["model"]
    Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
    Xnf = np.vstack([np.load(os.path.join(DATA, f"{t}_{cfg}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    Tst = Pc["s temperature"].to_numpy()
    correct = lambda X, y: ((fm.predict_proba(ff.transform(X))[:, 1] >= 0.5).astype(int) == y)

    sets = [("clean", Xc, yc)]
    for c in CASES:
        if all(os.path.exists(os.path.join(DATA, f"{t}_native_{c}.npy")) for t in TESTS):
            Xs, ys = shifted_test(c, cfg); sets.append((c, Xs, ys))
    rng = np.random.default_rng(SEED + 71)
    sets.append(("correlated_snr8", corr_noise(Xc, rng, 8, Xnf=Xnf, params=Pc, cen=cen), yc))
    sets.append(("white_snr5", corr_noise(Xc, rng, 5, kind="white", Xnf=Xnf, params=Pc, cen=cen), yc))
    # the consortium-simulator noise shape at the training SNR, the case the host-star divergence predicts
    Xe, _ = add_noise(Xnf, Pc, cen, snr=15.0, shape="exosim", seed=777); sets.append(("exosim_shape_snr15", Xe, yc))
    Xe, _ = add_noise(Xnf, Pc, cen, snr=7.0, shape="exosim", seed=778); sets.append(("exosim_shape_snr7", Xe, yc))

    rows = []; c0 = correct(Xc, yc)
    for name, X, y in sets:
        cr = correct(X, y)
        for band, lo, hi in BANDS:
            m = (Tst >= lo) & (Tst < hi)
            rows.append(dict(case=name, band=band, n=int(m.sum()), accuracy=cr[m].mean(), clean=c0[m].mean(), loss=(c0[m].mean() - cr[m].mean()) * 100))
    df = pd.DataFrame(rows)
    L = [f"Frozen screen ({best}) accuracy by host-star band, configuration {cfg}", "",
         "loss = clean accuracy of the SAME band minus shifted accuracy, in points", "",
         f"{'case':<26}" + "".join(f"{b:>18}" for b, _, _ in BANDS)]
    for name in df.case.unique():
        g = df[df.case == name].set_index("band")
        if name == "clean":
            L.append(f"{name:<26}" + "".join(f"{g.loc[b,'accuracy']*100:16.2f}%  " for b, _, _ in BANDS))
            L.append(f"{'  n':<26}" + "".join(f"{g.loc[b,'n']:>18d}" for b, _, _ in BANDS)); continue
        L.append(f"{name:<26}" + "".join(f"{g.loc[b,'accuracy']*100:9.2f}% ({g.loc[b,'loss']:+5.2f})" for b, _, _ in BANDS))
    sh = df[df.case != "clean"]
    ratio = sh.groupby("band").loss.mean()
    L += ["", "mean loss over shifted cases, by band: " + ", ".join(f"{b} {ratio[b]:+.2f}" for b, _, _ in BANDS)]
    spot_rows = sh[sh.case.str.startswith("tlse_spots")]; other = sh[~sh.case.str.startswith("tlse") & ~sh.case.str.startswith("compound")]
    L += [f"stellar-contamination cases only: M {spot_rows[spot_rows.band.str.startswith('M')].loss.mean():+.2f}"
          f" vs F+ {spot_rows[spot_rows.band.str.startswith('F')].loss.mean():+.2f};"
          f" every other axis: M {other[other.band.str.startswith('M')].loss.mean():+.2f} vs F+ {other[other.band.str.startswith('F')].loss.mean():+.2f}"]
    open(os.path.join(RESULTS, f"{cfg}_host_dependence.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, f"{cfg}_host_dependence.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
