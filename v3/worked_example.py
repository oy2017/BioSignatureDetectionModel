"""A worked case of indistinguishable compositions for Section 3.2: planets whose H2O label lies at least a decade above the threshold, undecided by the retrieval, with the best opposite-label fit of (eq1). Writes results/worked_example.txt."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import generate_grid as G  # noqa: E402
import identifiability as I  # noqa: E402

src = AF.DIRS["faithful"]; wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = AF.layout_edges("tier3_r20"); Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet")); Y = np.load(os.path.join(src, "pop1_native.npy")).astype(float); ok = np.isfinite(Y).all(1); Pte = Pte[ok].reset_index(drop=False).rename(columns={"index": "_index"}); Y = Y[ok]
mcs = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet"))[["name", "atm temperature"]].rename(columns={"atm temperature": "T_p"}); Pte = Pte.merge(mcs.drop_duplicates("name"), on="name", how="left"); sig = AF.sigma(Pte, edges, "radiometric"); R = pd.read_csv(os.path.join(RESULTS, "identifiability_calibration.csv")).drop_duplicates("name").set_index("name").reindex(Pte.name.to_numpy())
G.FILL_GAS = AF.FILL; c = np.where((R.H2O_true.to_numpy() > -3) & (R.H2O_p_true.to_numpy() < 0.95))[0]; rng = np.random.default_rng(SEED + 4400); pick = rng.choice(c, min(12, len(c)), replace=False); L = [f"{len(c)} planets have H2O more than a decade above 1e-4 and undecided at 0.95"]; best = None; bin_ = None
for k in pick:
    r = Pte.iloc[k].to_dict(); pl = I.Planet(r, Y[k], wl, edges, sig[k], "none", list(G.GASES), None); d, bx = pl.dchi2("H2O", 3, rng); L.append(f"{r['name']}: dchi2 {d:.3f}, p_true {R.H2O_p_true.iloc[k]:.3f}")
    if bx is not None and (best is None or d < best[0]): best = (d, bx, k, r, pl)
d, bx, k, r, pl = best; ab = dict(zip([m for m in I.MOLS if m != "H2O"], bx[2:])); ab["H2O"] = I.THRESH; y = pl.render(bx[0], bx[1], ab); t = pl.truth; dd = t - y; dd = dd - np.sum(dd / pl.sigma ** 2) / np.sum(1 / pl.sigma ** 2)
L += ["", f"Worked case: {r['name']}, dchi2 {d:.3f} over 104 bins, p_true {R.H2O_p_true.iloc[k]:.3f}", "true: T %.0f K, cloud top %.3g Pa, " % (r["atm temperature"], r["cloud_pressure"]) + ", ".join(f"{m} {r[f'atm {m}']:.2f}" for m in I.MOLS), "opposite-label fit: T %.0f K, cloud top %.3g Pa, " % (bx[0], 10 ** bx[1]) + ", ".join(f"{m} {ab[m]:.2f}" for m in I.MOLS), f"largest difference {np.abs(dd).max() * 1e6:.0f} ppm, {np.abs(dd / pl.sigma).max():.2f} sigma; median sigma {np.median(pl.sigma) * 1e6:.0f} ppm"]
t = "\n".join(L); open(os.path.join(RESULTS, "worked_example.txt"), "w").write(t + "\n"); print(t)
