"""Controls for the JWST verdicts (REVISION_PLAN.md, A6-A7; findings in REVISION_FINDINGS.md).

For each planet with a converted spectrum: (1) a simulated clear H2/He "twin" with the planet's MCS bulk
parameters and the published abundances, binned and pushed through the SAME coverage mask and flat fill
as the real spectrum, noise-free and scattered with the real per-bin JWST errors (200 draws); (2) the
10-nearest-neighbour distance of the real spectrum, and of the twins, from the training set, against the
threshold that declines 10 % of clean POP-I planets (the note's decline rule). If the twin is detected
and inside the training distribution while the real spectrum is missed and outside it, the miss is the
real atmosphere's shape, not the noise, the coverage or the classifier's in-simulator skill.

Usage: python jwst_controls.py      (after jwst_panel.py and wasp39_assemble.py)
Writes results/jwst_panel/controls.txt
"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)   # numpy on macOS Accelerate reports spurious FP flags from matmul
V3 = os.path.dirname(os.path.abspath(__file__)); os.chdir(V3); sys.path.insert(0, V3)
_out = []
_print = print
def print(*a, **k):
    _print(*a, **k); _out.append(" ".join(str(x) for x in a))
import generate_grid as G, alfnoor_faithful as AF, jwst_to_ariel as J, alfnoor_jwst as AJ
from bin_spectra import bin_native
from sklearn.neighbors import NearestNeighbors
G.FILL_GAS = AF.FILL
edges, labels = J.tier3_layout(); wl = np.load("data/native_wl.npy")
M = pd.read_parquet("data/mcs/mcs_params.parquet")
# published abundances (log10 VMR) from the survey; others at the grid floor
TWINS = {"TOI-270d": {"CH4": -1.64, "H2O": -1.10, "CO2": -1.67, "NH3": -5.0},
         "K2-18b":   {"CH4": -1.74, "H2O": -4.0, "CO2": -3.35, "NH3": -5.6},
         "WASP-107b": {"CH4": -5.8, "H2O": -2.1, "CO2": -3.9, "NH3": -5.0},
         "GJ3470b":  {"CH4": -4.05, "H2O": -1.08, "CO2": -2.47, "NH3": -7.3},
         "HAT-P-26b": {"CH4": -5.5, "H2O": -2.07, "CO2": -2.91, "NH3": -4.2},
         "WASP-166b": {"CH4": -7.9, "H2O": -1.30, "CO2": -2.23, "NH3": -4.0},
         "K2-18b-alt": {"CH4": -1.74, "H2O": -4.0, "CO2": -3.35, "NH3": -5.6},
         "WASP-39b": {"CH4": -7.0, "H2O": -2.3, "CO2": -4.5, "NH3": -7.0},
         "TOI-1231b": {"CH4": -0.85, "H2O": -3.0, "CO2": -1.90, "NH3": -4.0},   # Sarkar+2026; H2O and NH3 at their 95 % upper limits' side
         "K2-18b-hu": {"CH4": -1.74, "H2O": -4.0, "CO2": -3.35, "NH3": -5.6},
         "TOI-270d-hm": {"CH4": -1.64, "H2O": -1.10, "CO2": -1.67, "NH3": -5.0}}   # WASP-39 b abundances: the benchmark + MIRI assembly (wasp39_assemble.py)
# training-set distance rule, as in the note: 10-NN distance, threshold declining 10 % of clean test planets
Xtr, Ptr, _ = AJ.training_set(); Ztr = AJ.znorm(Xtr, np.ones(104, bool))
src = AF.DIRS["faithful"]; Xte = np.load(f"{src}/pop1_native.npy"); ok = np.all(np.isfinite(Xte), 1)
Pte = pd.read_parquet(f"{src}/pop1_params.parquet")[ok].reset_index(drop=True)
rng = np.random.default_rng(42 + 405); _ = rng.normal(0, 1, Xtr.shape)          # consume the training draw as do_fit does
Xb = bin_native(Xte[ok].astype(float), wl, edges); Xb = Xb + rng.normal(0, 1, Xb.shape) * AF.sigma(Pte, edges, "radiometric")
knn = NearestNeighbors(n_neighbors=10).fit(Ztr); d_clean = knn.kneighbors(AJ.znorm(Xb, np.ones(104, bool)))[0].mean(1); thr = np.quantile(d_clean, 0.9)
print(f"kNN distance threshold declining 10 % of clean POP-I planets: {thr:.2f} (median clean {np.median(d_clean):.2f})\n")
print(f"{'planet':<10} {'input':<14} {'kNN dist':>9} {'declined':>9}   verdicts at 1e-4 (all inputs; KNN MLP RFC SVC) and Tier-1 frequency")
for pl, ab in TWINS.items():
    covf = f"results/jwst_panel/{pl}_coverage.csv" if pl != "WASP-39b" else "results/wasp39/coverage_benchmark.csv"
    cov = pd.read_csv(covf); filled = cov.filled.to_numpy(bool); real = cov.depth.to_numpy()
    if pl == "WASP-39b":
        real = real.copy(); real[filled] = np.nanmean(real[~filled])
    r = M[M.name == pl.replace("-alt", "").replace("-hu", "").replace("-hm", "")].iloc[0].to_dict(); r.update({"atm top_pressure": 1e-4, "atm CO": -15.0, "atm O3": -15.0, **{f"atm {k}": v for k, v in ab.items()}})
    y = G.one_native(r, wl)
    if y is None: print(pl, "twin failed to render"); continue
    twin = bin_native(y.astype(float)[None, :], wl, edges)[0]; twin_m = twin.copy(); twin_m[filled] = np.nanmean(twin[~filled])
    sig = J.ariel_sigma(pl.replace("-alt", "").replace("-hu", "").replace("-hm", ""), edges)
    for name, x in (("real", real), ("twin, masked", twin_m), ("twin, full", twin)):
        d = knn.kneighbors(AJ.znorm(x, np.ones(104, bool)))[0].mean()
        V = AJ.verdict_table(x, variants=("all",)); V = V[V.threshold == "1e-4"]
        F = AJ.tier_frequencies(x, sig, n=200, variants=("all",))
        cells = "  ".join(f"{m}:{''.join(str(int(v)) for v in V[V.molecule == m].sort_values('learner').present)} {F[F.molecule == m].frequency.mean()*100:3.0f}%" for m in ("CH4", "H2O", "CO2", "NH3"))
        print(f"{pl:<10} {name:<14} {d:9.2f} {'yes' if d > thr else 'no':>9}   {cells}")
    amp_real = np.nanstd(real[~filled]) * 1e6; amp_twin = np.std(twin[~filled]) * 1e6
    print(f"{'':<10} {'':<14} {'':>9} {'':>9}   dispersion of the measured bins: real {amp_real:.0f} ppm, H2/He twin {amp_twin:.0f} ppm; filled bins {filled.sum()}/104\n")

# control 2: the twin scattered with the real per-bin errors, through the same mask
print("\n== twins scattered with the real per-bin JWST errors (200 draws): detection frequency and decline rate")
L = AJ.learners("all"); rng2 = np.random.default_rng(3)
for pl, ab in TWINS.items():
    covf = f"results/jwst_panel/{pl}_coverage.csv" if pl != "WASP-39b" else "results/wasp39/coverage_benchmark.csv"
    cov = pd.read_csv(covf); filled = cov.filled.to_numpy(bool); err = cov.err.to_numpy(); err = np.where(filled | ~np.isfinite(err), 0.0, err)
    r = M[M.name == pl.replace("-alt", "").replace("-hu", "").replace("-hm", "")].iloc[0].to_dict(); r.update({"atm top_pressure": 1e-4, "atm CO": -15.0, "atm O3": -15.0, **{f"atm {k}": v for k, v in ab.items()}})
    y = G.one_native(r, wl)
    if y is None: continue
    twin = bin_native(y.astype(float)[None, :], wl, edges)[0]
    for name, scale in (("real errors", 1.0), ("twice the errors", 2.0)):
        X = twin[None, :] + rng2.normal(0, 1, (200, 104)) * err[None, :] * scale; X[:, filled] = X[:, ~filled].mean(1, keepdims=True)
        Z = AJ.znorm(X, np.ones(104, bool)); dec = (knn.kneighbors(Z)[0].mean(1) > thr).mean()
        freq = {m: np.mean([L["models"][("1e-4", m, k)].predict(Z).mean() for k in ("KNN", "MLP", "RFC", "SVC")]) for m in ("CH4", "H2O", "CO2", "NH3")}
        print(f"{pl:<10} twin + {name:<18} median err {np.median(err[~filled])*1e6*scale:4.0f} ppm  declined {dec*100:3.0f} %   " + "  ".join(f"{m} {v*100:3.0f}%" for m, v in freq.items()))
open("results/jwst_panel/controls.txt", "w").write("\n".join(_out) + "\n")
