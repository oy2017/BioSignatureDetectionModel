"""WASP-39 b control twins (REVISION_FINDINGS.md, F1): the clear H2/He twin at the published abundances, twins with
a grey cloud deck at eight pressures, clear twins at four water abundances, and the benchmark best-fit model binned
directly, all scored like the real spectrum (all-input verdicts at 1e-4, Tier-1 frequencies, H2O scores).

Usage: python wasp39_cloud_twins.py   (after wasp39_assemble.py)   -> results/wasp39/cloud_twins.txt
"""
import sys, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)   # numpy on macOS Accelerate reports spurious FP flags from matmul
V3 = os.path.dirname(os.path.abspath(__file__)); os.chdir(V3); sys.path.insert(0, V3)
import generate_grid as G, alfnoor_faithful as AF, forward_model as FM, jwst_to_ariel as J, alfnoor_jwst as AJ
from bin_spectra import bin_native
G.FILL_GAS = AF.FILL
edges, labels = J.tier3_layout(); wl = np.load("data/native_wl.npy")
M = pd.read_parquet("data/mcs/mcs_params.parquet"); r = M[M.name == "WASP-39b"].iloc[0].to_dict()
r.update({"atm top_pressure": 1e-4, "atm CO": -15.0, "atm O3": -15.0, "atm CH4": -7.0, "atm H2O": -2.3, "atm CO2": -4.5, "atm NH3": -7.0})
cov = pd.read_csv("results/wasp39/coverage_benchmark.csv"); real = cov.depth.to_numpy(); filled = cov.filled.to_numpy(bool)
sig = J.ariel_sigma("WASP-39b", edges); L = AJ.learners("all"); _out = []
def score(x, name):
    V = AJ.verdict_table(x, variants=("all",)); V = V[V.threshold == "1e-4"]; F = AJ.tier_frequencies(x, sig, n=300, variants=("all",))
    sc = {k: float(V[(V.molecule == "H2O") & (V.learner == k)].score.iloc[0]) for k in ("KNN", "MLP", "RFC", "SVC")}
    _out.append(f"{name:<34} dispersion {np.std(x[~filled])*1e6:5.0f} ppm   " + "  ".join(f"{m}:{''.join(str(int(v)) for v in V[V.molecule == m].sort_values('learner').present)} {F[F.molecule == m].frequency.mean()*100:3.0f}%" for m in ("H2O", "CH4", "CO2", "NH3")) + "   H2O scores " + " ".join(f"{k} {v:+.2f}" for k, v in sc.items())); print(_out[-1])
score(real, "real (benchmark + MIRI)")
wl_m, y_m = __import__("jwst_readers").carter_model("data/jwst/wasp39/10161743/extracted/ZENODO/5_MODEL_SPECTRUM/BestFitModel.txt")
score(bin_native(y_m[None, :], wl_m, edges)[0], "benchmark model, binned directly")
for cp in (None, 1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1):
    y = FM.spectrum(r, wl, G.GASES, G.FILL_GAS, cloud_pressure=cp)
    if y is None: print(cp, "failed"); continue
    x = bin_native(np.asarray(y, float)[None, :], wl, edges)[0]
    score(x, f"clear twin" if cp is None else f"twin, grey deck at {cp:g} Pa")
for mh in (-3.0, -2.5, -2.0, -1.5):    # water abundance of a clear twin
    rr = dict(r); rr["atm H2O"] = mh
    y = FM.spectrum(rr, wl, G.GASES, G.FILL_GAS); x = bin_native(np.asarray(y, float)[None, :], wl, edges)[0]
    score(x, f"clear twin, log H2O = {mh}")
open("results/wasp39/cloud_twins.txt", "w").write("\n".join(_out) + "\n")
