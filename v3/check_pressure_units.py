import os, sys, json, numpy as np, pandas as pd, joblib, warnings
os.chdir("/mnt/c/Users/owenh/BioSignatureDetectionModel/v3"); sys.path.insert(0, os.getcwd())
from joblib import Parallel, delayed
from common import DATA, MODELS, RESULTS, configs, centres, metrics
from bin_spectra import bin_native
from noise import add_noise
S = "/tmp/claude-1000/-mnt-c-Users-owenh-BioSignatureDetectionModel/6df27690-b19a-4d43-895a-c8a267082e66/scratchpad/"

def worker(rows, wl, absorbers):
    warnings.filterwarnings("ignore")
    import taurex.opacity.exotransmit as ET
    if not getattr(ET.ExoTransmitOpacity, "_pfix", False):
        orig = ET.ExoTransmitOpacity._load_exo_transmit
        def patched(self, filename):
            orig(self, filename); self._pressure_grid = self._pressure_grid / 1e5
            self._min_pressure = self._pressure_grid.min(); self._max_pressure = self._pressure_grid.max()
        ET.ExoTransmitOpacity._load_exo_transmit = patched; ET.ExoTransmitOpacity._pfix = True
    import multirex, shift_aerosol as A, shift_absorbers as SA, generate_grid as G
    if absorbers:
        return [SA.one_native(r, wl, list(G.GASES) + ["HCN", "C2H2"]) for r in rows]
    return [A.one_native(r, wl) for r in rows]

wl = np.load(os.path.join(DATA, "native_wl.npy")); split = "test1"
out = {}
for name, pf, ab in (("clean", f"{split}_params.parquet", False), ("absorbers", f"{split}_params_absorbers.parquet", True)):
    f = S + f"pfix_{split}_{name}.npy"
    if os.path.exists(f): out[name] = np.load(f); continue
    P = pd.read_parquet(os.path.join(DATA, pf)); rows = [P.iloc[i].to_dict() for i in range(len(P))]
    ch = [rows[i:i+25] for i in range(0, len(rows), 25)]
    res = Parallel(n_jobs=8)(delayed(worker)(c, wl, ab) for c in ch)
    X = np.array([s if s is not None else np.full(len(wl), np.nan) for c in res for s in c]); np.save(f, X); out[name] = X
    print(name, "rendered", X.shape, "failures", int(np.isnan(X).any(1).sum()), flush=True)

cfg = "ariel"; edges = np.array(configs()[cfg]["edges"]); cen = centres(cfg)
best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]; fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib"))
ff, fm = fr["features"], fr["model"]
P = pd.read_parquet(os.path.join(DATA, f"{split}_params.parquet")); y = P["label_co"].to_numpy()
stored = {"clean": np.load(os.path.join(DATA, f"{split}_native.npy")).astype(float), "absorbers": np.load(os.path.join(DATA, f"{split}_native_absorbers.npy")).astype(float)}
ok = np.ones(len(P), bool)
for d in (stored, out):
    for v in d.values(): ok &= np.isfinite(v).all(1)
print(best, "planets", ok.sum())
cr = y[ok] == 1
for tag, d in (("stored (TauREx units)", stored), ("pressure fixed", out)):
    accs = {}
    for k, X in d.items():
        Xb = bin_native(X[ok].astype(float), wl, edges); Xn, _ = add_noise(Xb, P[ok], cen, snr=15.0, shape="ariel", seed=2001)
        p = fm.predict_proba(ff.transform(Xn))[:, 1]; accs[k] = (metrics(y[ok], p)["accuracy"], ((p >= .5) == y[ok])[cr].mean())
    print(f"{tag:22s} clean {accs['clean'][0]*100:.1f} (C-rich {accs['clean'][1]*100:.1f}) | with HCN+C2H2 {accs['absorbers'][0]*100:.1f} (C-rich {accs['absorbers'][1]*100:.1f})")
