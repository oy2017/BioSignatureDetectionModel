"""The omitted-species result on Ariel's real target list under mission noise (audit check, 2026-09-13).
The 965 known MCS targets (mcs_testset.py) re-rendered with HCN and C2H2 at their FastChem equilibrium
abundances; scored with the clean-trained screens at Tier-3, Tier-2 and Tier-1 binning, noise from the payload
model at each tier's integer transit count (noise.radiometric_sigma), identical noise draw with and without.
Writes results/mcs_absorbers.txt"""
import os, sys, json, numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, configs, metrics
from bin_spectra import bin_native
from noise import radiometric_sigma
import generate_grid as G, shift_absorbers as SA

def main():
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    P = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet")); X0 = np.load(os.path.join(DATA, "mcs", "mcs_native.npy")).astype(float)
    out = os.path.join(DATA, "mcs", "mcs_native_absorbers.npy")
    if not os.path.exists(out):
        chem = G.Chemistry()
        for g, h in SA.EXTRA.items(): chem.idx[g] = chem.fc.getGasSpeciesIndex(h)
        comp = [chem.composition(r["atm temperature"], r.co_ratio, r.mh) for _, r in P.iterrows()]
        for g in G.GASES: assert np.allclose([c[g] for c in comp], P[f"atm {g}"].to_numpy(), atol=1e-6), g
        for g in SA.EXTRA: P[f"atm {g}"] = [c[g] for c in comp]
        rows = [P.iloc[i].to_dict() for i in range(len(P))]
        res = Parallel(n_jobs=8)(delayed(SA.worker)(c, wl, list(G.GASES) + list(SA.EXTRA)) for c in G.chunked(rows, 50))
        X = np.full(X0.shape, np.nan, np.float32)
        for i, s in enumerate(x for c in res for x in c):
            if s is not None: X[i] = s
        np.save(out, X); P.to_parquet(os.path.join(DATA, "mcs", "mcs_params_absorbers.parquet"))
    X1 = np.load(out).astype(float)
    ok = np.isfinite(X0).all(1) & np.isfinite(X1).all(1); P = P[ok].reset_index(drop=True); X0, X1 = X0[ok], X1[ok]
    y = P.label_co.to_numpy(); cr = y == 1
    best = json.load(open(os.path.join(RESULTS, "ariel_best.json")))["best"]
    L = [f"Omitted HCN + C2H2 on the {len(P)} known Ariel targets, payload noise at each tier's own transit count", ""]
    for cfg, model, trans in (("ariel", f"ariel_{best}", "tier2_transits"), ("tier2", "tier2_norm_xgb", "tier2_transits"), ("tier1", "tier1_norm_xgb", "tier1_transits")):
        e = np.array(configs()[cfg]["edges"]); fr = joblib.load(os.path.join(MODELS, f"{model}.joblib")); f, m = fr["features"], fr["model"]
        S, _ = radiometric_sigma(P, e, trans); eps = np.random.default_rng(SEED + 909).normal(0, 1, S.shape)
        p0 = m.predict_proba(f.transform(bin_native(X0, wl, e) + eps * S))[:, 1]; p1 = m.predict_proba(f.transform(bin_native(X1, wl, e) + eps * S))[:, 1]
        a0, a1 = ((p0 >= .5) == y), ((p1 >= .5) == y)
        L.append(f"{cfg:<6} ({trans[:5]}): all {a0.mean()*100:5.1f} -> {a1.mean()*100:5.1f} %; carbon-rich {a0[cr].mean()*100:5.1f} -> {a1[cr].mean()*100:5.1f} %; "
                 f"oxygen-rich {a0[~cr].mean()*100:5.1f} -> {a1[~cr].mean()*100:5.1f} %; share called carbon-rich {np.mean(p0>=.5):.2f} -> {np.mean(p1>=.5):.2f}; "
                 f"mean p(carbon-rich) on carbon-rich {p0[cr].mean():.2f} -> {p1[cr].mean():.2f}")
    open(os.path.join(RESULTS, "mcs_absorbers.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
