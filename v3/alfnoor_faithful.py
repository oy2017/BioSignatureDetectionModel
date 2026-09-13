"""The consortium's Tier-1 screen (Mugnai et al. 2021, AJ 162, 288), rebuilt a second time after a
line-by-line re-read of their Sections 2.1, 2.2 and 2.5 found three departures in alfnoor_screen.py:

  1. Classifier input.  Their "observed spectra" are the TauREx spectra "binned at Ariel's Tier 3
     spectral resolution" (their Tier 3: R = 20 / 100 / 30 in NIRSpec / AIRS-CH0 / AIRS-CH1) and
     scattered with "the noise estimated with ArielRad at each spectral bin ... a re-scaled version
     of the Tier 3 noise, obtained by combining the number of transit observations needed to match
     the Tier 1 required SNR" (Sec. 2.2; Fig. 1 and Fig. 3 show these points).  The ML classifiers
     "gather information from all the spectral data points" (Sec. 4.5).  alfnoor_screen.py binned
     to the seven Tier-1 points instead.
  2. Atmosphere.  100 layers from 1e6 to 1e-4 Pa and an H2/He fill with He/H2 = 0.17 (Sec. 2.2);
     alfnoor_screen.py used a 1 Pa top and pure H2.  The top pressure alone halves feature
     amplitudes on a typical planet.
  3. Noise level.  ArielRad noise for the integer number of Tier-1 transits, so the achieved Tier-1
     SNR is >= 7.  The candidate list gives integer transit counts only; the exact count is
     estimated from the Tier-3 count (N1_exact ~ (N3 - 0.5) / r, r = median N3/N1 over planets
     with N1 >= 20), clipped to (N1 - 1, N1].

Noise per Tier-3 bin: the ExoSim 2 shape at the bin centres, scaled within each Tier-1 point so
that inverse-variance combination of its Tier-3 bins gives exactly the Tier-1 point sigma
(modulation_5H / 7 x shape, x sqrt(N1_exact / N1) when the transit correction is on).

Same planets, same abundance and cloud draws as alfnoor_screen.py (identical seed).

Usage:
  python alfnoor_faithful.py --render --jobs 8
  python alfnoor_faithful.py --fit --natives faithful --layout tier3_r20 --noise exactN
Writes data/alfnoor_faithful/*.npy, results/alfnoor_faithful_<natives>_<layout>_<noise>.txt/.csv
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED, configs  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
from noise import nsr_shapes, EXOSIM_NPZ  # noqa: E402
import noise as NZ  # noqa: E402
import ariel_bins as AB  # noqa: E402
import alfnoor_screen as AS  # noqa: E402

FILL = ["H2", "He"]            # TauREx default He/H2 ratio 0.1757 (paper: 0.17)
TOP_PA = 1e-4
DIRS = {"old": os.path.join(DATA, "alfnoor"), "faithful": os.path.join(DATA, "alfnoor_faithful")}


def layout_edges(layout):
    if layout == "tier3_r20":
        return np.array(AB.ariel_edges(AB.tier_channels(20, 100, 30))[0])
    return np.array(configs()[layout]["edges"])


def _worker(rows, wl, gases, fixed):
    import generate_grid as G, shift_aerosol as A
    G.FILL_GAS = FILL; A.FILL_GAS = FILL
    return AS._w(rows, wl, gases, fixed)


def render(P, wl, jobs, gases=None, fixed=None, haze=None):
    rows = []
    for i in range(len(P)):
        r = P.iloc[i].to_dict()
        if haze is not None: r["_haze"] = haze
        rows.append(r)
    chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
    res = Parallel(n_jobs=jobs)(delayed(_worker)(c, wl, gases, fixed or {}) for c in chunks)
    X = np.full((len(rows), len(wl)), np.nan, dtype=np.float32)
    for i, s in enumerate(s for c in res for s in c):
        if s is not None: X[i] = s
    return X


def do_render(jobs):
    out_dir = DIRS["faithful"]; os.makedirs(out_dir, exist_ok=True)
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    Ptr = pd.read_parquet(os.path.join(DIRS["old"], "pop3_params.parquet"))
    Pte = pd.read_parquet(os.path.join(DIRS["old"], "pop1_params.parquet"))
    for P in (Ptr, Pte):
        P["atm top_pressure"] = TOP_PA; P["atm fill_gas"] = "H2+He"
    Ptr.to_parquet(os.path.join(out_dir, "pop3_params.parquet")); Pte.to_parquet(os.path.join(out_dir, "pop1_params.parquet"))
    import generate_grid as G
    jobs_ = [("pop3_native", Ptr, {}), ("pop1_native", Pte, {}),
             ("pop1_native_haze3e7", Pte, {"haze": 3e7}), ("pop1_native_cloud1e2", Pte, {"fixed": {"cloud_pressure": 1e2}}),
             ("pop1_native_absorbers", Pte, {"gases": list(G.GASES) + ["HCN", "C2H2"]})]
    for name, P, kw in jobs_:
        out = os.path.join(out_dir, f"{name}.npy")
        if os.path.exists(out): print(f"  {name}: exists"); continue
        t0 = time.time(); X = render(P, wl, jobs, **kw); bad = int((~np.all(np.isfinite(X), axis=1)).sum())
        np.save(out, X); print(f"  {name}: {X.shape}, {bad} failures, {time.time()-t0:.0f} s", flush=True)


def n1_exact(P):
    M = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet"))[["name", "tier1_transits", "tier3_transits"]]
    big = M[M.tier1_transits >= 20]; r = float(np.median(big.tier3_transits / big.tier1_transits))
    m = P[["name"]].merge(M.drop_duplicates("name"), on="name", how="left")
    n1 = m.tier1_transits.to_numpy(float); est = (m.tier3_transits.to_numpy(float) - 0.5) / r
    lo = np.maximum(n1 - 1.0, 0.0) + 1e-3
    return np.clip(est, lo, n1), n1, r


def radiometric_sigma(P, edges, k2=None):
    """Tier-3-style per-bin noise at the Tier-1 transit count: 'a re-scaled version of the Tier 3 noise,
    obtained by combining the number of transit observations needed to match the Tier 1 required SNR'
    (Mugnai et al. 2021, Sec. 2.2). Shared implementation: noise.radiometric_sigma."""
    return NZ.radiometric_sigma(P, edges, "tier1_transits", k2)


def sigma(P, edges, mode, factor=1.0):
    """Per-bin sigma on the given layout, calibrated so each Tier-1 point carries the Tier-1 noise."""
    if mode == "radiometric":
        return factor * radiometric_sigma(P, edges)[0]
    t1_edges = np.array(configs()["tier1"]["edges"]); t1_cen = 0.5 * (t1_edges[1:] + t1_edges[:-1])
    cen = 0.5 * (edges[1:] + edges[:-1])
    s_t1 = AS.tier1_sigma(P, t1_cen, factor)                       # (n, 7): modulation_5H / 7 x shape
    if mode == "exactN":
        ne, n1, _ = n1_exact(P); s_t1 = s_t1 * np.sqrt(ne / n1)[:, None]
    if len(cen) == len(t1_cen) and np.allclose(cen, t1_cen):
        return s_t1
    teffs, shapes = nsr_shapes(cen, EXOSIM_NPZ, edges=edges)
    node = teffs[np.argmin(np.abs(teffs[None, :] - P["s temperature"].to_numpy()[:, None]), axis=1)]
    a = np.vstack([shapes[int(t)] for t in node])                  # (n, nb) relative shape at the bin centres
    point = np.clip(np.searchsorted(t1_edges, cen, side="right") - 1, 0, len(t1_cen) - 1)
    out = np.empty_like(a)
    for i in range(len(t1_cen)):
        j = point == i
        c = s_t1[:, i] * np.sqrt((1.0 / a[:, j] ** 2).sum(1))       # inverse-variance sum reproduces s_t1
        out[:, j] = c[:, None] * a[:, j]
    return out


def do_fit(natives, layout, mode, factor=1.0):
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import shift_tlse as T
    src = DIRS[natives]; wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = layout_edges(layout)
    Ptr = pd.read_parquet(os.path.join(src, "pop3_params.parquet")); Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet"))
    load = lambda n: np.load(os.path.join(src, f"{n}.npy")).astype(np.float64)
    Xtr_n = load("pop3_native"); ok_tr = np.all(np.isfinite(Xtr_n), axis=1); Ptr = Ptr[ok_tr].reset_index(drop=True); Xtr_n = Xtr_n[ok_tr]
    Xte_n = load("pop1_native"); ok = np.all(np.isfinite(Xte_n), axis=1)
    cases = {"clean": Xte_n}
    for name in ("haze3e7", "cloud1e2", "absorbers"):
        Y = load(f"pop1_native_{name}"); ok &= np.all(np.isfinite(Y), axis=1); cases[name] = Y
    Pte = Pte[ok].reset_index(drop=True); cases = {k: v[ok] for k, v in cases.items()}
    T.set_grid(wl); Tst = Pte["s temperature"].to_numpy(); logg = np.log10(T.G_SUN * Pte["s mass"].to_numpy() / Pte["s radius"].to_numpy() ** 2 * 100)
    for f in (0.10, 0.20):
        cases[f"spots{int(f*100)}"] = cases["clean"] * T.contamination(Tst, logg, f, 0.0)
    rng = np.random.default_rng(SEED + 405)
    Xtr = bin_native(Xtr_n, wl, edges); Xtr = Xtr + rng.normal(0, 1, Xtr.shape) * sigma(Ptr, edges, mode, factor)
    binned = {k: bin_native(v, wl, edges) for k, v in cases.items()}
    sig = sigma(Pte, edges, mode, factor); eps = rng.normal(0, 1, sig.shape)
    noisy = {k: v + eps * sig for k, v in binned.items()}
    noisy["noise_x2"] = binned["clean"] + eps * 2 * sig; noisy["noise_x3"] = binned["clean"] + eps * 3 * sig
    norm = lambda X: (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    Ztr = norm(Xtr); Z = {k: norm(v) for k, v in noisy.items()}
    amp = np.ptp(binned["clean"], axis=1)
    t1_equiv = sigma(Pte, np.array(configs()["tier1"]["edges"]), mode, factor)
    snr_t1 = np.median(Pte["modulation_5H"].to_numpy()[:, None] / t1_equiv, axis=1)
    makers = {"KNN": lambda: KNeighborsClassifier(), "MLP": lambda: MLPClassifier(random_state=SEED),
              "RFC": lambda: RandomForestClassifier(random_state=SEED, n_jobs=8), "SVC": lambda: SVC()}
    tag = f"{natives}_{layout}_{mode}" + ("" if factor == 1.0 else f"_x{factor:g}")
    L = [f"Mugnai et al. 2021 Tier-1 screen, faithful rebuild [{tag}]: natives {natives}, layout {layout} ({len(edges)-1} bins), noise {mode} x {factor}",
         f"train {len(Ztr)} spectra, test {len(Pte)} planets; Tier-1 SNR on 5H modulation, median {np.median(snr_t1):.1f} (IQR {np.percentile(snr_t1,25):.1f}-{np.percentile(snr_t1,75):.1f})",
         f"median peak-to-peak of the clean binned spectra {np.median(amp)*1e6:.0f} ppm; median per-bin sigma {np.median(sig)*1e6:.0f} ppm",
         "their Table 6 at 1e-4: CH4 82-86, CO2 79-83, H2O 71-78, NH3 82-87 %", ""]
    rows = []
    for th_name, th in AS.THRESH.items():
        L.append(f"=== threshold: abundance > {th_name}")
        L.append(f"{'molecule':<9}{'model':<6}{'base':>7}" + "".join(f"{c:>11}" for c in Z))
        for mol in AS.MOLS:
            ytr = (Ptr[f"atm {mol}"] > th).astype(int).to_numpy(); yte = (Pte[f"atm {mol}"] > th).astype(int).to_numpy()
            for mname, mk in makers.items():
                m = mk().fit(Ztr, ytr); r = dict(threshold=th_name, molecule=mol, model=mname, base_rate=max(yte.mean(), 1 - yte.mean()))
                for c, Zc in Z.items():
                    r[c] = (m.predict(Zc) == yte).mean()
                rows.append(r)
                L.append(f"{mol:<9}{mname:<6}{r['base_rate']*100:6.1f}%" + "".join(f"{r[c]*100:10.1f}%" for c in Z))
        L.append("")
    knn = NearestNeighbors(n_neighbors=10).fit(Ztr); d_clean = knn.kneighbors(Z["clean"])[0].mean(1); thr_d = np.quantile(d_clean, 0.9)
    L.append("=== decline rules at the 1e-4 threshold: accepted accuracy (coverage), thresholds fixed on clean")
    L.append(f"{'molecule':<9}{'rule':<10}" + "".join(f"{c:>16}" for c in Z))
    for mol in AS.MOLS:
        ytr = (Ptr[f"atm {mol}"] > -4).astype(int).to_numpy(); yte = (Pte[f"atm {mol}"] > -4).astype(int).to_numpy()
        fitted = {mname: mk().fit(Ztr, ytr) for mname, mk in makers.items()}
        votes = lambda Zc: np.column_stack([m.predict(Zc) for m in fitted.values()])
        thr_e = np.quantile(votes(Z["clean"]).std(1), 0.9)
        for rule in ("ensemble", "knn"):
            cells, rr = "", dict(threshold="1e-4", molecule=mol, model=f"vote+{rule}", base_rate=max(yte.mean(), 1 - yte.mean()))
            for c, Zc in Z.items():
                V = votes(Zc); pred = (V.mean(1) >= 0.5).astype(int)
                keep = (V.std(1) <= thr_e) if rule == "ensemble" else (knn.kneighbors(Zc)[0].mean(1) <= thr_d)
                acc_k = (pred[keep] == yte[keep]).mean() if keep.any() else np.nan
                cells += f"{acc_k*100:8.1f} ({keep.mean()*100:3.0f}%)"; rr[c] = acc_k; rr[f"{c}_coverage"] = keep.mean()
            rows.append(rr); L.append(f"{mol:<9}{rule:<10}" + cells)
    df = pd.DataFrame(rows)
    mid = df[(df.threshold == "1e-4") & ~df.model.str.startswith("vote")]
    L += ["", "At 1e-4, mean over the four classifiers:"]
    for mol in AS.MOLS:
        g = mid[mid.molecule == mol]
        L.append(f"  {mol:<4} clean {g.clean.mean()*100:5.1f} (range {g.clean.min()*100:.0f}-{g.clean.max()*100:.0f}); cost in points: "
                 + ", ".join(f"{c} {(g.clean - g[c]).mean()*100:+.1f}" for c in Z if c != "clean"))
    open(os.path.join(RESULTS, f"alfnoor_faithful_{tag}.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, f"alfnoor_faithful_{tag}.csv"), index=False); print("\n".join(L[:4] + L[-6:]), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--render", action="store_true"); ap.add_argument("--fit", action="store_true")
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--natives", default="faithful", choices=list(DIRS))
    ap.add_argument("--layout", default="tier3_r20"); ap.add_argument("--noise", default="exactN", choices=["requirement", "exactN", "radiometric"])
    ap.add_argument("--factor", type=float, default=1.0); a = ap.parse_args()
    if a.render: do_render(a.jobs)
    if a.fit: do_fit(a.natives, a.layout, a.noise, a.factor)


if __name__ == "__main__":
    main()
