"""The consortium's own Tier-1 screen (Mugnai et al. 2021, AJ 162, 288, "Alfnoor"), rebuilt from
their text and run through the mismatch map.

Their design, reproduced as literally as the paper allows (every deviation listed at the bottom):
  planets     the Ariel candidate list (here: the 965 known MCS targets already in data/mcs)
  POP-III     training population: each planet 4x, T in [0.7, 1.05] x T_p, CH4/H2O/CO2/NH3
              log-uniform 1e-9..1e-2, grey cloud deck log-uniform 5e2..1e6 Pa, H2/He atmosphere
  POP-I       test population: same planets once, abundances log-uniform 1e-7..1e-2
  Tier 1      7 points: 3 photometric + 1 NIRSpec + 2 AIRS-CH0 + 1 AIRS-CH1 (our 'tier1' layout)
  noise       Gaussian scatter with the Tier-1 requirement sigma per target (SNR 7 on the 5-H
              modulation; ExoSim2 shape), applied to training and test spectra alike
  labels      per molecule, abundance > 1e-4 (their middle threshold; also 1e-5 and 1e-3)
  inputs      spectrum normalised to zero mean and unit dispersion
  models      scikit-learn defaults: KNeighborsClassifier(5), MLPClassifier(100),
              RandomForestClassifier (sqrt features), SVC
Their Table 6 at the 1e-4 threshold: CH4 82-86, CO2 79-83, H2O 71-78, NH3 82-87 %.

Mismatch axes on the POP-I test population (same planets, same draws):
  haze 3e7 m^-3, stellar spots 10 % and 20 % (multiplicative TLSE factor), noise x2 and x3 the
  requirement sigma, cloud deck at 1e2 Pa (outside their range), and the unmodelled absorbers
  HCN + C2H2 drawn log-uniform 1e-7..1e-4 (Moses 2013 carbon-rich / quenched range).

Usage: python alfnoor_screen.py --render --jobs 8 ; python alfnoor_screen.py --fit
Writes data/alfnoor/*.npy, results/alfnoor_screen.txt / .csv
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED, centres, configs  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
from noise import nsr_shapes, EXOSIM_NPZ  # noqa: E402
import generate_grid as G  # noqa: E402
import shift_aerosol as A  # noqa: E402
import shift_absorbers as SA  # noqa: E402

OUT = os.path.join(DATA, "alfnoor"); os.makedirs(OUT, exist_ok=True)
MOLS = ["CH4", "H2O", "CO2", "NH3"]
THRESH = {"1e-5": -5, "1e-4": -4, "1e-3": -3}


def population(P0, rng, n_rep, lo_ab):
    P = pd.concat([P0] * n_rep, ignore_index=True); n = len(P)
    P["atm temperature"] = P["atm temperature"].to_numpy() * rng.uniform(0.7, 1.05, n)
    P["atm base_pressure"] = 1e6; P["atm top_pressure"] = 1.0
    for g in G.GASES:
        P[f"atm {g}"] = rng.uniform(lo_ab, -2, n) if g in MOLS else G.LOG_FLOOR
    P["cloud_pressure"] = 10 ** rng.uniform(np.log10(5e2), 6, n)
    P["atm fill_gas"] = "H2"
    return P


def _w(rows, wl, gases, fixed):
    import warnings; warnings.filterwarnings("ignore")
    out = []
    for r in rows:
        cp = r.pop("cloud_pressure"); hz = r.pop("_haze", None)
        kw = dict(fixed); kw.setdefault("cloud_pressure", cp)
        if hz is not None: kw["haze_density"] = hz
        if gases is None:
            out.append(A.one_native(r, wl, **kw))
        else:
            out.append(SA.one_native(r, wl, gases) if kw.get("cloud_pressure") is None else _one_native_gases_cloud(r, wl, gases, kw["cloud_pressure"]))
    return out


def _one_native_gases_cloud(row, wl_native, gases, cloud_pressure):
    """SA.one_native with a grey cloud deck: rebuild the system with both."""
    try:
        from multirex import Atmosphere, Planet, Star, System
        atm = Atmosphere(temperature=float(row["atm temperature"]), base_pressure=float(row["atm base_pressure"]),
                         top_pressure=float(row["atm top_pressure"]), composition={g: float(row[f"atm {g}"]) for g in gases},
                         fill_gas=G.FILL_GAS, cloud_pressure=float(cloud_pressure))
        planet = Planet(radius=float(row["p_radius"]), mass=float(row["p_mass"]), atmosphere=atm)
        star = Star(temperature=float(row["s temperature"]), radius=float(row["s radius"]), mass=float(row["s mass"]))
        system = System(planet=planet, star=star, sma=float(row["sma"])); system.make_tm()
        wn, rprs = system.transmission.model()[:2]
        wl = 1e4 / np.asarray(wn); o = np.argsort(wl); wl, rprs = wl[o], np.asarray(rprs)[o]
        m = (wl >= G.WL_MIN) & (wl <= G.WL_MAX); y = rprs[m]
        if y.shape[0] != wl_native.shape[0] or not np.allclose(wl[m], wl_native):
            y = np.interp(wl_native, wl[m], y)
        if not np.all(np.isfinite(y)) or np.any(y > 1.0) or np.any(y <= 0):
            return None
        return y.astype(np.float32)
    except Exception:
        return None


def render(P, wl, jobs, gases=None, fixed=None, haze=None):
    rows = []
    for i in range(len(P)):
        r = P.iloc[i].to_dict()
        if haze is not None: r["_haze"] = haze
        rows.append(r)
    chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
    res = Parallel(n_jobs=jobs)(delayed(_w)(c, wl, gases, fixed or {}) for c in chunks)
    X = np.full((len(rows), len(wl)), np.nan, dtype=np.float32)
    for i, s in enumerate(s for c in res for s in c):
        if s is not None: X[i] = s
    return X


def do_render(jobs):
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    P0 = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet"))
    P0 = P0[["name", "p_radius", "p_mass", "atm temperature", "s temperature", "s radius", "s mass", "sma",
             "tier1_transits", "modulation_5H"]].reset_index(drop=True)
    rng = np.random.default_rng(SEED + 404)
    Ptr = population(P0, rng, 4, -9); Pte = population(P0, rng, 1, -7)
    Pte["atm HCN"] = rng.uniform(-7, -4, len(Pte)); Pte["atm C2H2"] = rng.uniform(-7, -4, len(Pte))
    Ptr.to_parquet(os.path.join(OUT, "pop3_params.parquet")); Pte.to_parquet(os.path.join(OUT, "pop1_params.parquet"))
    jobs_ = [("pop3_native", Ptr, {}), ("pop1_native", Pte, {}),
             ("pop1_native_haze3e7", Pte, {"haze": 3e7}), ("pop1_native_cloud1e2", Pte, {"fixed": {"cloud_pressure": 1e2}}),
             ("pop1_native_absorbers", Pte, {"gases": list(G.GASES) + ["HCN", "C2H2"]})]
    for name, P, kw in jobs_:
        out = os.path.join(OUT, f"{name}.npy")
        if os.path.exists(out): print(f"  {name}: exists"); continue
        t0 = time.time(); X = render(P, wl, jobs, **kw); bad = int((~np.all(np.isfinite(X), axis=1)).sum())
        np.save(out, X); print(f"  {name}: {X.shape}, {bad} failures, {time.time()-t0:.0f} s", flush=True)


def tier1_sigma(P, cen, factor=1.0):
    teffs, shapes = nsr_shapes(np.asarray(cen, float), EXOSIM_NPZ)
    node = teffs[np.argmin(np.abs(teffs[None, :] - P["s temperature"].to_numpy()[:, None]), axis=1)]
    S = np.vstack([shapes[int(t)] for t in node])
    return factor * (P["modulation_5H"].to_numpy() / 7.0)[:, None] * S


def do_fit():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import shift_tlse as T
    wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = np.array(configs()["tier1"]["edges"]); cen = centres("tier1")
    Ptr = pd.read_parquet(os.path.join(OUT, "pop3_params.parquet")); Pte = pd.read_parquet(os.path.join(OUT, "pop1_params.parquet"))
    load = lambda n: np.load(os.path.join(OUT, f"{n}.npy")).astype(np.float64)
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
    Xtr = bin_native(Xtr_n, wl, edges); Xtr = Xtr + rng.normal(0, 1, Xtr.shape) * tier1_sigma(Ptr, cen)
    binned = {k: bin_native(v, wl, edges) for k, v in cases.items()}
    sig = tier1_sigma(Pte, cen); eps = rng.normal(0, 1, sig.shape)
    noisy = {k: v + eps * sig for k, v in binned.items()}
    noisy["noise_x2"] = binned["clean"] + eps * 2 * sig; noisy["noise_x3"] = binned["clean"] + eps * 3 * sig
    norm = lambda X: (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    Ztr = norm(Xtr); Z = {k: norm(v) for k, v in noisy.items()}
    makers = {"KNN": lambda: KNeighborsClassifier(5), "MLP": lambda: MLPClassifier((100,), max_iter=1000, random_state=SEED),
              "RFC": lambda: RandomForestClassifier(random_state=SEED, n_jobs=8), "SVC": lambda: SVC()}
    rows, L = [], ["Mugnai et al. 2021 Tier-1 molecular screen, rebuilt (POP-III train, POP-I test, 7 points, requirement noise)",
                   f"train {len(Ztr)} spectra, test {len(Pte)} planets; their Table 6 at 1e-4: CH4 82-86, CO2 79-83, H2O 71-78, NH3 82-87 %", ""]
    for th_name, th in THRESH.items():
        L.append(f"=== threshold: abundance > {th_name}")
        L.append(f"{'molecule':<9}{'model':<6}" + "".join(f"{c:>11}" for c in Z))
        for mol in MOLS:
            ytr = (Ptr[f"atm {mol}"] > th).astype(int).to_numpy(); yte = (Pte[f"atm {mol}"] > th).astype(int).to_numpy()
            for mname, mk in makers.items():
                m = mk().fit(Ztr, ytr); r = dict(threshold=th_name, molecule=mol, model=mname, base_rate=yte.mean())
                for c, Zc in Z.items():
                    r[c] = (m.predict(Zc) == yte).mean()
                rows.append(r)
                L.append(f"{mol:<9}{mname:<6}" + "".join(f"{r[c]*100:10.1f}%" for c in Z))
        L.append("")
    df = pd.DataFrame(rows)
    mid = df[df.threshold == "1e-4"]
    L.append("At the 1e-4 threshold, mean over the four molecules and four models: " +
             ", ".join(f"{c} {mid[c].mean()*100:.1f}" for c in Z))
    L += ["", "Deviations from Mugnai et al. 2021: forward model MultiREx/TauREx 3 with Exo-Transmit tables (theirs: TauREx 3 with ExoMol k-tables);",
          "noise shape from ExoSim2 rather than ArielRad, level set by the Tier-1 requirement per target; 965 known MCS planets (2026 list)",
          "rather than their 1000 (2019 list incl. TESS predictions); training spectra noised once, not resampled per epoch."]
    open(os.path.join(RESULTS, "alfnoor_screen.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, "alfnoor_screen.csv"), index=False); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--render", action="store_true"); ap.add_argument("--fit", action="store_true")
    ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    if a.render: do_render(a.jobs)
    if a.fit: do_fit()


if __name__ == "__main__":
    main()
