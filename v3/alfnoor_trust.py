"""The full trust procedure on the consortium's Tier-1 molecular screen (Mugnai et al. 2021),
as re-implemented in alfnoor_faithful.py: the same questions the carbon-rich screen answers.

Screen   Tier-3 binning (R 20/100/30, 104 bins), Tier-1 noise from the payload model at each target's
         integer Tier-1 transit count, POP-III training (3,908 spectra), POP-I test (the 965 known
         Ariel targets), KNN / MLP / RFC / SVC at scikit-learn defaults, per-spectrum normalisation.
Labels   molecule present if its abundance exceeds 1e-4 (1e-5 and 1e-3 in the appendix table).

Mismatch axes on POP-I (same planets, same noise draw as the clean set; failed re-renders fall
back to clean):
  cloud     fixed grey deck at 1e4, 1e3 (inside the published cloud prior), 1e2, 1e1 Pa (outside)
  haze      Lee et al. Mie haze 2e5, 2e6, 3e7, 2.4e8, 1e10 m^-3 on top of each planet's own deck
  spots     unocculted spots 2, 5, 10, 20 % (contrast 0.85); faculae 10 %
  compound  spots 20 % + haze 3e7
  white     extra white noise to 1.5x, 2x, 3x the per-bin sigma
  correlated extra noise smoothed over 3 bins to the same per-bin variance, 1.5x, 2x, 3x
  ramp      a multiplicative tilt of 1 and 2 noise levels across the spectrum, sign per planet
  exomol    ExoMol cross sections for H2O, CH4, CO2 and CO (NH3, O3, O2 keep Exo-Transmit tables)
  exotransmit  the same atmospheres through Exo-Transmit's own code (He fill and deck included)
  absorbers HCN and C2H2 added at 1e-7..1e-4

Measurements (1e-4 threshold; per molecule, mean over the four classifiers unless stated):
  frozen      accuracy of the screen as published, on every case
  ceiling     accuracy of the same design retrained at the test condition (POP-III re-rendered)
  randomized  the design retrained on a randomized POP-III (haze on 60 % log-uniform 1e5-3e8, spots on
              70 % U(0, 20 %), noise x U(1, 3); the published random deck already covers clouds), and
              the three leave-one-ingredient-out variants
  detection   probability margin, ensemble spread (four classifiers), Mahalanobis, k-NN and PCA
              reconstruction distances; thresholds decline 10 % of CLEAN planets; accepted accuracy,
              coverage, credit against the clean selective baseline at equal coverage, AUROC(error),
              AUROC(shift)
  calibration expected calibration error and split-conformal coverage (90 %, calibrated on half the clean
              planets) for the three probabilistic classifiers
  host        loss by host type (M, K, G, F+)
  trade-off   Mahalanobis shift AUROC and margin error AUROC, frozen vs randomized

Usage:  python alfnoor_trust.py --render --jobs 8 ; python alfnoor_trust.py --fit
Writes data/alfnoor_faithful/*.npy and results/alfnoor_trust_*.csv / .txt
"""
import argparse, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as F  # noqa: E402
import alfnoor_screen as AS  # noqa: E402

OUT = F.DIRS["faithful"]; LAYOUT = "tier3_r20"; MOLS = AS.MOLS; TH = -4
HAZE = {"haze2e5": 2e5, "haze2e6": 2e6, "haze3e7": 3e7, "haze2p4e8": 2.4e8, "haze1e10": 1e10}
CLOUD = {"cloud1e4": 1e4, "cloud1e3": 1e3, "cloud1e2": 1e2, "cloud1e1": 1e1}
SPOTS = {"spots02": (0.02, 0), "spots05": (0.05, 0), "spots10": (0.10, 0), "spots20": (0.20, 0), "fac10": (0, 0.10)}
NOISE = {"white_x1.5": ("white", 1.5), "white_x2": ("white", 2.0), "white_x3": ("white", 3.0),
         "corr_x1.5": ("corr", 1.5), "corr_x2": ("corr", 2.0), "corr_x3": ("corr", 3.0),
         "ramp_x1": ("ramp", 1.0), "ramp_x2": ("ramp", 2.0)}
IN_RANGE = {**{k: True for k in ("cloud1e4", "cloud1e3", "haze2e5", "haze2e6", "haze3e7", "haze2p4e8", "spots02", "spots05",
                                 "spots10", "spots20", "compound", "white_x1.5", "white_x2", "white_x3")},
            **{k: False for k in ("cloud1e2", "cloud1e1", "haze1e10", "fac10", "corr_x1.5", "corr_x2", "corr_x3", "ramp_x1",
                                  "ramp_x2", "exomol", "exotransmit", "absorbers")}}
AXIS = {**{k: "cloud" for k in CLOUD}, **{k: "haze" for k in HAZE}, **{k: "spots" for k in SPOTS}, "compound": "compound",
        **{k: k.split("_")[0] for k in NOISE}, "exomol": "opacity", "exotransmit": "code", "absorbers": "absorbers"}
CEILING_TRAIN = ["haze3e7", "cloud1e2", "spots10", "spots20", "compound", "white_x2", "white_x3", "corr_x2", "corr_x3",
                 "ramp_x2", "exomol", "exotransmit", "absorbers"]
HOSTS = [("M", 0, 4000), ("K", 4000, 5300), ("G", 5300, 6000), ("F+", 6000, 1e9)]


# ------------------------------------------------------------------------------------------ rendering
def _params():
    Ptr = pd.read_parquet(os.path.join(OUT, "pop3_params.parquet")); Pte = pd.read_parquet(os.path.join(OUT, "pop1_params.parquet"))
    return Ptr, Pte


def _save(name, X):
    np.save(os.path.join(OUT, f"{name}.npy"), X)
    print(f"  {name}: {X.shape}, {int((~np.all(np.isfinite(X), axis=1)).sum())} failures", flush=True)


def _exists(name):
    return os.path.exists(os.path.join(OUT, f"{name}.npy"))


def _exomol_worker(rows, wl, swap):
    import warnings; warnings.filterwarnings("ignore")
    import generate_grid as G, shift_aerosol as A, shift_opacity as SO
    G.FILL_GAS = F.FILL; A.FILL_GAS = F.FILL; SO.configure_opacities(swap)
    out = []
    for r in rows:
        try:
            system = A.build_system(r, cloud_pressure=r["cloud_pressure"])
            wn, d = system.transmission.model()[:2]
            w = 1e4 / np.asarray(wn, float); o = np.argsort(w); w, d = w[o], np.asarray(d, float)[o]
            m = (w >= G.WL_MIN) & (w <= G.WL_MAX)
            y = bin_native(d[m][None, :], w[m], SO.native_edges(wl))[0]
            out.append(y.astype(np.float32) if np.all(np.isfinite(y)) and np.all(y > 0) and np.all(y < 1) else None)
        except Exception:
            out.append(None)
    return out


def _exotransmit_planet(index, row, grid):
    """Exo-Transmit's own code on a consortium atmosphere: H2/He fill (He/H2 = 0.17), the planet's grey deck."""
    import shift_exotransmit as S
    t0 = time.time()
    try:
        wd = S._worker_dir(); tp = S.write_tp(wd, row)
        vmr = {g: 10.0 ** float(row[f"atm {g}"]) for g in S.GASES}; rest = max(1.0 - sum(vmr.values()), 0.0)
        vmr["H2"] = rest / 1.17; vmr["He"] = rest * 0.17 / 1.17
        abund = {s: 0.0 for s in S.EOS_SPECIES}; abund.update(vmr)
        vals = "\t".join(f"{abund[s]:.6e}" for s in S.EOS_SPECIES)
        lines = ["T\t\tP\t\t" + "\t\t".join(S.EOS_SPECIES), ""]
        for p in S.EOS_P:
            lines.append(f"{p:.6e}"); lines.append("")
            for T in S.EOS_T:
                lines.append(f"{float(T):.6e}\t{p:.6e}\t{vals}")
            lines.append("")
        open(os.path.join(wd, "EOS", "eos_planet.dat"), "w").write("\n".join(lines) + "\n")
        outfile = S.write_userinput(wd, row, tp, "/EOS/eos_planet.dat")
        ui = open(os.path.join(wd, "userInput.in")).read().split("\n")
        k = ui.index("Pressure of cloud top (in Pa):"); ui[k + 1] = f"{float(row['cloud_pressure']):.6e}"
        open(os.path.join(wd, "userInput.in"), "w").write("\n".join(ui))
        if os.path.exists(outfile): os.remove(outfile)
        subprocess.run(["./Exo_Transmit"], cwd=wd, capture_output=True, text=True, timeout=S.EXO_TIMEOUT_S)
        d = np.loadtxt(outfile, skiprows=2)
        y = S.bin_to_grid(d[:, 0] * 1e6, d[:, 1] / 100.0, grid).astype(np.float32)
        return index, (y if np.all(np.isfinite(y)) else None), time.time() - t0
    except Exception:
        return index, None, time.time() - t0


def do_render(jobs):
    wl = np.load(os.path.join(DATA, "native_wl.npy")); Ptr, Pte = _params()
    import generate_grid as G
    rng = np.random.default_rng(SEED + 406)
    # test re-renders
    for name, lev in CLOUD.items():
        if not _exists(f"pop1_native_{name}"): _save(f"pop1_native_{name}", F.render(Pte, wl, jobs, fixed={"cloud_pressure": lev}))
    for name, lev in HAZE.items():
        if not _exists(f"pop1_native_{name}"): _save(f"pop1_native_{name}", F.render(Pte, wl, jobs, haze=lev))
    # training re-renders for the ceilings
    if not _exists("pop3_native_haze3e7"): _save("pop3_native_haze3e7", F.render(Ptr, wl, jobs, haze=3e7))
    if not _exists("pop3_native_cloud1e2"): _save("pop3_native_cloud1e2", F.render(Ptr, wl, jobs, fixed={"cloud_pressure": 1e2}))
    pab = os.path.join(OUT, "pop3_params_absorbers.parquet")
    if not os.path.exists(pab):
        P = Ptr.copy(); P["atm HCN"] = rng.uniform(-7, -4, len(P)); P["atm C2H2"] = rng.uniform(-7, -4, len(P)); P.to_parquet(pab)
    if not _exists("pop3_native_absorbers"):
        _save("pop3_native_absorbers", F.render(pd.read_parquet(pab), wl, jobs, gases=list(G.GASES) + ["HCN", "C2H2"]))
    # randomized training grid: draws, then the hazy planets rendered with their own density
    dfile = os.path.join(OUT, "pop3_rand_draws.parquet")
    if not os.path.exists(dfile):
        r2 = np.random.default_rng(SEED + 408); n = len(Ptr)
        has_haze = r2.random(n) < 0.6; has_spot = r2.random(n) < 0.7
        pd.DataFrame({"haze_density": np.where(has_haze, 10 ** r2.uniform(5, np.log10(3e8), n), np.nan),
                      "spot_frac": np.where(has_spot, r2.uniform(0, 0.20, n), 0.0),
                      "noise_factor": r2.uniform(1.0, 3.0, n)}).to_parquet(dfile)
    if not _exists("pop3_native_rand_haze"):
        D = pd.read_parquet(dfile); rows = []
        for i in range(len(Ptr)):
            r = Ptr.iloc[i].to_dict()
            if np.isfinite(D.haze_density.iloc[i]): r["_haze"] = float(D.haze_density.iloc[i])
            rows.append(r)
        chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
        res = Parallel(n_jobs=jobs)(delayed(F._worker)(c, wl, None, {}) for c in chunks)
        X = np.full((len(rows), len(wl)), np.nan, dtype=np.float32)
        for i, s in enumerate(s for c in res for s in c):
            if s is not None: X[i] = s
        _save("pop3_native_rand_haze", X)
    # ExoMol opacity swap (test and train)
    import shift_opacity as SO
    swap = SO.build_swap_dir("exomol")
    for tag, P in (("pop1", Pte), ("pop3", Ptr)):
        if _exists(f"{tag}_native_exomol"): continue
        rows = [P.iloc[i].to_dict() for i in range(len(P))]; chunks = [rows[i:i + 10] for i in range(0, len(rows), 10)]
        t0 = time.time(); res = Parallel(n_jobs=min(jobs, 6))(delayed(_exomol_worker)(c, wl, swap) for c in chunks)
        X = np.full((len(rows), len(wl)), np.nan, dtype=np.float32)
        for i, s in enumerate(s for c in res for s in c):
            if s is not None: X[i] = s
        _save(f"{tag}_native_exomol", X); print(f"    {time.time()-t0:.0f} s", flush=True)
    # Exo-Transmit's own code (test and train)
    import shift_exotransmit as S
    os.makedirs(S.WORK_ROOT, exist_ok=True)
    for tag, P in (("pop1", Pte), ("pop3", Ptr)):
        if _exists(f"{tag}_native_exotransmit"): continue
        t0 = time.time()
        res = Parallel(n_jobs=jobs)(delayed(_exotransmit_planet)(i, P.iloc[i].to_dict(), wl) for i in range(len(P)))
        X = np.full((len(P), len(wl)), np.nan, dtype=np.float32)
        for i, y, _ in res:
            if y is not None: X[i] = y
        _save(f"{tag}_native_exotransmit", X); print(f"    {time.time()-t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------ analysis
def makers():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    return {"KNN": lambda: KNeighborsClassifier(), "MLP": lambda: MLPClassifier(random_state=SEED),
            "RFC": lambda: RandomForestClassifier(random_state=SEED, n_jobs=8), "SVC": lambda: SVC()}


def prob(m, Z):
    if hasattr(m, "predict_proba") and not m.__class__.__name__ == "SVC":
        return m.predict_proba(Z)[:, 1]
    return 1.0 / (1.0 + np.exp(-m.decision_function(Z)))


norm = lambda X: (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def auroc(labels, score):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels).astype(int)
    return roc_auc_score(labels, score) if 0 < labels.sum() < len(labels) else np.nan


def ece(p, y, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def do_fit():
    import shift_tlse as T
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    t_start = time.time()
    wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = F.layout_edges(LAYOUT)
    Ptr, Pte = _params()
    load = lambda n: np.load(os.path.join(OUT, f"{n}.npy")).astype(np.float64)
    Xtr_n = load("pop3_native"); ok_tr = np.all(np.isfinite(Xtr_n), axis=1)
    Xte_n = load("pop1_native"); ok = np.all(np.isfinite(Xte_n), axis=1)
    Ptr = Ptr[ok_tr].reset_index(drop=True); Pte = Pte[ok].reset_index(drop=True)
    B = lambda X: bin_native(X, wl, edges)

    def shifted_native(name, base, mask, P):
        Y = load(name)[mask]; bad = ~np.all(np.isfinite(Y), axis=1); Y[bad] = base[bad]; return Y, int(bad.sum())

    clean_te_n = Xte_n[ok]; clean_tr_n = Xtr_n[ok_tr]
    # noise exactly as the reproduction run (alfnoor_faithful.do_fit): train draw first, then the test draw
    rng = np.random.default_rng(SEED + 405)
    sig_tr = F.sigma(Ptr, edges, "radiometric"); eps_tr = rng.normal(0, 1, (len(Ptr), len(edges) - 1))
    sig_te = F.sigma(Pte, edges, "radiometric"); eps_te = rng.normal(0, 1, sig_te.shape)
    rng_extra = np.random.default_rng(SEED + 407); eps2_te = rng_extra.normal(0, 1, sig_te.shape); eps2_tr = rng_extra.normal(0, 1, sig_tr.shape)
    sign_te = rng_extra.choice([-1.0, 1.0], (len(Pte), 1)); sign_tr = rng_extra.choice([-1.0, 1.0], (len(Ptr), 1))

    def contam(P, fs, ff):
        T.set_grid(wl); logg = np.log10(T.G_SUN * P["s mass"].to_numpy() / P["s radius"].to_numpy() ** 2 * 100)
        return T.contamination(P["s temperature"].to_numpy(), logg, fs, ff)

    def noisy(Xb, sig, eps, eps2, sign, kind=None, k=1.0):
        X = Xb + eps * sig
        if kind == "white":
            X = X + eps2 * sig * np.sqrt(k ** 2 - 1)
        elif kind == "corr":
            z = gaussian_filter1d(eps2, sigma=3.0, axis=1); z /= z.std(axis=1, keepdims=True) + 1e-12
            X = X + z * sig * np.sqrt(k ** 2 - 1)
        elif kind == "ramp":
            amp = k * np.median(sig, axis=1, keepdims=True) / np.abs(Xb).mean(axis=1, keepdims=True)
            X = X * (1 + sign * amp * np.linspace(-1, 1, Xb.shape[1]))
        return X

    # ---- test cases (noise-free binned, then noised with the shared draw)
    Bc = B(clean_te_n); tests, fallbacks = {"clean": noisy(Bc, sig_te, eps_te, eps2_te, sign_te)}, {}
    for name in list(CLOUD) + list(HAZE) + ["absorbers", "exomol", "exotransmit"]:
        Y, nb = shifted_native(f"pop1_native_{name}", clean_te_n, ok, Pte); fallbacks[name] = nb
        tests[name] = noisy(B(Y), sig_te, eps_te, eps2_te, sign_te)
    for name, (fs, ff) in SPOTS.items():
        tests[name] = noisy(B(clean_te_n * contam(Pte, fs, ff)), sig_te, eps_te, eps2_te, sign_te)
    Yh, _ = shifted_native("pop1_native_haze3e7", clean_te_n, ok, Pte)
    tests["compound"] = noisy(B(Yh * contam(Pte, 0.20, 0)), sig_te, eps_te, eps2_te, sign_te)
    for name, (kind, k) in NOISE.items():
        tests[name] = noisy(Bc, sig_te, eps_te, eps2_te, sign_te, kind, k)
    Z = {c: norm(X) for c, X in tests.items()}
    y_te = {th: {mol: (Pte[f"atm {mol}"] > th).astype(int).to_numpy() for mol in MOLS} for th in (-5, -4, -3)}

    # ---- training sets
    Btr = B(clean_tr_n)
    def train_set(name):
        if name == "clean": return noisy(Btr, sig_tr, eps_tr, eps2_tr, sign_tr)
        if name in ("haze3e7", "cloud1e2", "absorbers", "exomol", "exotransmit"):
            Y, _ = shifted_native(f"pop3_native_{name}", clean_tr_n, ok_tr, Ptr); return noisy(B(Y), sig_tr, eps_tr, eps2_tr, sign_tr)
        if name in SPOTS:
            fs, ff = SPOTS[name]; return noisy(B(clean_tr_n * contam(Ptr, fs, ff)), sig_tr, eps_tr, eps2_tr, sign_tr)
        if name == "compound":
            Y, _ = shifted_native("pop3_native_haze3e7", clean_tr_n, ok_tr, Ptr); return noisy(B(Y * contam(Ptr, 0.20, 0)), sig_tr, eps_tr, eps2_tr, sign_tr)
        if name in NOISE:
            kind, k = NOISE[name]; return noisy(Btr, sig_tr, eps_tr, eps2_tr, sign_tr, kind, k)
        raise KeyError(name)
    D = pd.read_parquet(os.path.join(OUT, "pop3_rand_draws.parquet"))[ok_tr].reset_index(drop=True)
    Yrh, _ = shifted_native("pop3_native_rand_haze", clean_tr_n, ok_tr, Ptr)
    def randomized_set(variant):
        haze = np.isfinite(D.haze_density.to_numpy()) & (variant != "no_haze")
        X = np.where(haze[:, None], Yrh, clean_tr_n)
        if variant != "no_spots":
            # contamination at each planet's own coverage f: eps_f = 1 / (1 - f t), t = 1 - 1/eps(f=1) (exact for spots only)
            t = 1.0 - 1.0 / contam(Ptr, 1.0, 0.0)
            X = X / (1.0 - D.spot_frac.to_numpy()[:, None] * t)
        Xb = B(X); k = D.noise_factor.to_numpy()[:, None] if variant != "no_noise" else 1.0
        return Xb + eps_tr * sig_tr + eps2_tr * sig_tr * np.sqrt(np.maximum(k ** 2 - 1, 0))

    MK = makers(); rows_frozen, rows_ceiling, rows_rand = [], [], []
    def fit_all(Xtrain, th=TH, mols=MOLS):
        Zt = norm(Xtrain)
        return Zt, {mol: {c: mk().fit(Zt, (Ptr[f"atm {mol}"] > th).astype(int).to_numpy()) for c, mk in MK.items()} for mol in mols}

    # ---- frozen screen: all thresholds, all cases
    Ztr, frozen = fit_all(train_set("clean"))
    frozen_by_th = {TH: frozen}
    for th in (-5, -3):
        frozen_by_th[th] = fit_all(train_set("clean"), th)[1]
    for th, models in frozen_by_th.items():
        for mol in MOLS:
            for c, m in models[mol].items():
                r = dict(threshold=f"1e{th}", molecule=mol, classifier=c)
                for case, Zc in Z.items():
                    r[case] = (m.predict(Zc) == y_te[th][mol]).mean()
                rows_frozen.append(r)
    frozen_df = pd.DataFrame(rows_frozen); print(f"frozen done {time.time()-t_start:.0f} s", flush=True)

    # ---- ceilings (1e-4)
    for name in CEILING_TRAIN:
        _, mods = fit_all(train_set(name))
        for mol in MOLS:
            for c, m in mods[mol].items():
                rows_ceiling.append(dict(case=name, molecule=mol, classifier=c, ceiling=(m.predict(Z[name]) == y_te[TH][mol]).mean()))
        print(f"ceiling {name} {time.time()-t_start:.0f} s", flush=True)
    ceiling_df = pd.DataFrame(rows_ceiling)

    # ---- randomized grid and held-out variants (1e-4)
    rand_models, rand_Ztr = {}, {}
    for v in ("full", "no_haze", "no_spots", "no_noise"):
        rand_Ztr[v], rand_models[v] = fit_all(randomized_set(v))
        for mol in MOLS:
            for c, m in rand_models[v][mol].items():
                r = dict(variant=v, molecule=mol, classifier=c)
                for case, Zc in Z.items():
                    r[case] = (m.predict(Zc) == y_te[TH][mol]).mean()
                rows_rand.append(r)
        print(f"randomized {v} {time.time()-t_start:.0f} s", flush=True)
    rand_df = pd.DataFrame(rows_rand)

    # ---- detection (frozen screen, 1e-4)
    def distance_scores(Zt):
        mu = Zt.mean(0); Ci = np.linalg.inv(np.cov(Zt, rowvar=False) + 1e-3 * np.eye(Zt.shape[1]))
        nn = NearestNeighbors(n_neighbors=10).fit(Zt); pca = PCA(n_components=0.99, random_state=SEED).fit(Zt)
        def f(Zc):
            d = Zc - mu
            return {"mahalanobis": np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d)), "knn": nn.kneighbors(Zc)[0].mean(1),
                    "pca_recon": np.linalg.norm(Zc - pca.inverse_transform(pca.transform(Zc)), axis=1)}
        return f
    dist = distance_scores(Ztr); dist_cache = {case: dist(Zc) for case, Zc in Z.items()}
    rows_det = []
    for mol in MOLS:
        y = y_te[TH][mol]; P_case = {case: {c: prob(m, Zc) for c, m in frozen[mol].items()} for case, Zc in Z.items()}
        for c in MK:
            S_case = {}
            for case in Z:
                p = P_case[case][c]; pred = (p >= 0.5).astype(int)
                sc = {"margin": 1 - np.abs(2 * p - 1), "ensemble": np.std(np.column_stack(list(P_case[case].values())), axis=1), **dist_cache[case]}
                S_case[case] = (pred, sc)
            pred0, sc0 = S_case["clean"]; wrong0 = pred0 != y
            for rule in sc0:
                thr = np.quantile(sc0[rule], 0.90); order0 = np.argsort(sc0[rule])
                for case, (pred, sc) in S_case.items():
                    keep = sc[rule] <= thr; wrong = pred != y; cov = keep.mean()
                    acc_k = (~wrong[keep]).mean() if keep.any() else np.nan
                    nkeep = max(int(round(cov * len(y))), 1); base = (~wrong0[order0[:nkeep]]).mean()
                    rows_det.append(dict(molecule=mol, classifier=c, rule=rule, case=case, all=(~wrong).mean(), coverage=cov, accepted=acc_k,
                                         credit=acc_k - base if np.isfinite(acc_k) else np.nan, auroc_error=auroc(wrong, sc[rule]),
                                         auroc_shift=auroc(np.r_[np.zeros(len(y)), np.ones(len(y))], np.r_[sc0[rule], sc[rule]]) if case != "clean" else np.nan))
    det_df = pd.DataFrame(rows_det); print(f"detection {time.time()-t_start:.0f} s", flush=True)

    # ---- calibration and conformal (probabilistic classifiers, 1e-4)
    rows_cal = []; idx = np.arange(len(Pte)); cal, ev = idx % 2 == 0, idx % 2 == 1
    for mol in MOLS:
        y = y_te[TH][mol]
        for c in ("KNN", "MLP", "RFC"):
            m = frozen[mol][c]; p0 = m.predict_proba(Z["clean"])[:, 1]
            s_cal = np.where(y[cal] == 1, 1 - p0[cal], p0[cal]); n = cal.sum()
            q = np.quantile(s_cal, min(1.0, np.ceil((n + 1) * 0.9) / n), method="higher")
            for case, Zc in Z.items():
                p = m.predict_proba(Zc)[:, 1][ev]; yy = y[ev]
                in1, in0 = (1 - p) <= q, p <= q; covered = np.where(yy == 1, in1, in0)
                rows_cal.append(dict(molecule=mol, classifier=c, case=case, accuracy=((p >= .5) == yy).mean(), ece=ece(p, yy),
                                     coverage=covered.mean(), empty=(~in1 & ~in0).mean(), ambiguous=(in1 & in0).mean()))
    cal_df = pd.DataFrame(rows_cal)

    # ---- host dependence (frozen, 1e-4, mean over molecules and classifiers)
    tst = Pte["s temperature"].to_numpy(); rows_host = []
    for case in ("clean", "spots10", "spots20", "haze3e7", "compound", "cloud1e2"):
        for h, lo, hi in HOSTS:
            mh = (tst >= lo) & (tst < hi)
            accs = [(frozen[mol][c].predict(Z[case][mh]) == y_te[TH][mol][mh]).mean() for mol in MOLS for c in MK]
            rows_host.append(dict(case=case, host=h, n=int(mh.sum()), accuracy=float(np.mean(accs))))
    host_df = pd.DataFrame(rows_host)

    # ---- absorb/detect trade-off: shift AUROC of Mahalanobis and error AUROC of the margin, frozen vs randomized
    dist_full = distance_scores(rand_Ztr["full"]); rows_tr = []
    d0c, d1c = dist_cache["clean"]["mahalanobis"], dist_full(Z["clean"])["mahalanobis"]
    for case in [k for k in Z if k != "clean"]:
        d0, d1 = dist_cache[case]["mahalanobis"], dist_full(Z[case])["mahalanobis"]
        lab = np.r_[np.zeros(len(d0c)), np.ones(len(d0))]
        e0, e1 = [], []
        for mol in MOLS:
            y = y_te[TH][mol]
            for c in MK:
                p0 = prob(frozen[mol][c], Z[case]); p1 = prob(rand_models["full"][mol][c], Z[case])
                e0.append(auroc((p0 >= .5) != y, 1 - np.abs(2 * p0 - 1))); e1.append(auroc((p1 >= .5) != y, 1 - np.abs(2 * p1 - 1)))
        rows_tr.append(dict(case=case, shift_auroc_frozen=auroc(lab, np.r_[d0c, d0]), shift_auroc_randomized=auroc(lab, np.r_[d1c, d1]),
                            error_auroc_frozen=np.nanmean(e0), error_auroc_randomized=np.nanmean(e1)))
    tr_df = pd.DataFrame(rows_tr)

    for nm, df in (("frozen", frozen_df), ("ceiling", ceiling_df), ("randomized", rand_df), ("detect", det_df), ("calibration", cal_df),
                   ("host", host_df), ("tradeoff", tr_df)):
        df.to_csv(os.path.join(RESULTS, f"alfnoor_trust_{nm}.csv"), index=False)
    pd.Series(fallbacks).to_csv(os.path.join(RESULTS, "alfnoor_trust_fallbacks.csv"))
    write_summary(frozen_df, ceiling_df, rand_df, det_df, cal_df, host_df, tr_df, len(Ptr), len(Pte), fallbacks)
    print(f"all done {time.time()-t_start:.0f} s", flush=True)


def write_summary(frozen_df, ceiling_df, rand_df, det_df, cal_df, host_df, tr_df, ntr, nte, fallbacks):
    fz = frozen_df[frozen_df.threshold == f"1e{TH}"]
    cases = [c for c in fz.columns if c not in ("threshold", "molecule", "classifier")]
    mean_mol = fz.groupby("molecule")[cases].mean() * 100; clean = mean_mol["clean"]
    L = [f"The consortium Tier-1 screen under the full trust procedure (train {ntr}, test {nte} planets; abundance > 1e-4;",
         "mean over KNN/MLP/RFC/SVC). Loss = points below the screen's own clean accuracy.", ""]
    L.append("== frozen accuracy and loss by molecule")
    L.append(f"{'case':<14}{'range':>6}" + "".join(f"{m:>16}" for m in MOLS) + f"{'mean loss':>11}")
    for case in cases:
        rng_tag = "" if case == "clean" else ("in" if IN_RANGE.get(case) else "OUT")
        cells = "".join(f"{mean_mol.loc[m, case]:8.1f} ({clean[m] - mean_mol.loc[m, case]:+5.1f})" for m in MOLS)
        L.append(f"{case:<14}{rng_tag:>6}{cells}{(clean - mean_mol[case]).mean():11.1f}")
    L += ["", "== ceilings: the same design retrained at the test condition (mean over molecules)"]
    L.append(f"{'case':<14}{'frozen':>8}{'ceiling':>9}{'loss':>7}{'irreducible':>12}{'reducible %':>12}")
    cm = ceiling_df.groupby("case").ceiling.mean() * 100; c0 = clean.mean()
    ceil_rows = {}
    for case in CEILING_TRAIN:
        fr = mean_mol[case].mean(); ce = cm[case]; loss = c0 - fr; irr = c0 - ce
        red = 100 * (ce - fr) / loss if loss > 0.5 else np.nan; ceil_rows[case] = (fr, ce, loss, irr, red)
        L.append(f"{case:<14}{fr:8.1f}{ce:9.1f}{loss:7.1f}{irr:12.1f}{red:12.0f}")
    L += ["", "== randomized grid (haze 60 %, spots 70 %, noise x1-3) and held-out variants (mean over molecules and classifiers)"]
    rm = rand_df.groupby("variant")[cases].mean() * 100
    L.append(f"{'case':<14}{'range':>6}{'frozen':>8}{'full':>8}{'no_haze':>9}{'no_spots':>9}{'no_noise':>9}{'ceiling':>9}")
    for case in cases:
        rng_tag = "" if case == "clean" else ("in" if IN_RANGE.get(case) else "OUT")
        ce = f"{cm[case]:9.1f}" if case in cm.index else f"{'-':>9}"
        L.append(f"{case:<14}{rng_tag:>6}{mean_mol[case].mean():8.1f}{rm.loc['full', case]:8.1f}{rm.loc['no_haze', case]:9.1f}{rm.loc['no_spots', case]:9.1f}{rm.loc['no_noise', case]:9.1f}{ce}")
    L.append(f"clean cost of randomization: {mean_mol['clean'].mean():.1f} -> {rm.loc['full', 'clean']:.1f}")
    shares = []
    for case in CEILING_TRAIN:
        if IN_RANGE.get(case):
            fr, ce, loss, _, _ = ceil_rows[case]; full = rm.loc["full", case]
            if ce - fr > 0.5: shares.append((case, 100 * (full - fr) / (ce - fr)))
    L.append("randomized share of the single-axis ceiling (in-range): " + ", ".join(f"{c} {s:.0f} %" for c, s in shares))
    held = {"haze": "no_haze", "spots": "no_spots", "white": "no_noise"}
    for ax, v in held.items():
        cs = [c for c in cases if AXIS.get(c) == ax and IN_RANGE.get(c)]
        fr = np.mean([mean_mol[c].mean() for c in cs]); full = np.mean([rm.loc["full", c] for c in cs]); ho = np.mean([rm.loc[v, c] for c in cs])
        tr = 100 * (ho - fr) / (full - fr) if full - fr > 0.5 else np.nan
        L.append(f"held-out {ax:<6}: frozen {fr:.1f}, without the ingredient {ho:.1f}, full {full:.1f}; transfer {tr:.0f} %  ({len(cs)} cases)")
    L += ["", "== decline rules on the frozen screen (thresholds decline 10 % of clean planets; mean over molecules and classifiers)"]
    dm = det_df.groupby(["rule", "case"])[["coverage", "accepted", "credit", "auroc_error", "auroc_shift"]].mean()
    for rule in ("margin", "ensemble", "mahalanobis", "knn", "pca_recon"):
        L.append(f"-- {rule}")
        for case in ["clean", "cloud1e3", "cloud1e2", "haze3e7", "haze2p4e8", "spots20", "compound", "white_x3", "corr_x3", "exomol", "exotransmit", "absorbers"]:
            r = dm.loc[(rule, case)]
            L.append(f"   {case:<12} keep {100*r.coverage:5.1f} %  accepted {100*r.accepted:5.1f} %  credit {100*r.credit:+6.1f}  AUROC(error) {r.auroc_error:.3f}  AUROC(shift) {r.auroc_shift:.3f}")
    L += ["", "== calibration and split-conformal coverage (90 % target; KNN/MLP/RFC; mean over molecules and classifiers)"]
    cmn = cal_df.groupby("case")[["accuracy", "ece", "coverage", "empty", "ambiguous"]].mean()
    for case in ["clean", "cloud1e2", "haze3e7", "haze2p4e8", "spots20", "compound", "white_x3", "corr_x3", "exomol", "exotransmit", "absorbers"]:
        r = cmn.loc[case]; L.append(f"   {case:<12} ECE {r.ece:.3f}  coverage {100*r.coverage:5.1f} %  empty {100*r.empty:4.1f} %  ambiguous {100*r.ambiguous:4.1f} %")
    L += ["", "== host dependence (accuracy; loss vs the same host's clean accuracy)"]
    hp = host_df.pivot(index="case", columns="host", values="accuracy") * 100
    for case in hp.index:
        L.append(f"   {case:<10}" + "".join(f"  {h} {hp.loc[case, h]:5.1f} ({hp.loc['clean', h]-hp.loc[case, h]:+5.1f})" for h, _, _ in HOSTS))
    L.append("   planets per host: " + ", ".join(f"{r.host} {r.n}" for r in host_df[host_df.case == "clean"].itertuples()))
    L += ["", "== absorb/detect trade-off: Mahalanobis shift AUROC and margin error AUROC, frozen -> randomized"]
    for r in tr_df.itertuples():
        L.append(f"   {r.case:<12} shift {r.shift_auroc_frozen:.3f} -> {r.shift_auroc_randomized:.3f}   error {r.error_auroc_frozen:.3f} -> {r.error_auroc_randomized:.3f}")
    comp = (clean - mean_mol["compound"]).mean() - ((clean - mean_mol["spots20"]).mean() + (clean - mean_mol["haze3e7"]).mean())
    L += ["", f"compound excess over the sum of its parts: {comp:+.1f} points (negative = sub-additive)",
          f"failed re-renders that fell back to clean: " + ", ".join(f"{k} {v}" for k, v in fallbacks.items())]
    open(os.path.join(RESULTS, "alfnoor_trust.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--render", action="store_true"); ap.add_argument("--fit", action="store_true")
    ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    if a.render: do_render(a.jobs)
    if a.fit: do_fit()


if __name__ == "__main__":
    main()
