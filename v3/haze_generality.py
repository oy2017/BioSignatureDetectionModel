"""Is the optical-photometry haze blindness general? Three checks (2026-09-13, after alfnoor_haze_mechanism.py).

1 carbonrich   The carbon-rich screen (norm_xgb, Tier-3 hyper-parameters) retrained with and without the three optical
               photometric points, at Tier-3 (102 bins) and Tier-1 (7 bins) binning; scored on the haze, spot, cloud and
               compound test sets (augment.shifted_test noise convention).
2 photnoise    The consortium design (alfnoor_trust.py) retrained with the photometric-point noise multiplied by 3 and 10
               (training and test), published inputs vs the optical points dropped.
3 particles    The consortium screen on POP-I re-rendered with haze particles of 0.03 and 0.5 um (0.1 um is the default),
               the density scaled so that the haze cross section at 0.35 um times the density equals that of the
               0.1 um, 3e7 m^-3 case (the same Rayleigh-enhancement level); published inputs vs optical dropped.

Usage: python haze_generality.py [--parts carbonrich photnoise particles] [--jobs 8]
Writes results/haze_generality.txt / .csv and data/alfnoor_faithful/pop1_native_haze_r{0p03,0p5}.npy
"""
import argparse, json, os, sys, time
import joblib, numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, TESTS, Features, centres, configs, load_split, metrics
from bin_spectra import bin_native

ROWS, LINES = [], []


def carbonrich():
    from pipeline import make_xgb
    from augment import shifted_test
    hp = joblib.load(os.path.join(MODELS, "ariel_norm_xgb.joblib"))["params"]
    cases = ["haze_2p0e6", "haze_3p0e7", "haze_2p4e8", "haze_1p0e10", "cloud_1e3Pa", "cloud_1e2Pa", "tlse_spots10", "tlse_spots20",
             "compound_spots20_haze3e7", "absorbers"]
    LINES.append("== 1 carbon-rich screen: all bins vs optical points dropped (accuracy %, loss vs own clean)")
    for cfg in ("ariel", "tier1"):
        cen = centres(cfg); keep = {"all bins": np.ones(len(cen), bool), "no optical": cen > 1.1}
        Xtr, ytr, _ = load_split("train", cfg)
        Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
        sets = {"clean": (Xc, yc)}
        for c in cases:
            if all(os.path.exists(os.path.join(DATA, f"{t}_native_{c}.npy")) for t in TESTS):
                sets[c] = shifted_test(c, cfg)
        for vname, cols in keep.items():
            f = Features("norm").fit(Xtr[:, cols]); m = make_xgb(hp).fit(f.transform(Xtr[:, cols]), ytr)
            acc = {c: metrics(y, m.predict_proba(f.transform(X[:, cols]))[:, 1])["accuracy"] * 100 for c, (X, y) in sets.items()}
            for c, a in acc.items():
                ROWS.append(dict(part="carbonrich", config=cfg, variant=vname, case=c, accuracy=a, loss=acc["clean"] - a))
            LINES.append(f"  {cfg:<6}{vname:<11}" + "  ".join(f"{c.replace('compound_spots20_haze3e7','compound')} {a:.1f} ({acc['clean']-a:+.1f})" for c, a in acc.items()))
        print("carbonrich", cfg, "done", flush=True)
    LINES.append("")


def _consortium_setup():
    import alfnoor_trust as AT, alfnoor_faithful as F
    wl = np.load(os.path.join(DATA, "native_wl.npy")); e = F.layout_edges("tier3_r20"); cen = 0.5 * (e[1:] + e[:-1])
    Ptr, Pte = AT._params(); load = lambda n: np.load(os.path.join(AT.OUT, f"{n}.npy")).astype(float)
    Xtr = load("pop3_native"); ok_tr = np.isfinite(Xtr).all(1); Xte = load("pop1_native"); ok = np.isfinite(Xte).all(1)
    Ptr, Pte = Ptr[ok_tr].reset_index(drop=True), Pte[ok].reset_index(drop=True)
    rng = np.random.default_rng(SEED + 405)
    sig_tr = F.sigma(Ptr, e, "radiometric"); eps_tr = rng.normal(0, 1, (len(Ptr), len(e) - 1))
    sig_te = F.sigma(Pte, e, "radiometric"); eps_te = rng.normal(0, 1, sig_te.shape)
    return dict(AT=AT, F=F, wl=wl, e=e, cen=cen, Ptr=Ptr, Pte=Pte, Xtr=Xtr[ok_tr], Xte=Xte[ok], ok=ok, load=load,
                sig_tr=sig_tr, eps_tr=eps_tr, sig_te=sig_te, eps_te=eps_te)


def _score(S, Xtr_noisy, tests, variants, tag, extra):
    AT = S["AT"]; mk = AT.makers(); out = {}
    for vname, cols in variants.items():
        zf = lambda X: (X[:, cols] - X[:, cols].mean(1, keepdims=True)) / (X[:, cols].std(1, keepdims=True) + 1e-12)
        Ztr = zf(Xtr_noisy); Z = {c: zf(X) for c, X in tests.items()}; acc = {c: [] for c in tests}; rec = {c: [] for c in tests}
        for mol in AT.MOLS:
            ytr = (S["Ptr"][f"atm {mol}"] > -4).astype(int).to_numpy(); y = (S["Pte"][f"atm {mol}"] > -4).astype(int).to_numpy()
            for cname, mkr in mk.items():
                m = mkr().fit(Ztr, ytr)
                for c, Zc in Z.items():
                    pr = m.predict(Zc); acc[c].append((pr == y).mean() * 100); rec[c].append((pr[y == 1] == 1).mean() * 100)
        out[vname] = {c: (np.mean(acc[c]), np.mean(rec[c])) for c in tests}
        for c in tests:
            ROWS.append(dict(part=tag, variant=vname, case=c, accuracy=out[vname][c][0], recall=out[vname][c][1],
                             loss=out[vname]["clean"][0] - out[vname][c][0], **extra))
    return out


def photnoise():
    S = _consortium_setup(); B = lambda X: bin_native(X, S["wl"], S["e"]); cen = S["cen"]
    import shift_tlse as T
    T.set_grid(S["wl"]); P = S["Pte"]; logg = np.log10(T.G_SUN * P["s mass"].to_numpy() / P["s radius"].to_numpy() ** 2 * 100)
    haze = S["load"]("pop1_native_haze3e7")[S["ok"]]; bad = ~np.isfinite(haze).all(1); haze[bad] = S["Xte"][bad]
    spots = S["Xte"] * T.contamination(P["s temperature"].to_numpy(), logg, 0.20, 0.0)
    LINES.append("== 2 consortium design with the photometric-point noise inflated (mean over 4 molecules x 4 classifiers; accuracy / recall, loss vs own clean)")
    for k in (1.0, 3.0, 10.0):
        fac = np.where(cen < 1.1, k, 1.0)[None, :]
        noisy = lambda Xb, sig, eps: Xb + eps * sig * fac
        Xtr_noisy = noisy(B(S["Xtr"]), S["sig_tr"], S["eps_tr"])
        tests = {c: noisy(B(X), S["sig_te"], S["eps_te"]) for c, X in (("clean", S["Xte"]), ("haze3e7", haze), ("spots20", spots))}
        res = _score(S, Xtr_noisy, tests, {"published": np.ones(len(cen), bool), "no optical": cen > 1.1}, "photnoise", dict(phot_noise_factor=k))
        for v, r in res.items():
            LINES.append(f"  phot noise x{k:<4g} {v:<11}" + "  ".join(f"{c} {a:.1f}/{rc:.1f} ({r['clean'][0]-a:+.1f})" for c, (a, rc) in r.items()))
        print("photnoise", k, "done", flush=True)
    LINES.append("")


def _render_haze(rows_P, wl, jobs, radius, density):
    import alfnoor_faithful as F

    def worker(rows):
        import warnings; warnings.filterwarnings("ignore")
        import generate_grid as G, shift_aerosol as A, alfnoor_screen as AS
        G.FILL_GAS = F.FILL; A.FILL_GAS = F.FILL; A.HAZE_RADIUS = radius
        return AS._w(rows, wl, None, {})
    rows = []
    for i in range(len(rows_P)):
        r = rows_P.iloc[i].to_dict(); r["_haze"] = density; rows.append(r)
    chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
    res = Parallel(n_jobs=jobs)(delayed(worker)(c) for c in chunks)
    X = np.full((len(rows), len(wl)), np.nan, np.float32)
    for i, s in enumerate(s for c in res for s in c):
        if s is not None: X[i] = s
    return X


def particles(jobs):
    import alfnoor_trust as AT
    S = _consortium_setup(); B = lambda X: bin_native(X, S["wl"], S["e"]); cen = S["cen"]
    Pte_all = pd.read_parquet(os.path.join(AT.OUT, "pop1_params.parquet"))
    def sig035(r):
        x = 2 * np.pi * r / 0.35; q = 5.0 / (40.0 * x ** -4 + x ** 0.2); return q * np.pi * (r * 1e-6) ** 2
    tests = {"clean": B(S["Xte"])}
    LINES.append("== 3 haze particle size at a fixed 0.35-um haze extinction (that of 0.1 um at 3e7 m^-3)")
    for r in (0.03, 0.1, 0.5):
        dens = 3e7 * sig035(0.1) / sig035(r)
        name = "pop1_native_haze3e7" if r == 0.1 else f"pop1_native_haze_r{str(r).replace('.', 'p')}"
        path = os.path.join(AT.OUT, f"{name}.npy")
        if not os.path.exists(path):
            t0 = time.time(); X = _render_haze(Pte_all, S["wl"], jobs, r, dens); np.save(path, X); print(f"rendered {name} density {dens:.3g} ({time.time()-t0:.0f} s)", flush=True)
        H = np.load(path).astype(float)[S["ok"]]; bad = ~np.isfinite(H).all(1); H[bad] = S["Xte"][bad]
        Hb = B(H); tests[f"r{r}"] = Hb
        d = (Hb - tests["clean"]) * 1e6; ir = cen > 1.95
        LINES.append(f"  radius {r} um, density {dens:.3g} m^-3: adds optical {np.median(d[:, cen < 1.1]):.0f} ppm, NIRSpec {np.median(d[:, (cen > 1.1) & (cen < 1.95)]):.0f}, AIRS {np.median(d[:, ir]):.0f}; "
                     f"AIRS peak-to-peak ratio {np.median(np.ptp(Hb[:, ir], 1) / np.ptp(tests['clean'][:, ir], 1)):.2f}; failures fell back to clean {int(bad.sum())}")
    noisy = {c: X + S["eps_te"] * S["sig_te"] for c, X in tests.items()}
    Xtr_noisy = B(S["Xtr"]) + S["eps_tr"] * S["sig_tr"]
    res = _score(S, Xtr_noisy, noisy, {"published": np.ones(len(cen), bool), "no optical": cen > 1.1, "AIRS only": cen > 1.95}, "particles", {})
    for v, rr in res.items():
        LINES.append(f"  {v:<11}" + "  ".join(f"{c} {a:.1f}/{rc:.1f} ({rr['clean'][0]-a:+.1f})" for c, (a, rc) in rr.items()))
    LINES.append("")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", nargs="+", default=["carbonrich", "photnoise", "particles"]); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if "carbonrich" in a.parts: carbonrich()
    if "photnoise" in a.parts: photnoise()
    if "particles" in a.parts: particles(a.jobs)
    tag = "_".join(a.parts)
    pd.DataFrame(ROWS).to_csv(os.path.join(RESULTS, f"haze_generality_{tag}.csv"), index=False)
    open(os.path.join(RESULTS, f"haze_generality_{tag}.txt"), "w").write("\n".join(LINES) + "\n"); print("\n".join(LINES))


if __name__ == "__main__":
    main()
