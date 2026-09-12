"""Tests 1 and 2 of the trust programme on the randomized training grid.

Training variants, all built per planet from the aligned renders written by randomize_train.py
(clean / aerosol, equilibrium / quenched), a spot factor and a per-planet noise level:
  full        every ingredient randomized                      (the candidate robust screen)
  no_spots    spots never shown       -> spots are the held-out axis
  no_quench   equilibrium only        -> quenching held out
  no_aero     no haze, no cloud       -> aerosols held out
  no_noise    SNR fixed at 15         -> noise level held out
Opacity tables and the Exo-Transmit code are never in any training set: held out by construction.

For every shifted test case: accuracy of the frozen clean-trained screen, of `full`, of the
variant that holds the case's axis out, and of the per-case oracle where oracle.py measured one.
Each case is tagged in-range / out-of-range relative to the randomized training draws.
Then the absorb/detect trade-off: AUROC of a Mahalanobis shift score and of the margin error
score, in the frozen model's and in `full`'s own feature space.

Usage: python trust_randomized.py --config ariel
Writes results/ariel_trust_randomized.txt / .csv, results/ariel_trust_tradeoff.csv
"""
import argparse, json, os, sys, time
import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features, centres, load_split, metrics  # noqa: E402
from noise import sigma_matrix  # noqa: E402
from pipeline import make_xgb  # noqa: E402
from augment import binned, shifted_test, corr_noise  # noqa: E402

# case, axis it belongs to, in-range for the randomized draws?
CASES = [("cloud_1e5Pa", "aero", True), ("cloud_1e4Pa", "aero", True), ("cloud_1e3Pa", "aero", True),
         ("cloud_1e2Pa", "aero", False), ("cloud_1e1Pa", "aero", False),
         ("haze_2p0e5", "aero", True), ("haze_2p0e6", "aero", True), ("haze_3p0e7", "aero", True),
         ("haze_2p4e8", "aero", True), ("haze_1p0e10", "aero", False),
         ("tlse_spots02", "spots", True), ("tlse_spots05", "spots", True), ("tlse_spots10", "spots", True),
         ("tlse_spots20", "spots", True), ("tlse_mixed", "spots", False), ("tlse_fac10", "spots", False),
         ("quenched", "quench", True), ("exotransmit", "code", False), ("exomol", "opacity", False),
         ("compound_spots10_haze3e7", "compound", True), ("compound_spots20_haze3e7", "compound", True),
         ("compound_spots20_haze3e7_snr8", "compound", True)]
NOISE = [("white", 12, True), ("white", 8, True), ("white", 5, True), ("correlated", 12, True), ("correlated", 8, True), ("correlated", 5, True)]
HELD_OUT = {"aero": "no_aero", "spots": "no_spots", "quench": "no_quench", "noise": "no_noise"}


def spot_factor(P, frac):
    import shift_tlse as T
    T.set_grid(np.load(os.path.join(DATA, "native_wl.npy")))
    Tst = P["s temperature"].to_numpy(); logg = np.log10(T.G_SUN * P["s mass"].to_numpy() / P["s radius"].to_numpy() ** 2 * 100)
    fac = np.ones((len(P), T._edges.size - 1))
    fr = np.round(frac, 2)
    for f in np.unique(fr):
        if f <= 0: continue
        m = fr == f; fac[m] = T.contamination(Tst[m], logg[m], float(f), 0.0)
    return fac


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel"); a = ap.parse_args(); cfg = a.config
    cen = centres(cfg)
    best = json.load(open(os.path.join(RESULTS, f"{cfg}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{cfg}_{best}.joblib")); ff, fm, params = fr["features"], fr["model"], fr["params"]; kind = best.split("_")[0]

    Ptr = pd.read_parquet(os.path.join(DATA, "train_params.parquet")); ytr = Ptr["label_co"].to_numpy()
    D = pd.read_parquet(os.path.join(DATA, "train_rand_draws.parquet"))
    nat = {k: np.load(os.path.join(DATA, f)).astype(np.float64) for k, f in
           (("clean_eq", "train_native.npy"), ("clean_q", "train_native_quenched.npy"),
            ("aero_eq", "train_native_rand_aero_eq.npy"), ("aero_q", "train_native_rand_aero_q.npy"))}
    for k in ("clean_q", "aero_eq", "aero_q"):                       # failed renders fall back to the clean equilibrium planet
        bad = ~np.all(np.isfinite(nat[k]), axis=1); nat[k][bad] = nat["clean_eq"][bad]
        if bad.any(): print(f"  {k}: {bad.sum()} failed rows replaced by clean", flush=True)
    has_aero = np.isfinite(D.haze_density.to_numpy()) | np.isfinite(D.cloud_pressure.to_numpy())
    quench = D.quenched.to_numpy(); fac = spot_factor(Ptr, D.spot_frac.to_numpy()); snr_draw = D.snr.to_numpy()
    tstar = Ptr["s temperature"].to_numpy()

    def variant(name):
        aero = has_aero if name != "no_aero" else np.zeros_like(has_aero)
        q = quench if name != "no_quench" else np.zeros_like(quench)
        key = np.where(aero, np.where(q, 2, 1), np.where(q, 3, 0))    # 0 clean_eq, 1 aero_eq, 2 aero_q, 3 clean_q
        X = np.empty_like(nat["clean_eq"])
        for i, k in enumerate(("clean_eq", "aero_eq", "aero_q", "clean_q")):
            X[key == i] = nat[k][key == i]
        if name != "no_spots":
            X = X * fac
        Xb = binned(X, cfg)
        snr = snr_draw if name != "no_noise" else np.full(len(Xb), SNR)
        sig = sigma_matrix(Xb, tstar, cen, 1.0, "ariel") / snr[:, None]
        rng = np.random.default_rng(SEED + 1000)
        Xn = Xb + rng.normal(0.0, 1.0, sig.shape) * sig
        t0 = time.time(); f = Features(kind).fit(Xn); m = make_xgb(params).fit(f.transform(Xn), ytr)
        print(f"  trained {name} ({time.time()-t0:.0f} s)", flush=True)
        return f, m, Xn

    models = {"frozen": (ff, fm, None)}
    for v in ("full", "no_spots", "no_quench", "no_aero", "no_noise"):
        models[v] = variant(v)

    # test sets
    Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
    Xc_nf = np.vstack([np.load(os.path.join(DATA, f"{t}_{cfg}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    sets = [("clean", "clean", True, Xc, yc)]
    for case, axis, inr in CASES:
        if all(os.path.exists(os.path.join(DATA, f"{t}_native_{case}.npy")) for t in TESTS):
            Xs, ys = shifted_test(case, cfg); sets.append((case, axis, inr, Xs, ys))
    rng = np.random.default_rng(SEED + 61)
    for kindn, s, inr in NOISE:
        sets.append((f"{kindn}_snr{s}", "noise", inr, corr_noise(Xc, rng, s, kind=kindn, Xnf=Xc_nf, params=Pc, cen=cen), yc))

    orc = pd.read_csv(os.path.join(RESULTS, f"{cfg}_oracle.csv")) if os.path.exists(os.path.join(RESULTS, f"{cfg}_oracle.csv")) else None
    def oracle_for(case):
        if orc is None: return np.nan
        key = case.replace("white_", "").replace("correlated_", "") if case.startswith(("white_", "correlated_")) else case
        axis = "white noise" if case.startswith("white_") else "correlated noise" if case.startswith("correlated_") else None
        g = orc[(orc.case == key) & ((orc.axis == axis) if axis else True)]
        return float(g.oracle.iloc[0]) if len(g) else np.nan

    acc = lambda f, m, X, y: metrics(y, m.predict_proba(f.transform(X))[:, 1])["accuracy"]
    rows = []
    for name, axis, inr, X, y in sets:
        r = dict(case=name, axis=axis, in_range=inr, oracle=oracle_for(name))
        for v, (f, m, _) in models.items():
            r[v] = acc(f, m, X, y)
        r["held_out"] = r[HELD_OUT[axis]] if axis in HELD_OUT else (r["full"] if axis in ("code", "opacity") else np.nan)
        rows.append(r)
        print(f"{name:<30}{'in ' if inr else 'OUT'}  frozen {r['frozen']*100:6.2f}  full {r['full']*100:6.2f}  held-out {r['held_out']*100 if np.isfinite(r['held_out']) else float('nan'):6.2f}"
              f"  oracle {r['oracle']*100 if np.isfinite(r['oracle']) else float('nan'):6.2f}", flush=True)
    df = pd.DataFrame(rows)

    # trade-off: does randomizing an axis into training blind the detector to it?
    Ztr_f = ff.transform(load_split("train", cfg)[0]); Ztr_r = models["full"][0].transform(models["full"][2])
    def maha(Z, Ztr):
        mu = Ztr.mean(axis=0); Ci = np.linalg.inv(np.cov(Ztr, rowvar=False) + 1e-3 * np.eye(Ztr.shape[1])); d = Z - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d))
    base = {}
    for tag, (f, m, Ztr) in (("frozen", (ff, fm, Ztr_f)), ("full", (models["full"][0], models["full"][1], Ztr_r))):
        base[tag] = maha(f.transform(Xc), Ztr)
    trows = []
    for name, axis, inr, X, y in sets:
        if name == "clean": continue
        t = dict(case=name, axis=axis)
        for tag, (f, m, Ztr) in (("frozen", (ff, fm, Ztr_f)), ("full", (models["full"][0], models["full"][1], Ztr_r))):
            Z = f.transform(X); p = m.predict_proba(Z)[:, 1]; wrong = ((p >= 0.5).astype(int) != y)
            s_shift = maha(Z, Ztr); s_err = 1 - np.abs(2 * p - 1)
            t[f"{tag}_auroc_shift"] = roc_auc_score(np.r_[np.zeros(len(base[tag])), np.ones(len(s_shift))], np.r_[base[tag], s_shift])
            t[f"{tag}_auroc_error"] = roc_auc_score(wrong, s_err) if 0 < wrong.sum() < len(wrong) else np.nan
        trows.append(t)
    tdf = pd.DataFrame(trows)

    L = [f"Randomized training grid vs the frozen screen, configuration {cfg}, pipeline {best}", "",
         "full = every ingredient randomized; held-out = the variant trained WITHOUT the case's axis",
         "(for the opacity and code axes every variant is held out; 'full' is shown). oracle = trained at the",
         "test strength (oracle.py), where measured. in/OUT = test strength inside the randomized training range.", "",
         f"{'case':<30}{'range':>6}{'frozen':>9}{'full':>9}{'held-out':>10}{'oracle':>9}"]
    for _, r in df.iterrows():
        ho = f"{r.held_out*100:9.2f}" if np.isfinite(r.held_out) else f"{'-':>9}"
        oc = f"{r.oracle*100:8.2f}" if np.isfinite(r.oracle) else f"{'-':>8}"
        L.append(f"{r.case:<30}{'in' if r.in_range else 'OUT':>6}{r.frozen*100:9.2f}{r.full*100:9.2f}{ho:>10}{oc:>9}")
    c = df[df.case == "clean"].iloc[0]
    L += ["", f"clean cost of randomization: {c.frozen*100:.2f}% -> {c.full*100:.2f}% ({(c.full-c.frozen)*100:+.2f} points)"]
    inr = df[(df.case != "clean") & df.in_range & np.isfinite(df.oracle)]
    if len(inr):
        frac = ((inr.full - inr.frozen) / (inr.oracle - inr.frozen).where((inr.oracle - inr.frozen) > 0.005)).dropna() * 100
        L.append(f"in-range cases with an oracle: full reaches {frac.min():.0f}-{frac.max():.0f}% of the ceiling (n = {len(frac)})")
    for axis, v in HELD_OUT.items():
        g = df[(df.axis == axis) & (df.case != "clean")]
        if len(g):
            L.append(f"held-out {axis:<7}: frozen loss {((c.frozen - g.frozen)*100).mean():+.2f} -> held-out variant loss {((c.frozen - g[v])*100).mean():+.2f} (mean over {len(g)} cases)"
                     f" ; transfer = {100*((g[v]-g.frozen)/(g.full-g.frozen).where((g.full-g.frozen).abs()>0.002)).dropna().mean():.0f}% of what full achieves")
    L += ["", "Absorb/detect trade-off (Mahalanobis shift AUROC; margin error AUROC), frozen -> full:"]
    for _, t in tdf.iterrows():
        L.append(f"  {t.case:<30} shift {t.frozen_auroc_shift:.3f} -> {t.full_auroc_shift:.3f}   error {t.frozen_auroc_error:.3f} -> {t.full_auroc_error:.3f}")
    open(os.path.join(RESULTS, f"{cfg}_trust_randomized.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, f"{cfg}_trust_randomized.csv"), index=False); tdf.to_csv(os.path.join(RESULTS, f"{cfg}_trust_tradeoff.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
