"""Oracle ceilings for the repair tests: how much of each loss is recoverable at all?

The recovered fraction in augment*.py is measured against the frozen pipeline's clean
accuracy. That ceiling is wrong for a shift that destroys information: no training can
get the clean accuracy back on SNR-5 spectra. Without a ceiling, "recovers 45 %" could
mean "recovers everything that is recoverable", and the draw-count rule would collapse
into "noise destroys information and physics does not".

The oracle is a pipeline trained ONLY at the test strength (every training planet
shifted exactly as the test planets are), with the frozen pipeline's hyper-parameters,
features and noise seed. Its accuracy on the shifted test set is the practical ceiling;
the recovered fraction is then re-expressed against it:

    recovered_vs_oracle = (augmented - frozen) / (oracle - frozen)

If the deterministic and stochastic axes still separate on that measure, the rule is
about what augmentation at mixed strengths can absorb, not about irreducible loss.

Needs the augment*.py CSVs (frozen and augmented accuracies on the identical test arrays).
Usage: python oracle.py [--jobs 4] [--skip-haze]
Writes results/ariel_oracle.txt / .csv; caches data/train_native_oracle_haze_3p0e7.npy
"""
import argparse, json, os, sys, time
import joblib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features, centres, load_split, metrics  # noqa: E402
from noise import add_noise  # noqa: E402
from pipeline import make_xgb  # noqa: E402
from augment import binned, shifted_test, corr_noise  # noqa: E402
from augment_ramp import ramp, TEST_STRENGTHS  # noqa: E402

CFG = "ariel"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--skip-haze", action="store_true")
    a = ap.parse_args()
    cen = centres(CFG)
    best = json.load(open(os.path.join(RESULTS, f"{CFG}_best.json")))["best"]
    fr = joblib.load(os.path.join(MODELS, f"{CFG}_{best}.joblib"))
    ff, fm, params = fr["features"], fr["model"], fr["params"]; kind = best.split("_")[0]

    Xtr, ytr, Ptr = load_split("train", CFG)                       # noisy clean training set, the frozen pipeline's own
    Xtr_nf = np.load(os.path.join(DATA, f"train_{CFG}.npy")).astype(float)
    Xtr_native = np.load(os.path.join(DATA, "train_native.npy")).astype(np.float64)
    wl = np.load(os.path.join(DATA, "native_wl.npy"))
    Xc = np.vstack([load_split(t, CFG)[0] for t in TESTS]); yc = np.concatenate([load_split(t, CFG)[1] for t in TESTS])
    Xc_nf = np.vstack([np.load(os.path.join(DATA, f"{t}_{CFG}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, CFG, noisy=False)[2] for t in TESTS], ignore_index=True)
    clean = metrics(yc, fm.predict_proba(ff.transform(Xc))[:, 1])["accuracy"]

    # frozen / augmented numbers already measured on these exact test arrays
    prev = pd.concat([pd.read_csv(os.path.join(RESULTS, f"{CFG}_{f}.csv")) for f in ("augment", "augment_white", "augment_ramp")],
                     ignore_index=True)
    prev["key"] = prev.axis + "|" + prev.case.astype(str)
    prev = prev.set_index("key")

    def fit_on_noisy(Xn):
        f = Features(kind).fit(Xn); return f, make_xgb(params).fit(f.transform(Xn), ytr)

    def fit_on_native(Xnat):
        Xn, _ = add_noise(binned(Xnat, CFG), Ptr, cen, snr=SNR, shape="ariel", seed=1000)   # same seed as augment.py
        return fit_on_noisy(Xn)

    rows = []
    def record(axis, case, kind_, Xs, f_o, m_o):
        k = f"{axis}|{case}"
        a_fr, a_au = float(prev.loc[k, "frozen"]), float(prev.loc[k, "augmented"])
        a_or = metrics(yc, m_o.predict_proba(f_o.transform(Xs))[:, 1])["accuracy"]
        # sanity: the frozen pipeline on the array rebuilt here must reproduce the CSV to the 4th decimal
        a_chk = metrics(yc, fm.predict_proba(ff.transform(Xs))[:, 1])["accuracy"]
        assert abs(a_chk - a_fr) < 5e-4, f"{k}: test array not reproduced ({a_chk:.4f} vs {a_fr:.4f})"
        gap_c, gap_o = clean - a_fr, a_or - a_fr
        r = dict(axis=axis, case=case, kind=kind_, clean=clean, frozen=a_fr, augmented=a_au, oracle=a_or,
                 pct_vs_clean=(a_au - a_fr) / gap_c * 100 if gap_c > 0.005 else np.nan,
                 pct_vs_oracle=(a_au - a_fr) / gap_o * 100 if gap_o > 0.005 else np.nan,
                 irreducible=(clean - a_or) * 100)
        rows.append(r)
        print(f"{axis:<18}{case:<14}{kind_:<14}frozen {a_fr*100:6.2f}  aug {a_au*100:6.2f}  oracle {a_or*100:6.2f}"
              f"  vs-clean {r['pct_vs_clean']:5.0f}%  vs-oracle {r['pct_vs_oracle']:5.0f}%  irreducible {r['irreducible']:+5.2f}", flush=True)

    # --- stochastic: white and correlated noise at SNR 8 and 5; test arrays rebuilt with the scripts' own seeds
    rng_w = np.random.default_rng(SEED + 8); Xw = {}
    for s in (12, 10, 8, 5):
        Xw[s] = corr_noise(Xc, rng_w, s, kind="white", Xnf=Xc_nf, params=Pc, cen=cen)
    rng_c = np.random.default_rng(SEED + 3); Xcorr = {}
    for s in (12, 10, 8, 5):
        Xcorr[s] = corr_noise(Xc, rng_c, s, Xnf=Xc_nf, params=Pc, cen=cen)
    rng_tr = np.random.default_rng(SEED + 31)
    for s in (8, 5):
        f_o, m_o = fit_on_noisy(corr_noise(Xtr, rng_tr, s, kind="white", Xnf=Xtr_nf, params=Ptr, cen=cen))
        record("white noise", f"snr{s}", "stochastic", Xw[s], f_o, m_o)
        f_o, m_o = fit_on_noisy(corr_noise(Xtr, rng_tr, s, Xnf=Xtr_nf, params=Ptr, cen=cen))
        record("correlated noise", f"snr{s}", "stochastic", Xcorr[s], f_o, m_o)

    # --- stochastic, augmentation whose training range INCLUDES the test strength: the published
    # augmentation trains at SNR {12, 10, 8} and is tested at SNR 5, so its SNR-5 row is an
    # extrapolation; the deterministic axes are always tested inside their training range.
    # Same protocol (a quarter of planets per level, one level per planet), levels extended to 5.
    for kind_noise, Xtest in (("white", Xw[5]), ("correlated", Xcorr[5])):
        rng_a = np.random.default_rng(SEED + 41)
        lev = rng_a.choice([15, 12, 10, 8, 5], size=len(ytr)); Xa = Xtr.copy()
        for s in (12, 10, 8, 5):
            m = lev == s
            Xa[m] = corr_noise(Xtr[m], rng_a, s, kind=kind_noise, Xnf=Xtr_nf[m], params=Ptr[m].reset_index(drop=True), cen=cen)
        f_a, m_a = fit_on_noisy(Xa)
        a_au = metrics(yc, m_a.predict_proba(f_a.transform(Xtest))[:, 1])["accuracy"]
        axis = f"{kind_noise} noise"; k = f"{axis}|snr5"
        r = next(r for r in rows if r["axis"] == axis and r["case"] == "snr5")
        a_fr, a_or = r["frozen"], r["oracle"]
        r2 = dict(axis=axis, case="snr5 (aug incl. SNR 5)", kind="stochastic", clean=clean, frozen=a_fr, augmented=a_au, oracle=a_or,
                  pct_vs_clean=(a_au - a_fr) / (clean - a_fr) * 100, pct_vs_oracle=(a_au - a_fr) / (a_or - a_fr) * 100,
                  irreducible=(clean - a_or) * 100)
        rows.append(r2)
        print(f"{axis:<18}{'snr5 in-range':<14}{'stochastic':<14}frozen {a_fr*100:6.2f}  aug {a_au*100:6.2f}  oracle {a_or*100:6.2f}"
              f"  vs-clean {r2['pct_vs_clean']:5.0f}%  vs-oracle {r2['pct_vs_oracle']:5.0f}%  irreducible {r2['irreducible']:+5.2f}", flush=True)

    # --- deterministic, injected: gain ramp x1 and x2
    rng_r = np.random.default_rng(SEED + 12); Xr = {}
    for s in TEST_STRENGTHS:
        Xr[s] = ramp(Xc, Xc_nf, Pc, cen, s, rng_r)
    for s in (1.0, 2.0):
        f_o, m_o = fit_on_noisy(ramp(Xtr, Xtr_nf, Ptr, cen, s, rng_tr))
        record("gain ramp", f"x{s}", "deterministic", Xr[s], f_o, m_o)

    # --- deterministic, re-rendered: spots 10 % and 20 % (multiplicative, cheap)
    import shift_tlse as T
    T.set_grid(wl)
    Tst = Ptr["s temperature"].to_numpy()
    logg = np.log10(T.G_SUN * Ptr["s mass"].to_numpy() / Ptr["s radius"].to_numpy() ** 2 * 100)
    for frac, case in ((0.10, "tlse_spots10"), (0.20, "tlse_spots20")):
        f_o, m_o = fit_on_native(Xtr_native * T.contamination(Tst, logg, frac, 0.0))
        Xs, ys = shifted_test(case, CFG); assert (ys == yc).all()
        record("stellar spots", case, "deterministic", Xs, f_o, m_o)

    # --- deterministic, re-rendered: haze 3e7 on every training planet (the expensive one)
    if not a.skip_haze:
        p = os.path.join(DATA, "train_native_oracle_haze_3p0e7.npy")
        if not os.path.exists(p):
            import shift_aerosol as A
            t0 = time.time(); Y = A.render(Ptr, wl, jobs=a.jobs, haze_density=3e7)
            print(f"  haze render: {len(Ptr)} planets, {int((~np.all(np.isfinite(Y), axis=1)).sum())} failures, {time.time()-t0:.0f} s", flush=True)
            np.save(p, Y)
        Y = np.load(p).astype(np.float64); ok = np.all(np.isfinite(Y), axis=1)
        Y = np.where(ok[:, None], Y, Xtr_native)
        f_o, m_o = fit_on_native(Y)
        Xs, ys = shifted_test("haze_3p0e7", CFG)
        record("haze", "haze_3p0e7", "deterministic", Xs, f_o, m_o)

    # --- deterministic, re-rendered with another code / other opacity tables, when the training
    # re-render exists (shift_exotransmit.py --splits train; shift_opacity.py --splits train)
    for case, fname, axis in (("exotransmit", "train_native_exotransmit.npy", "exotransmit"),
                              ("exomol", "train_native_exomol.npy", "exomol"),
                              ("absorbers", "train_native_absorbers.npy", "absorbers")):
        p_tr = os.path.join(DATA, fname)
        if not os.path.exists(p_tr):
            print(f"  {case}: no training re-render on disk, oracle skipped", flush=True); continue
        Y = np.load(p_tr).astype(np.float64); ok = np.all(np.isfinite(Y), axis=1)
        Y = np.where(ok[:, None], Y, Xtr_native)
        f_o, m_o = fit_on_native(Y)
        Xs, ys = shifted_test(case, CFG)
        a_or = metrics(yc, m_o.predict_proba(f_o.transform(Xs))[:, 1])["accuracy"]
        a_fr = metrics(yc, fm.predict_proba(ff.transform(Xs))[:, 1])["accuracy"]
        # the single-axis augmentation for these axes is the oracle itself (no strength to mix), so the
        # row reports the ceiling and the irreducible part; 'augmented' is left as the oracle
        rows.append(dict(axis=axis, case=case, kind="deterministic", clean=clean, frozen=a_fr, augmented=a_or, oracle=a_or,
                         pct_vs_clean=(a_or - a_fr) / (clean - a_fr) * 100 if clean - a_fr > 0.005 else np.nan,
                         pct_vs_oracle=np.nan, irreducible=(clean - a_or) * 100))
        print(f"{axis:<18}{case:<14}{'deterministic':<14}frozen {a_fr*100:6.2f}  oracle {a_or*100:6.2f}  irreducible {(clean-a_or)*100:+5.2f}"
              f"  (no mixed augmentation for this axis: the oracle IS the augmentation)", flush=True)

    df = pd.DataFrame(rows)
    det = df[df.kind == "deterministic"].pct_vs_oracle.dropna(); sto = df[df.kind == "stochastic"].pct_vs_oracle.dropna()
    L = ["Oracle ceilings for the repair tests, configuration ariel, pipeline " + best, "",
         "oracle = trained only at the test strength; irreducible = clean minus oracle (points the shift",
         "removes for any training); vs-oracle = share of the recoverable loss that mixed-strength augmentation got.", "",
         f"{'axis':<18}{'case':<14}{'kind':<14}{'frozen':>8}{'augmented':>11}{'oracle':>8}{'irreducible':>13}{'vs clean':>10}{'vs oracle':>11}"]
    for r in rows:
        L.append(f"{r['axis']:<18}{r['case']:<14}{r['kind']:<14}{r['frozen']*100:7.2f}%{r['augmented']*100:10.2f}%{r['oracle']*100:7.2f}%"
                 f"{r['irreducible']:+12.2f}{r['pct_vs_clean']:9.0f}%{r['pct_vs_oracle']:10.0f}%")
    L += ["", f"deterministic, vs oracle: {det.min():.0f}-{det.max():.0f}%   stochastic, vs oracle: {sto.min():.0f}-{sto.max():.0f}%"
          + ("   (non-overlapping)" if sto.max() < det.min() else "   (OVERLAPPING: the rule does not survive the ceiling correction)"),
          f"irreducible loss, deterministic: {df[df.kind=='deterministic'].irreducible.min():+.2f} to {df[df.kind=='deterministic'].irreducible.max():+.2f} pts;"
          f" stochastic: {df[df.kind=='stochastic'].irreducible.min():+.2f} to {df[df.kind=='stochastic'].irreducible.max():+.2f} pts"]
    open(os.path.join(RESULTS, f"{CFG}_oracle.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, f"{CFG}_oracle.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
