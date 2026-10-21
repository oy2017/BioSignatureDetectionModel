"""Retraining on a departure-aware training population: are the haze errors recoverable on the real planets?
(REVISION_PLAN.md E4 on real data; the reviewer calls haze retraining "familiar domain randomization", so this is
a test of the error class, not a method.)

Stage 1 (--render): a third of the faithful POP-III training planets get a Lee-Mie haze with density drawn
log-uniform between 1e6 and 3e8 m^-3 (the departure tests use 3e7), and a further sixth get a grey deck drawn
log-uniform between 5 and 500 Pa, two decades above the population prior's 5e2 Pa top (F7: the real warm giants
are more muted than the prior allows). Two training sets: 'haze' (haze rows only) and 'hazedeck' (both).
Writes data/alfnoor_faithful/pop3_native_aug_{haze,hazedeck}.npy and pop3_aug_draws.csv.

Stage 2 (--fit): the four learners retrained on each set, same binning, noise draw and normalization as
alfnoor_faithful.do_fit, scored at the 1e-4 threshold on POP-I clean and under every departure, for the three
input variants of alfnoor_jwst (all / no_optical / airs_only). Writes results/alfnoor_retrain_pop1.{csv,txt}.

Stage 3 (--panel): jwst_panel.py rerun with ALFNOOR_TRAIN=haze and =hazedeck (results/jwst_panel_<set>/), then
the verdicts of the three training sets side by side. Writes results/alfnoor_retrain_panel.{csv,txt}.

Usage:
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python alfnoor_retrain.py --render --jobs 2
  python alfnoor_retrain.py --fit
  python alfnoor_retrain.py --panel
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as AF  # noqa: E402
import alfnoor_screen as AS  # noqa: E402

SRC = AF.DIRS["faithful"]; MOLS = AS.MOLS
SETS = ["faithful", "haze", "hazedeck"]
HAZE_FRAC, DECK_FRAC = 1 / 3, 1 / 6
HAZE_LOG = (6.0, np.log10(3e8)); DECK_LOG = (np.log10(5.0), np.log10(500.0))
VARIANTS = {"all": lambda cen: np.ones_like(cen, bool), "no_optical": lambda cen: cen > 1.1, "airs_only": lambda cen: cen > 1.95}
CASES = ["clean", "haze3e7", "cloud1e2", "absorbers", "spots10", "spots20", "noise_x2", "noise_x3"]


def natives(name):
    return np.load(os.path.join(SRC, "pop3_native.npy" if name == "faithful" else f"pop3_native_aug_{name}.npy")).astype(np.float64)


def do_render(jobs):
    wl = np.load(os.path.join(DATA, "native_wl.npy")); P = pd.read_parquet(os.path.join(SRC, "pop3_params.parquet"))
    X = np.load(os.path.join(SRC, "pop3_native.npy")).astype(np.float32)
    rng = np.random.default_rng(SEED + 11); u = rng.uniform(size=len(P))
    haze_m = u < HAZE_FRAC; deck_m = (u >= HAZE_FRAC) & (u < HAZE_FRAC + DECK_FRAC)
    draws = pd.DataFrame(dict(name=P.name, group=np.where(haze_m, "haze", np.where(deck_m, "deck", "clean")), haze_density=np.nan, deck_pa=np.nan, rendered=False))
    draws.loc[haze_m, "haze_density"] = 10.0 ** rng.uniform(*HAZE_LOG, int(haze_m.sum()))
    draws.loc[deck_m, "deck_pa"] = 10.0 ** rng.uniform(*DECK_LOG, int(deck_m.sum()))
    rows = []
    for i in np.where(haze_m | deck_m)[0]:
        r = P.iloc[i].to_dict()
        if haze_m[i]:
            r["_haze"] = float(draws.haze_density.iloc[i])
        else:
            r["cloud_pressure"] = float(draws.deck_pa.iloc[i])
        rows.append((int(i), r))
    chunks = [rows[k:k + 25] for k in range(0, len(rows), 25)]
    t0 = time.time(); print(f"rendering {len(rows)} of {len(P)} training planets ({int(haze_m.sum())} hazed, {int(deck_m.sum())} deep decks) on {jobs} jobs", flush=True)
    res = Parallel(n_jobs=jobs)(delayed(AF._worker)([r for _, r in c], wl, None, {}) for c in chunks)
    Y = X.copy(); n_ok = 0
    for (i, _), s in zip((x for c in chunks for x in c), (s for c in res for s in c)):
        if s is not None and np.all(np.isfinite(s)):
            Y[i] = s; draws.loc[i, "rendered"] = True; n_ok += 1
    print(f"  {n_ok} rendered, {len(rows) - n_ok} failures kept their clean spectrum, {time.time() - t0:.0f} s", flush=True)
    Yh = X.copy(); Yh[haze_m] = Y[haze_m]
    np.save(os.path.join(SRC, "pop3_native_aug_haze.npy"), Yh); np.save(os.path.join(SRC, "pop3_native_aug_hazedeck.npy"), Y)
    draws.to_csv(os.path.join(SRC, "pop3_aug_draws.csv"), index=False)


def pop1_cases():
    """Test cases and per-planet sigma exactly as alfnoor_faithful.do_fit builds them (natives faithful, tier3_r20, radiometric)."""
    import shift_tlse as T
    wl = np.load(os.path.join(DATA, "native_wl.npy")); edges = AF.layout_edges("tier3_r20")
    Pte = pd.read_parquet(os.path.join(SRC, "pop1_params.parquet")); load = lambda n: np.load(os.path.join(SRC, f"{n}.npy")).astype(np.float64)
    Xte_n = load("pop1_native"); ok = np.all(np.isfinite(Xte_n), axis=1); cases = {"clean": Xte_n}
    for name in ("haze3e7", "cloud1e2", "absorbers"):
        Y = load(f"pop1_native_{name}"); ok &= np.all(np.isfinite(Y), axis=1); cases[name] = Y
    Pte = Pte[ok].reset_index(drop=True); cases = {k: v[ok] for k, v in cases.items()}
    T.set_grid(wl); Tst = Pte["s temperature"].to_numpy(); logg = np.log10(T.G_SUN * Pte["s mass"].to_numpy() / Pte["s radius"].to_numpy() ** 2 * 100)
    for f in (0.10, 0.20):
        cases[f"spots{int(f * 100)}"] = cases["clean"] * T.contamination(Tst, logg, f, 0.0)
    binned = {k: bin_native(v, wl, edges) for k, v in cases.items()}
    return Pte, binned, AF.sigma(Pte, edges, "radiometric"), wl, edges


def do_fit():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    makers = {"KNN": lambda: KNeighborsClassifier(), "MLP": lambda: MLPClassifier(random_state=SEED),
              "RFC": lambda: RandomForestClassifier(random_state=SEED, n_jobs=4), "SVC": lambda: SVC()}
    Pte, binned, sig, wl, edges = pop1_cases(); cen = 0.5 * (edges[1:] + edges[:-1])
    Ptr = pd.read_parquet(os.path.join(SRC, "pop3_params.parquet")); X0 = natives("faithful"); ok_tr = np.all(np.isfinite(X0), axis=1); Ptr = Ptr[ok_tr].reset_index(drop=True)
    sig_tr = AF.sigma(Ptr, edges, "radiometric")
    # the same noise draws as do_fit / alfnoor_jwst.training_set: the training noise first, then the test noise
    rng = np.random.default_rng(SEED + 405); eps_tr = rng.normal(0, 1, (int(ok_tr.sum()), len(cen))); eps = rng.normal(0, 1, sig.shape)
    noisy = {k: v + eps * sig for k, v in binned.items()}; noisy["noise_x2"] = binned["clean"] + eps * 2 * sig; noisy["noise_x3"] = binned["clean"] + eps * 3 * sig
    znorm = lambda X, cols: (X[:, cols] - X[:, cols].mean(1, keepdims=True)) / (X[:, cols].std(1, keepdims=True) + 1e-12)
    rows = []
    for ts in SETS:
        Xtr = bin_native(natives(ts)[ok_tr], wl, edges) + eps_tr * sig_tr
        for v, mask in VARIANTS.items():
            cols = mask(cen); Ztr = znorm(Xtr, cols); Z = {k: znorm(x, cols) for k, x in noisy.items()}
            for mol in MOLS:
                ytr = (Ptr[f"atm {mol}"] > -4).astype(int).to_numpy(); yte = (Pte[f"atm {mol}"] > -4).astype(int).to_numpy()
                for mname, mk in makers.items():
                    m = mk().fit(Ztr, ytr)
                    for c in CASES:
                        pred = m.predict(Z[c])
                        rows.append(dict(train=ts, variant=v, molecule=mol, learner=mname, case=c, accuracy=float((pred == yte).mean()),
                                         recall=float(pred[yte == 1].mean()), fpr=float(pred[yte == 0].mean()), n=len(yte)))
            print(f"  {ts} {v} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "alfnoor_retrain_pop1.csv"), index=False)
    L = [f"Retraining on departure-aware POP-III (alfnoor_retrain.py): accuracy / recall of present at the 1e-4 threshold on POP-I ({len(Pte)} planets), mean over the four learners",
         "training sets: faithful = the consortium population as rebuilt; haze = a third of its planets hazed (1e6-3e8 m^-3); hazedeck = plus a sixth with 5-500 Pa decks", ""]
    for v in VARIANTS:
        L.append(f"== inputs: {v}"); L.append(f"  {'mol':<4} {'train':<9} " + " ".join(f"{c:>14}" for c in CASES))
        for mol in MOLS:
            for ts in SETS:
                g = df[(df.variant == v) & (df.molecule == mol) & (df.train == ts)].groupby("case")[["accuracy", "recall"]].mean()
                L.append(f"  {mol:<4} {ts:<9} " + " ".join(f"{100*g.loc[c,'accuracy']:6.1f} / {100*g.loc[c,'recall']:4.0f}" for c in CASES))
        L.append("")
    L.append("per-learner rows in results/alfnoor_retrain_pop1.csv (accuracy, recall, false-positive rate)")
    open(os.path.join(RESULTS, "alfnoor_retrain_pop1.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def do_panel(draws):
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    for ts in SETS[1:]:
        out = os.path.join(RESULTS, f"jwst_panel_{ts}", "panel_summary.csv")
        if os.path.exists(out):
            print(f"  {ts}: panel exists"); continue
        print(f"  running jwst_panel.py with ALFNOOR_TRAIN={ts}", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "jwst_panel.py"), "--draws", str(draws)], env=dict(env, ALFNOOR_TRAIN=ts), check=True, cwd=HERE)
    S = {ts: pd.read_csv(os.path.join(RESULTS, "jwst_panel" + ("" if ts == "faithful" else f"_{ts}"), "panel_summary.csv")) for ts in SETS}
    key = ["planet", "molecule", "variant"]; df = S["faithful"][key + ["label", "full_coverage", "band_metric"]].copy()
    for ts in SETS:
        s = S[ts].set_index(key); df[f"verdict_{ts}"] = s.loc[list(map(tuple, df[key].to_numpy())), "verdict"].to_numpy()
        df[f"learners_present_{ts}"] = s.loc[list(map(tuple, df[key].to_numpy())), "learners_present"].to_numpy()
        df[f"tier1_frequency_{ts}"] = s.loc[list(map(tuple, df[key].to_numpy())), "tier1_frequency"].to_numpy()
        df[f"correct_{ts}"] = [None if (lab is None or (isinstance(lab, float) and np.isnan(lab)) or vd == "split") else vd == lab for lab, vd in zip(df.label, df[f"verdict_{ts}"])]
    df.to_csv(os.path.join(RESULTS, "alfnoor_retrain_panel.csv"), index=False)
    L = ["Real JWST panel under the three training sets (alfnoor_retrain.py --panel): verdict = majority of the four learners at 1e-4", ""]
    for v in VARIANTS:
        g = df[df.variant == v]; L.append(f"== inputs: {v}")
        for ts in SETS:
            c = g[f"correct_{ts}"].dropna(); L.append(f"  {ts:<9} {int(c.sum())}/{len(c)} scored labels correct")
        L.append(f"  {'planet':<12} {'mol':<4} {'label':<8} " + " ".join(f"{ts:>14}" for ts in SETS) + "   (learners present of 4, Tier-1 frequency %)")
        for _, r in g.iterrows():
            L.append(f"  {r.planet:<12} {r.molecule:<4} {str(r.label):<8} " + " ".join(f"{r[f'verdict_{ts}']:<7} {int(r[f'learners_present_{ts}'])} {100*r[f'tier1_frequency_{ts}']:3.0f}%" for ts in SETS))
        L.append("")
    ch4 = df[(df.molecule == "CH4") & (df.label == "present")]
    L.append("CH4-present planets, verdict under all inputs / AIRS-only, by training set:")
    for pl in ch4.planet.unique():
        L.append(f"  {pl:<12} " + "   ".join(f"{ts}: " + " / ".join(ch4[(ch4.planet == pl) & (ch4.variant == v)][f"verdict_{ts}"].iloc[0] for v in ("all", "airs_only")) for ts in SETS))
    open(os.path.join(RESULTS, "alfnoor_retrain_panel.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--render", action="store_true"); ap.add_argument("--fit", action="store_true"); ap.add_argument("--panel", action="store_true")
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--draws", type=int, default=500); a = ap.parse_args()
    if a.render: do_render(a.jobs)
    if a.fit: do_fit()
    if a.panel: do_panel(a.draws)


if __name__ == "__main__":
    main()
