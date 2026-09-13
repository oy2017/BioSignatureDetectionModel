"""Causal test of the haze blindness of the consortium Tier-1 screen (alfnoor_trust.py): is it the
per-spectrum normalisation over the optical photometer points?

Four versions of the published design, identical except for the input normalisation, each retrained on the
clean POP-III grid (same spectra, same noise draw) and scored at the 1e-4 threshold (mean over KNN/MLP/RFC/SVC
and CH4/H2O/CO2/NH3):
  A published   per-spectrum zero mean / unit dispersion over all 104 bins
  B ir_stats    all 104 bins kept, mean and dispersion taken from the bins above 1.1 um only
  C no_optical  the three photometric points dropped, remaining 101 bins normalised
  D airs_only   only the AIRS bins (above 1.95 um), normalised
Writes results/alfnoor_haze_mechanism.txt / .csv
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED
from bin_spectra import bin_native
import alfnoor_trust as AT, alfnoor_faithful as F, shift_tlse as T

CASES = ["clean", "haze2e6", "haze3e7", "haze2p4e8", "haze1e10", "cloud1e3", "cloud1e2", "spots10", "spots20", "compound",
         "white_x2", "corr_x2", "exomol", "exotransmit", "absorbers"]

def main():
    wl = np.load(os.path.join(DATA, "native_wl.npy")); e = F.layout_edges("tier3_r20"); cen = 0.5 * (e[1:] + e[:-1])
    Ptr, Pte = AT._params(); load = lambda n: np.load(os.path.join(AT.OUT, f"{n}.npy")).astype(float)
    Xtr = load("pop3_native"); ok_tr = np.isfinite(Xtr).all(1); Xte = load("pop1_native"); ok = np.isfinite(Xte).all(1)
    Ptr, Pte = Ptr[ok_tr].reset_index(drop=True), Pte[ok].reset_index(drop=True); clean_n = Xte[ok]
    rng = np.random.default_rng(SEED + 405)
    sig_tr = F.sigma(Ptr, e, "radiometric"); eps_tr = rng.normal(0, 1, (len(Ptr), len(e) - 1))
    sig_te = F.sigma(Pte, e, "radiometric"); eps_te = rng.normal(0, 1, sig_te.shape)
    r2 = np.random.default_rng(SEED + 407); eps2 = r2.normal(0, 1, sig_te.shape)
    B = lambda X: bin_native(X, wl, e)
    def shifted(name):
        Y = load(f"pop1_native_{name}")[ok]; bad = ~np.isfinite(Y).all(1); Y[bad] = clean_n[bad]; return Y
    T.set_grid(wl); logg = np.log10(T.G_SUN * Pte["s mass"].to_numpy() / Pte["s radius"].to_numpy() ** 2 * 100)
    contam = lambda f: T.contamination(Pte["s temperature"].to_numpy(), logg, f, 0.0)
    from scipy.ndimage import gaussian_filter1d
    Bc = B(clean_n); nb = {}
    for c in CASES:
        if c == "clean": nb[c] = Bc + eps_te * sig_te
        elif c.startswith("spots"): nb[c] = B(clean_n * contam(int(c[5:]) / 100)) + eps_te * sig_te
        elif c == "compound": nb[c] = B(shifted("haze3e7") * contam(0.20)) + eps_te * sig_te
        elif c == "white_x2": nb[c] = Bc + eps_te * sig_te + eps2 * sig_te * np.sqrt(3)
        elif c == "corr_x2":
            z = gaussian_filter1d(eps2, 3.0, axis=1); z /= z.std(1, keepdims=True); nb[c] = Bc + eps_te * sig_te + z * sig_te * np.sqrt(3)
        else: nb[c] = B(shifted(c)) + eps_te * sig_te
    Xtr_noisy = B(Xtr[ok_tr]) + eps_tr * sig_tr
    ir, airs = cen > 1.1, cen > 1.95
    def z(X, cols):
        return (X[:, cols] - X[:, cols].mean(1, keepdims=True)) / (X[:, cols].std(1, keepdims=True) + 1e-12)
    VAR = {"A published": lambda X: AT.norm(X),
           "B ir_stats": lambda X: (X - X[:, ir].mean(1, keepdims=True)) / (X[:, ir].std(1, keepdims=True) + 1e-12),
           "C no_optical": lambda X: z(X, ir),
           "D airs_only": lambda X: z(X, airs)}
    mk = AT.makers(); rows = []
    for vname, fn in VAR.items():
        Ztr = fn(Xtr_noisy); Zc = {c: fn(X) for c, X in nb.items()}
        for mol in AT.MOLS:
            ytr = (Ptr[f"atm {mol}"] > -4).astype(int).to_numpy(); y = (Pte[f"atm {mol}"] > -4).astype(int).to_numpy()
            for cname, mkr in mk.items():
                m = mkr().fit(Ztr, ytr)
                for c in CASES:
                    pr = m.predict(Zc[c]); rows.append(dict(variant=vname, molecule=mol, classifier=cname, case=c,
                                                          accuracy=(pr == y).mean(), recall=(pr[y == 1] == 1).mean(), called_present=pr.mean()))
        print("done", vname, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "alfnoor_haze_mechanism.csv"), index=False)
    g = df.groupby(["variant", "case"])[["accuracy", "recall", "called_present"]].mean() * 100
    L = ["Haze blindness of the consortium screen: normalisation variants (1e-4; mean over 4 molecules x 4 classifiers)",
         "cells: accuracy % / recall on planets with the molecule % ; loss = clean accuracy minus case accuracy", ""]
    L.append(f"{'case':<12}" + "".join(f"{v:>27}" for v in VAR))
    for c in CASES:
        cells = ""
        for v in VAR:
            a, r, cl = g.loc[(v, c)]; loss = g.loc[(v, "clean")].accuracy - a
            cells += f"{a:8.1f} /{r:5.1f} ({loss:+5.1f})".rjust(27)
        L.append(f"{c:<12}{cells}")
    L += ["", "per molecule, haze 3e7 accuracy (clean):"]
    gm = df.groupby(["variant", "molecule", "case"]).accuracy.mean() * 100
    for v in VAR:
        L.append(f"  {v:<14}" + "  ".join(f"{mol} {gm.loc[(v, mol, 'haze3e7')]:.1f} ({gm.loc[(v, mol, 'clean')]:.1f})" for mol in AT.MOLS))
    open(os.path.join(RESULTS, "alfnoor_haze_mechanism.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
