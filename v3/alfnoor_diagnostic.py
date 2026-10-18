"""A diagnostic with measurable value (REVISION_PLAN.md, work package C; expectations C1-C6).

Candidates, scored per planet (and per molecule and classifier where they depend on one):
  (a) disagree   input-subset disagreement: max |p_all - p_no_optical|, |p_all - p_airs_only| of the
                 same learner retrained without the optical points / on the AIRS bins (alfnoor_jwst)
  (b) optical    optical-excess score: the mean of the three normalized photometric points minus the
                 mean of the normalized NIRSpec bins, as a z-score against the clean training set
  (c) bandmetric band-metric consistency: |p_all - 1[M_mol >= flag]| with Mugnai et al.'s M_mol and
                 their flags (H2O 0.80, CH4 0.22, CO2 0.26; NH3 has none and is not scored)
  (d) decidable  predicted decidability: a random forest on bulk and noise features predicts the
                 identifiability test's log dchi2 (identifiability.py; cross-validated out of fold on
                 the planets the test has reached); the score is minus the prediction
The five rules of the note (margin, ensemble spread, Mahalanobis, kNN, PCA reconstruction) are
recomputed on the same cases for comparison.

Evaluation, on POP-I under the cases of alfnoor_haze_mechanism (clean, haze3e7, spots20, cloud1e2,
compound, white_x2, white_x3), with the noise draws of the note:
  credit     selective credit as in alfnoor_trust: threshold declining 10 % of clean planets,
             accepted accuracy minus the clean selective baseline at equal coverage; AUROC(error)
  remedy     accuracy over ALL planets when the planets a candidate flags are re-scored by the fallback
             (the AIRS-only learner for (a) and (b); the band-metric flag for (c)), against the frozen
             classifier and the fallback used everywhere
  classes    (d): AUROC separating recoverable errors (wrong, label decidable) from irrecoverable ones
             (wrong, label not decidable) on the planets the identifiability test has reached
  selection  the Tier-2 selection simulation: 60 % of the planets get a slot; rank by the mean present
             probability of KNN, MLP and RFC; targets lost (label 1 not selected) and slots wasted
             (label 0 selected), cost = w x lost + wasted for w = 1, 3, 10; strategies: classifier
             alone, + each old rule (declined planets to the back), + each candidate (flagged planets
             re-ranked by the fallback), the band metric alone, random, and the oracle

Usage:  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python alfnoor_diagnostic.py [--real]
Writes results/diagnostic_{credit,remedy,selection,classes}.csv and results/diagnostic.txt; --real scores
the converted JWST spectra (WASP-39 b and the panel) with the candidates at the clean thresholds.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, RESULTS, SEED  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
import alfnoor_faithful as F  # noqa: E402
import alfnoor_trust as AT  # noqa: E402
import alfnoor_jwst as AJ  # noqa: E402
import jwst_to_ariel as J  # noqa: E402

CASES = ["clean", "haze3e7", "spots20", "cloud1e2", "compound", "white_x2", "white_x3"]
MOLS = ["CH4", "H2O", "CO2", "NH3"]; LEARNERS = ["KNN", "MLP", "RFC", "SVC"]
FLAGS = {"H2O": 0.80, "CH4": 0.22, "CO2": 0.26}
OLD_RULES = ["margin", "ensemble", "mahalanobis", "knn", "pca_recon"]; NEW = ["disagree", "optical", "bandmetric", "decidable"]
DECLINE = 0.10; BUDGET = 0.60; WEIGHTS = (1, 3, 10)


def test_cases():
    """POP-I under each case, binned and noised exactly as alfnoor_haze_mechanism; returns (cases, Pte, edges)."""
    import shift_tlse as T
    from scipy.ndimage import gaussian_filter1d
    wl = np.load(os.path.join(DATA, "native_wl.npy")); e = F.layout_edges("tier3_r20"); src = F.DIRS["faithful"]
    Ptr = pd.read_parquet(os.path.join(src, "pop3_params.parquet")); ok_tr = np.isfinite(np.load(os.path.join(src, "pop3_native.npy")).astype(float)).all(1)
    Xte = np.load(os.path.join(src, "pop1_native.npy")).astype(float); ok = np.isfinite(Xte).all(1); clean_n = Xte[ok]
    Pte = pd.read_parquet(os.path.join(src, "pop1_params.parquet"))[ok].reset_index(drop=True)
    rng = np.random.default_rng(SEED + 405); _ = rng.normal(0, 1, (int(ok_tr.sum()), len(e) - 1))     # the training draw comes first
    sig_te = F.sigma(Pte, e, "radiometric"); eps_te = rng.normal(0, 1, sig_te.shape)
    eps2 = np.random.default_rng(SEED + 407).normal(0, 1, sig_te.shape)
    T.set_grid(wl); logg = np.log10(T.G_SUN * Pte["s mass"].to_numpy() / Pte["s radius"].to_numpy() ** 2 * 100)
    contam = lambda f: T.contamination(Pte["s temperature"].to_numpy(), logg, f, 0.0)
    B = lambda X: bin_native(X, wl, e)

    def shifted(name):
        Y = np.load(os.path.join(src, f"pop1_native_{name}.npy")).astype(float)[ok]; bad = ~np.isfinite(Y).all(1); Y[bad] = clean_n[bad]; return Y
    Bc = B(clean_n); out = {}
    for c in CASES:
        if c == "clean": out[c] = Bc + eps_te * sig_te
        elif c.startswith("spots"): out[c] = B(clean_n * contam(int(c[5:]) / 100)) + eps_te * sig_te
        elif c == "compound": out[c] = B(shifted("haze3e7") * contam(0.20)) + eps_te * sig_te
        elif c.startswith("white_x"): k = float(c[7:]); out[c] = Bc + eps_te * sig_te + eps2 * sig_te * np.sqrt(k ** 2 - 1)
        else: out[c] = B(shifted(c)) + eps_te * sig_te
    return out, Pte, e, sig_te


class Scorer:
    """Everything needed to score a batch of 104-bin spectra: the three learner variants, the old rules'
    reference statistics, the band metric, the optical-excess reference, and the decidability regressor."""

    def __init__(self, edges):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        self.edges = edges; cen = 0.5 * (edges[1:] + edges[:-1]); self.cen = cen
        self.L = {v: AJ.learners(v) for v in ("all", "no_optical", "airs_only")}
        Xtr, Ptr, _ = AJ.training_set(); Ztr = AT.norm(Xtr); self.Ptr = Ptr
        self.mu = Ztr.mean(0); self.Ci = np.linalg.inv(np.cov(Ztr, rowvar=False) + 1e-3 * np.eye(Ztr.shape[1]))
        self.nn = NearestNeighbors(n_neighbors=10).fit(Ztr); self.pca = PCA(n_components=0.99, random_state=SEED).fit(Ztr)
        self.opt, self.nir = cen < 1.1, (cen > 1.1) & (cen < 1.95)
        xo = self._optical(Ztr); self.opt_mu, self.opt_sd = xo.mean(), xo.std()
        self.reg = None

    def _optical(self, Z):
        return Z[:, self.opt].mean(1) - Z[:, self.nir].mean(1)

    def probs(self, X):
        """{variant: {(mol, learner): p_present}} at 1e-4 for a batch X (n, 104)."""
        out = {}
        for v, L in self.L.items():
            Z = AJ.znorm(X, L["cols"]); out[v] = {(mol, k): AT.prob(L["models"][("1e-4", mol, k)], Z) for mol in MOLS for k in LEARNERS}
        return out

    def band(self, X):
        return {mol: np.array([J.band_metric(x, self.edges, mol) for x in X]) for mol in FLAGS}

    def scores(self, X, P=None, M=None):
        """Per-planet scores: molecule-independent ones under key None, the rest under (mol, learner)."""
        P = P or self.probs(X); M = M or self.band(X); Z = AT.norm(X); d = Z - self.mu; S = {}
        S[("mahalanobis", None)] = np.sqrt(np.einsum("ij,jk,ik->i", d, self.Ci, d)); S[("knn", None)] = self.nn.kneighbors(Z)[0].mean(1)
        S[("pca_recon", None)] = np.linalg.norm(Z - self.pca.inverse_transform(self.pca.transform(Z)), axis=1)
        S[("optical", None)] = np.abs(self._optical(Z) - self.opt_mu) / self.opt_sd
        for mol in MOLS:
            ens = np.std(np.column_stack([P["all"][(mol, k)] for k in LEARNERS]), axis=1)
            for k in LEARNERS:
                p = P["all"][(mol, k)]
                S[("margin", (mol, k))] = 1 - np.abs(2 * p - 1); S[("ensemble", (mol, k))] = ens
                S[("disagree", (mol, k))] = np.maximum(np.abs(p - P["no_optical"][(mol, k)]), np.abs(p - P["airs_only"][(mol, k)]))
                S[("bandmetric", (mol, k))] = np.abs(p - (M[mol] >= FLAGS[mol]).astype(float)) if mol in FLAGS else np.full(len(p), np.nan)
        return S, P, M

    def get(self, S, rule, mol, k):
        return S[(rule, None)] if (rule, None) in S else S[(rule, (mol, k))]


# ---- (d) predicted decidability -------------------------------------------------------------------------
def decidability_features(P, sig, X):
    """Bulk and noise features per planet: equilibrium temperature, log transit count, a log scale-height
    proxy T R_p / M_p, the MCS modulation metric, the per-channel median sigma (log ppm) and the spectrum's
    dispersion-to-sigma ratio."""
    ch = {"phot": slice(0, 3), "nirspec": slice(3, 14), "ch0": slice(14, 83), "ch1": slice(83, 104)}
    f = pd.DataFrame({"teq": P["atm temperature"].to_numpy(float), "logN": np.log10(P["tier1_transits"].to_numpy(float).clip(1)),
                      "logH": np.log10(P["atm temperature"].to_numpy(float) * P["p_radius"].to_numpy(float) / P["p_mass"].to_numpy(float)),
                      "mod5H": P["modulation_5H"].to_numpy(float)})
    for c, sl in ch.items():
        f[f"sig_{c}"] = np.log10(np.median(sig[:, sl], axis=1) * 1e6)
    f["amp_over_sig"] = np.log10(X.std(1) / np.median(sig, axis=1))
    return f


def decidability_regressor(Pte, sig, X, csv=os.path.join(RESULTS, "identifiability_none_tier1.csv")):
    """Out-of-fold predictions of log10 dchi2 per (planet, molecule) for the planets the identifiability run has
    reached, and a regressor fitted on all of them for the rest. Returns (pred_oof DataFrame, fitted model, features)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    if not os.path.exists(csv):
        return None, None, None
    I = pd.read_csv(csv); I = I[np.isfinite(I.dchi2)]; feats = decidability_features(Pte, sig, X); feats["index"] = np.arange(len(Pte)); name_to_i = {n: i for i, n in enumerate(Pte.name)}
    I = I[I.name.isin(name_to_i)].copy(); I["i"] = I.name.map(name_to_i); I["y"] = np.log10(I.dchi2.clip(lower=1e-3))
    rows = []
    for mol in MOLS:
        D = I[I.molecule == mol]
        if len(D) < 40:
            continue
        Xf = feats.iloc[D.i.to_numpy()].drop(columns="index").to_numpy(); y = D.y.to_numpy(); oof = np.full(len(D), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xf):
            oof[te] = RandomForestRegressor(300, random_state=SEED, n_jobs=4).fit(Xf[tr], y[tr]).predict(Xf[te])
        full = RandomForestRegressor(300, random_state=SEED, n_jobs=4).fit(Xf, y)
        rows.append(pd.DataFrame(dict(i=D.i.to_numpy(), molecule=mol, log_dchi2=y, pred=oof, label=D.label.to_numpy(), model=[full] * len(D))))
    return (pd.concat(rows) if rows else None), feats


# ---- evaluations ----------------------------------------------------------------------------------------------
def evaluate(cases, Pte, edges, sig_te, sc):
    y_true = {mol: (Pte[f"atm {mol}"] > -4).astype(int).to_numpy() for mol in MOLS}
    S, P, M = {}, {}, {}
    for c, X in cases.items():
        S[c], P[c], M[c] = sc.scores(X); print(f"scored {c}", flush=True)
    # (d): out-of-fold predicted decidability on the reached planets (the clean case's features; the score is per planet x molecule)
    dec, feats = decidability_regressor(Pte, sig_te, cases["clean"])
    reached = {}
    if dec is not None:
        for mol in MOLS:
            D = dec[dec.molecule == mol]
            if len(D):
                s = np.full(len(Pte), np.nan); s[D.i.to_numpy()] = -D.pred.to_numpy(); reached[mol] = (D.i.to_numpy(), D)
                for c in CASES:
                    for k in LEARNERS:
                        S[c][("decidable", (mol, k))] = s
    rules = OLD_RULES + [r for r in NEW if r != "decidable" or reached]
    # 1. selective credit
    rows = []
    for mol in MOLS:
        y = y_true[mol]
        for k in LEARNERS:
            pred0 = (P["clean"]["all"][(mol, k)] >= 0.5).astype(int); wrong0 = pred0 != y
            for rule in rules:
                s0 = sc.get(S["clean"], rule, mol, k); valid = np.isfinite(s0)
                if valid.sum() < 50:
                    continue
                thr = np.quantile(s0[valid], 1 - DECLINE); order0 = np.argsort(np.where(valid, s0, np.inf))
                for c in CASES:
                    s = sc.get(S[c], rule, mol, k); pred = (P[c]["all"][(mol, k)] >= 0.5).astype(int); wrong = pred != y
                    v = np.isfinite(s); keep = v & (s <= thr); cov = keep.sum() / v.sum()
                    acc_k = (~wrong[keep]).mean() if keep.any() else np.nan
                    nkeep = max(int(round(cov * valid.sum())), 1); base = (~wrong0[order0[:nkeep]]).mean()
                    rows.append(dict(molecule=mol, classifier=k, rule=rule, case=c, all=(~wrong[v]).mean(), coverage=cov, accepted=acc_k,
                                     credit=acc_k - base, auroc_error=AT.auroc(wrong[v], s[v]), declined_frac=1 - cov))
    credit = pd.DataFrame(rows)
    # 2. remedy: flagged planets re-scored by the fallback
    rows = []
    for mol in MOLS:
        y = y_true[mol]
        for k in LEARNERS:
            for rule in ["disagree", "optical", "bandmetric"]:
                s0 = sc.get(S["clean"], rule, mol, k); valid = np.isfinite(s0)
                if valid.sum() < 50:
                    continue
                thr = np.quantile(s0[valid], 1 - DECLINE)
                for c in CASES:
                    p = P[c]["all"][(mol, k)]; s = sc.get(S[c], rule, mol, k); flag = np.isfinite(s) & (s > thr)
                    fb = P[c]["airs_only"][(mol, k)] if rule != "bandmetric" else (M[c][mol] >= FLAGS[mol]).astype(float)
                    p_rem = np.where(flag, fb, p)
                    rows.append(dict(molecule=mol, classifier=k, rule=rule, case=c, flagged=flag.mean(), frozen=((p >= 0.5) == y).mean(),
                                     fallback_everywhere=((fb >= 0.5) == y).mean(), remedied=((p_rem >= 0.5) == y).mean(),
                                     recall_frozen=((p >= 0.5)[y == 1]).mean(), recall_remedied=((p_rem >= 0.5)[y == 1]).mean()))
    remedy = pd.DataFrame(rows)
    # 3. error classes (d)
    rows = []
    thr_file = os.path.join(RESULTS, "identifiability_calibration_thresholds.csv")
    THR = {r.molecule: (r.dchi2_at_95 if np.isfinite(r.dchi2_at_95) else 9.0) for r in pd.read_csv(thr_file).itertuples()} if os.path.exists(thr_file) else {}
    print("decidability thresholds (dchi2):", {m: round(THR.get(m, 9.0), 2) for m in MOLS}, "(calibrated at p_true > 0.95)" if THR else "(3 sigma, uncalibrated)", flush=True)
    for mol, (idx, D) in reached.items():
        for k in LEARNERS:
            for c in CASES:
                pred = (P[c]["all"][(mol, k)][idx] >= 0.5).astype(int); wrong = pred != y_true[mol][idx]
                decid = D.log_dchi2.to_numpy() > np.log10(THR.get(mol, 9.0))
                if wrong.sum() < 10 or (wrong & decid).sum() < 3 or (wrong & ~decid).sum() < 3:
                    continue
                # score: predicted undecidability; positive class: irrecoverable error (wrong & undecidable)
                rows.append(dict(molecule=mol, classifier=k, case=c, n_wrong=int(wrong.sum()), irrecoverable=float((~decid[wrong]).mean()),
                                 auroc_classes=AT.auroc(~decid[wrong], -D.pred.to_numpy()[wrong]), auroc_oracle=AT.auroc(~decid[wrong], -D.log_dchi2.to_numpy()[wrong]),
                                 spearman=pd.Series(D.pred.to_numpy()).corr(pd.Series(D.log_dchi2.to_numpy()), method="spearman")))
    classes = pd.DataFrame(rows)
    # 3b. self-consistency of the split: on labels the test calls decidable the classifier should be near perfect; on
    # undecidable ones near the majority-class rate of that subset (what a prior-only predictor scores)
    rows = []
    for mol, (idx, D) in reached.items():
        decid = D.log_dchi2.to_numpy() > np.log10(THR.get(mol, 9.0)); y = y_true[mol][idx]
        for k in LEARNERS:
            for c in CASES:
                pred = (P[c]["all"][(mol, k)][idx] >= 0.5).astype(int); right = pred == y
                r = dict(molecule=mol, classifier=k, case=c, n=len(y), n_decidable=int(decid.sum()), undecidable_frac=float((~decid).mean()))
                for nm, m in (("decidable", decid), ("undecidable", ~decid)):
                    r[f"acc_{nm}"] = right[m].mean() if m.any() else np.nan
                    r[f"base_{nm}"] = max(y[m].mean(), 1 - y[m].mean()) if m.any() else np.nan
                    r[f"recall_{nm}"] = pred[m & (y == 1)].mean() if (m & (y == 1)).any() else np.nan
                r["acc_all"] = right.mean(); r["base_all"] = max(y.mean(), 1 - y.mean())
                rows.append(r)
    consistency = pd.DataFrame(rows)
    # 4. selection simulation
    rows = []; n = len(Pte); slots = int(round(BUDGET * n)); rng = np.random.default_rng(SEED)
    for target in ("CH4", "H2O", "joint"):
        mols = ["CH4", "H2O"] if target == "joint" else [target]
        y = np.clip(sum(y_true[m] for m in mols), 0, 1)
        for c in CASES:
            score = sum(np.mean([P[c]["all"][(m, k)] for k in ("KNN", "MLP", "RFC")], axis=0) for m in mols)
            fb = sum(np.mean([P[c]["airs_only"][(m, k)] for k in ("KNN", "MLP", "RFC")], axis=0) for m in mols)
            strategies = {"classifier": score, "oracle": y + rng.uniform(0, 1e-3, n), "random": rng.uniform(0, 1, n),
                          "band metric": sum(M[c][m] for m in mols), "AIRS-only classifier": fb}
            for rule in rules:
                if rule == "decidable" and target == "joint":
                    continue
                s_rules, s0_rules = [], []
                for m in mols:
                    for k in ("KNN", "MLP", "RFC"):
                        s_rules.append(sc.get(S[c], rule, m, k)); s0_rules.append(sc.get(S["clean"], rule, m, k))
                s = np.nanmean(np.column_stack(s_rules), axis=1); s0 = np.nanmean(np.column_stack(s0_rules), axis=1); v = np.isfinite(s0)
                thr = np.quantile(s0[v], 1 - DECLINE); flag = np.isfinite(s) & (s > thr)
                if rule in OLD_RULES:
                    strategies[f"+ {rule} (declined to the back)"] = np.where(flag, score - 10, score)
                elif rule == "decidable":
                    strategies["+ decidable (undecidable ranked by decidability)"] = np.where(flag, -10 + (s - s.min()) / (np.ptp(s) + 1e-9), score)
                else:
                    strategies[f"+ {rule} (flagged re-scored by the fallback)"] = np.where(flag, fb, score)
            for name, sc_ in strategies.items():
                sel = np.zeros(n, bool); sel[np.argsort(-sc_)[:slots]] = True
                lost = int((y == 1).sum() - (sel & (y == 1)).sum()); wasted = int((sel & (y == 0)).sum())
                rows.append(dict(target=target, case=c, strategy=name, targets=int((y == 1).sum()), lost=lost, wasted=wasted,
                                 **{f"cost_w{w}": w * lost + wasted for w in WEIGHTS}))
    selection = pd.DataFrame(rows)
    return credit, remedy, classes, selection, consistency


def write_summary(credit, remedy, classes, selection, consistency=None):
    L = ["Diagnostic candidates on POP-I (1e-4; mean over molecules and the four learners unless stated)", ""]
    L.append("== selective credit (threshold declines 10 % of clean planets; credit = accepted accuracy - clean selective baseline at equal coverage), points")
    g = credit.groupby(["rule", "case"])[["coverage", "accepted", "credit", "auroc_error"]].mean()
    L.append(f"{'rule':<12}" + "".join(f"{c:>22}" for c in CASES)); L.append(f"{'':<12}" + "".join(f"{'keep% credit AUROCerr':>22}" for c in CASES))
    for rule in OLD_RULES + NEW:
        if rule not in g.index.get_level_values(0): continue
        L.append(f"{rule:<12}" + "".join(f"{100*g.loc[(rule,c)].coverage:5.0f} {100*g.loc[(rule,c)].credit:+6.1f} {g.loc[(rule,c)].auroc_error:7.3f}".rjust(22) for c in CASES))
    L.append(""); L.append("== per-molecule credit of the new candidates under haze3e7 and spots20 (mean over learners)")
    for rule in ["disagree", "optical", "bandmetric"]:
        gm = credit[credit.rule == rule].groupby(["molecule", "case"]).credit.mean()
        L.append(f"  {rule:<11}" + "  ".join(f"{m} haze {100*gm.get((m,'haze3e7'), np.nan):+5.1f} spots {100*gm.get((m,'spots20'), np.nan):+5.1f}" for m in MOLS if (m, "haze3e7") in gm))
    L.append(""); L.append("== remedy: accuracy over all planets when flagged planets take the fallback (points; mean over molecules and learners)")
    g = remedy.groupby(["rule", "case"])[["flagged", "frozen", "fallback_everywhere", "remedied", "recall_frozen", "recall_remedied"]].mean()
    L.append(f"{'rule':<12}" + "".join(f"{c:>30}" for c in CASES)); L.append(f"{'':<12}" + "".join(f"{'flag% frozen fallbk remedied':>30}" for c in CASES))
    for rule in ["disagree", "optical", "bandmetric"]:
        L.append(f"{rule:<12}" + "".join(f"{100*g.loc[(rule,c)].flagged:5.0f} {100*g.loc[(rule,c)].frozen:6.1f} {100*g.loc[(rule,c)].fallback_everywhere:6.1f} {100*g.loc[(rule,c)].remedied:6.1f}".rjust(30) for c in CASES))
    gm = remedy.groupby(["rule", "molecule", "case"])[["frozen", "remedied", "recall_frozen", "recall_remedied"]].mean()
    L.append("  CH4 under haze3e7, per candidate: accuracy frozen -> remedied, recall frozen -> remedied")
    for rule in ["disagree", "optical", "bandmetric"]:
        r = gm.loc[(rule, "CH4", "haze3e7")]; L.append(f"    {rule:<11} {100*r.frozen:5.1f} -> {100*r.remedied:5.1f}   recall {100*r.recall_frozen:5.1f} -> {100*r.recall_remedied:5.1f}")
    if classes is not None and len(classes):
        L.append(""); L.append("== (d) error classes on the planets the identifiability test has reached: AUROC of predicted (and oracle) decidability for irrecoverable vs recoverable errors")
        g = classes.groupby(["molecule", "case"])[["n_wrong", "irrecoverable", "auroc_classes", "auroc_oracle", "spearman"]].mean()
        for (m, c), r in g.iterrows():
            L.append(f"  {m} {c:<10} wrong {r.n_wrong:5.0f}  irrecoverable {100*r.irrecoverable:4.0f} %  AUROC predicted {r.auroc_classes:.3f}  oracle {r.auroc_oracle:.3f}  Spearman(pred, dchi2) {r.spearman:.2f}")
    if consistency is not None and len(consistency):
        L.append(""); L.append("== (d') self-consistency of the decidable / undecidable split at the calibrated threshold (3 sigma if uncalibrated) (clean case unless stated; base = majority-class rate of the subset)")
        L.append(f"  {'mol':<4} {'learner':<4} {'undec%':>6} {'acc_dec':>8} {'base_dec':>9} {'acc_undec':>10} {'base_undec':>11} {'rec_dec':>8} {'rec_undec':>10} {'acc_all':>8} {'base_all':>9}")
        for _, r in consistency[consistency.case == "clean"].iterrows():
            L.append(f"  {r.molecule:<4} {r.classifier:<4} {100*r.undecidable_frac:6.1f} {100*r.acc_decidable:8.1f} {100*r.base_decidable:9.1f} {100*r.acc_undecidable:10.1f} {100*r.base_undecidable:11.1f} {100*r.recall_decidable:8.1f} {100*r.recall_undecidable:10.1f} {100*r.acc_all:8.1f} {100*r.base_all:9.1f}")
        g = consistency.groupby(["molecule", "case"])[["acc_decidable", "acc_undecidable", "base_undecidable", "recall_decidable", "recall_undecidable"]].mean()
        L.append("  mean over learners, per case: acc on decidable / undecidable labels (base of undecidable), recall decidable / undecidable")
        for (m, c), r in g.iterrows():
            L.append(f"    {m} {c:<10} {100*r.acc_decidable:5.1f} / {100*r.acc_undecidable:5.1f} ({100*r.base_undecidable:5.1f})   recall {100*r.recall_decidable:5.1f} / {100*r.recall_undecidable:5.1f}")
    L.append(""); L.append(f"== Tier-2 selection: {int(BUDGET*100)} % of the planets get a slot; lost = targets not selected, wasted = slots on non-targets; cost = w x lost + wasted")
    for target in ("CH4", "H2O", "joint"):
        for c in ["clean", "haze3e7", "spots20", "white_x2"]:
            d = selection[(selection.target == target) & (selection.case == c)].sort_values("cost_w3")
            L.append(f"-- {target} under {c} ({int(d.targets.iloc[0])} targets)"); L.append(f"   {'strategy':<52} {'lost':>5} {'wasted':>7}   cost w=1  w=3  w=10")
            for _, r in d.iterrows():
                L.append(f"   {r.strategy:<52} {r.lost:5d} {r.wasted:7d}   {r.cost_w1:6d} {r.cost_w3:5d} {r.cost_w10:5d}")
    return "\n".join(L)


def score_real(sc):
    """The candidates on the converted JWST spectra, against the clean thresholds (expectation C5)."""
    cases, Pte, e, sig = test_cases(); S0, P0, M0 = sc.scores(cases["clean"])
    files = {"WASP-39b": os.path.join(RESULTS, "wasp39", "coverage_benchmark.csv")}
    for f in sorted(os.listdir(os.path.join(RESULTS, "jwst_panel"))):
        if f.endswith("_coverage.csv"):
            files[f[:-13]] = os.path.join(RESULTS, "jwst_panel", f)
    L = ["Diagnostic candidates on the real spectra (all-input learners at 1e-4; flag = score above the threshold declining 10 % of clean POP-I planets)", ""]
    L.append(f"{'planet':<12} {'molecule':<9} {'optical':>8} " + " ".join(f"{'disagree ' + k:>13} {'band ' + k:>9}" for k in LEARNERS))
    thr = {}
    for pl, f in files.items():
        cov = pd.read_csv(f); x = cov.depth.to_numpy(); filled = cov.filled.to_numpy(bool); x = x.copy(); x[filled] = np.nanmean(x[~filled])
        S, P, M = sc.scores(x[None, :])
        so = S[("optical", None)][0]; t_o = np.quantile(S0[("optical", None)], 1 - DECLINE)
        for mol in MOLS:
            cells = []
            for k in LEARNERS:
                for rule in ("disagree", "bandmetric"):
                    s = S[(rule, (mol, k))][0]; s0 = S0[(rule, (mol, k))]; t = np.quantile(s0[np.isfinite(s0)], 1 - DECLINE) if np.isfinite(s0).any() else np.nan
                    cells.append(f"{s:5.2f}{'*' if s > t else ' '}")
            L.append(f"{pl:<12} {mol:<9} {so:5.2f}{'*' if so > t_o else ' '}   " + "   ".join(f"{cells[2*i]:>12} {cells[2*i+1]:>8}" for i in range(4)))
    L.append(""); L.append("* = flagged. optical: the optical-excess z-score (molecule-independent); disagree: max |p_all - p_no_optical|, |p_all - p_airs_only|; band: |p_all - band-metric flag| (NH3 has no flag)")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true"); a = ap.parse_args()
    e = F.layout_edges("tier3_r20"); sc = Scorer(e)
    if a.real:
        text = score_real(sc); print(text); open(os.path.join(RESULTS, "diagnostic_real.txt"), "w").write(text + "\n"); return
    cases, Pte, e, sig = test_cases()
    credit, remedy, classes, selection, consistency = evaluate(cases, Pte, e, sig, sc)
    for name, df in (("credit", credit), ("remedy", remedy), ("classes", classes), ("selection", selection), ("consistency", consistency)):
        if df is not None: df.to_csv(os.path.join(RESULTS, f"diagnostic_{name}.csv"), index=False)
    text = write_summary(credit, remedy, classes, selection, consistency); print(text); open(os.path.join(RESULTS, "diagnostic.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
