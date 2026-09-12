"""Test 3: the reliability envelope of the randomized screen.

Builds the fully randomized training set exactly as trust_randomized.py does, trains the three
model families on it (XGBoost, random forest, MLP; the Tier-3 hyper-parameters), and runs the
decline rules of trust_detect.py with thresholds fixed on CLEAN test data (10 % declined). One
table: for every mismatch case, including held-out axes (opacity tables, Exo-Transmit code,
HCN/C2H2 absorbers) and out-of-range strengths, the accepted-set accuracy and coverage, and the
credit against the clean selective baseline at the same coverage (stress-test item S2).

Usage: python trust_envelope.py --config ariel
Writes results/ariel_trust_envelope.txt / .csv ; models/rand_full_{xgb,rf}.joblib, rand_full_mlp.keras
"""
import argparse, json, os, sys, time
import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features, centres, load_split, metrics  # noqa: E402
from noise import sigma_matrix  # noqa: E402
from pipeline import make_xgb, make_rf, fit_mlp, MLPWrap  # noqa: E402
from augment import binned, shifted_test, corr_noise  # noqa: E402
from trust_randomized import CASES, NOISE, spot_factor  # noqa: E402

EXTRA_CASES = [("absorbers", "absorbers", False), ("absorbers_quenched", "absorbers", False)]
DECLINE_CLEAN = 0.10


def build_full(cfg, cen):
    Ptr = pd.read_parquet(os.path.join(DATA, "train_params.parquet")); ytr = Ptr["label_co"].to_numpy()
    D = pd.read_parquet(os.path.join(DATA, "train_rand_draws.parquet"))
    nat = {k: np.load(os.path.join(DATA, f)).astype(np.float64) for k, f in
           (("clean_eq", "train_native.npy"), ("clean_q", "train_native_quenched.npy"),
            ("aero_eq", "train_native_rand_aero_eq.npy"), ("aero_q", "train_native_rand_aero_q.npy"))}
    for k in ("clean_q", "aero_eq", "aero_q"):
        bad = ~np.all(np.isfinite(nat[k]), axis=1); nat[k][bad] = nat["clean_eq"][bad]
    aero = np.isfinite(D.haze_density.to_numpy()) | np.isfinite(D.cloud_pressure.to_numpy()); q = D.quenched.to_numpy()
    key = np.where(aero, np.where(q, 2, 1), np.where(q, 3, 0)); X = np.empty_like(nat["clean_eq"])
    for i, k in enumerate(("clean_eq", "aero_eq", "aero_q", "clean_q")):
        X[key == i] = nat[k][key == i]
    X = X * spot_factor(Ptr, D.spot_frac.to_numpy())
    Xb = binned(X, cfg)
    sig = sigma_matrix(Xb, Ptr["s temperature"].to_numpy(), cen, 1.0, "ariel") / D.snr.to_numpy()[:, None]
    rng = np.random.default_rng(SEED + 1000)
    return Xb + rng.normal(0.0, 1.0, sig.shape) * sig, ytr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ariel"); a = ap.parse_args(); cfg = a.config
    cen = centres(cfg)
    hp = {k: joblib.load(os.path.join(MODELS, f"{cfg}_norm_{k}.joblib"))["params"] for k in ("xgb", "rf", "mlp")}
    Xn, ytr = build_full(cfg, cen)
    f = Features("norm").fit(Xn); Z = f.transform(Xn)
    t0 = time.time(); xgb = make_xgb(hp["xgb"]).fit(Z, ytr); rf = make_rf(hp["rf"]).fit(Z, ytr); mlp = MLPWrap(fit_mlp(Z, ytr, hp["mlp"]))
    print(f"trained randomized xgb/rf/mlp in {time.time()-t0:.0f} s", flush=True)
    joblib.dump({"features": f, "model": xgb, "params": hp["xgb"]}, os.path.join(MODELS, "rand_full_xgb.joblib"))
    joblib.dump({"features": f, "model": rf, "params": hp["rf"]}, os.path.join(MODELS, "rand_full_rf.joblib"))
    mlp.model.save(os.path.join(MODELS, "rand_full_mlp.keras"))
    mu = Z.mean(0); Ci = np.linalg.inv(np.cov(Z, rowvar=False) + 1e-3 * np.eye(Z.shape[1])); knn = NearestNeighbors(n_neighbors=10).fit(Z)

    def scores(X):
        Zx = f.transform(X); P = np.column_stack([m.predict_proba(Zx)[:, 1] for m in (xgb, rf, mlp)]); p = P[:, 0]; d = Zx - mu
        return dict(margin=1 - np.abs(2 * p - 1), ensemble=P.std(1), mahalanobis=np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d)),
                    knn=knn.kneighbors(Zx)[0].mean(1)), p

    Xc = np.vstack([load_split(t, cfg)[0] for t in TESTS]); yc = np.concatenate([load_split(t, cfg)[1] for t in TESTS])
    Xc_nf = np.vstack([np.load(os.path.join(DATA, f"{t}_{cfg}.npy")) for t in TESTS]).astype(float)
    Pc = pd.concat([load_split(t, cfg, noisy=False)[2] for t in TESTS], ignore_index=True)
    S0, p0 = scores(Xc); wrong0 = ((p0 >= 0.5).astype(int) != yc)
    thr = {k: np.quantile(v, 1 - DECLINE_CLEAN) for k, v in S0.items()}
    order0 = {k: np.sort(v) for k, v in S0.items()}

    def clean_selective(k, coverage):
        """clean accepted accuracy at the same coverage: the baseline a decline rule must beat"""
        t = np.quantile(S0[k], coverage); keep = S0[k] <= t
        return 1 - wrong0[keep].mean()

    sets = [("clean", "clean", True, Xc, yc)]
    for case, axis, inr in CASES + EXTRA_CASES:
        if all(os.path.exists(os.path.join(DATA, f"{t}_native_{case}.npy")) for t in TESTS):
            Xs, ys = shifted_test(case, cfg); sets.append((case, axis, inr, Xs, ys))
    rng = np.random.default_rng(SEED + 61)
    for kindn, s, inr in NOISE:
        sets.append((f"{kindn}_snr{s}", "noise", inr, corr_noise(Xc, rng, s, kind=kindn, Xnf=Xc_nf, params=Pc, cen=cen), yc))

    rows = []
    for name, axis, inr, X, y in sets:
        S, p = scores(X); wrong = ((p >= 0.5).astype(int) != y)
        for k, v in S.items():
            keep = v <= thr[k]; cov = keep.mean()
            acc_acc = 1 - wrong[keep].mean() if keep.any() else np.nan
            rows.append(dict(case=name, axis=axis, in_range=inr, score=k, accuracy_all=1 - wrong.mean(), coverage=cov, accuracy_accepted=acc_acc,
                             clean_selective_same_coverage=clean_selective(k, cov) if cov > 0 else np.nan,
                             auroc_error=roc_auc_score(wrong, v) if 0 < wrong.sum() < len(wrong) else np.nan))
    df = pd.DataFrame(rows); df["credit"] = df.accuracy_accepted - df.clean_selective_same_coverage

    L = [f"Reliability envelope of the randomized screen (norm xgb/rf/mlp trained on the randomized grid), configuration {cfg}",
         f"Decline thresholds: {DECLINE_CLEAN:.0%} of clean test planets declined, per score, fixed on clean data only.", "",
         "accepted = accuracy of the planets kept; cov = share kept; credit = accepted minus the CLEAN accepted accuracy at the same",
         "coverage (what declining low-score planets buys on clean data anyway). in/OUT = inside the randomized training range.", ""]
    for k in ("ensemble", "margin", "mahalanobis", "knn"):
        g = df[df.score == k]
        L += [f"--- score: {k}", f"{'case':<32}{'range':>6}{'all':>8}{'accepted':>10}{'cov':>7}{'credit':>8}{'AUROC(err)':>11}"]
        for _, r in g.iterrows():
            L.append(f"{r.case:<32}{'in' if r.in_range else 'OUT':>6}{r.accuracy_all*100:8.2f}{r.accuracy_accepted*100:10.2f}{r.coverage*100:7.1f}{r.credit*100:+8.2f}{r.auroc_error:11.3f}")
        sh = g[g.case != "clean"]
        L += [f"   over shifted cases: worst accepted {sh.accuracy_accepted.min()*100:.2f}%, mean coverage {sh.coverage.mean()*100:.1f}%, mean credit {sh.credit.mean()*100:+.2f}", ""]
    c = df[(df.case == "clean") & (df.score == "ensemble")].iloc[0]
    L.append(f"randomized screen on clean data: {c.accuracy_all*100:.2f}% (frozen clean-trained screen: see results/{cfg}_trust_randomized.txt)")
    open(os.path.join(RESULTS, f"{cfg}_trust_envelope.txt"), "w").write("\n".join(L) + "\n")
    df.to_csv(os.path.join(RESULTS, f"{cfg}_trust_envelope.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
