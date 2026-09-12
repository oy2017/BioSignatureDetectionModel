"""Score the screen on Ariel's known targets (the MCS set rendered by mcs_testset.py).

Two noise conventions, both reported:
  grid     the study's convention, SNR 15 on each planet's own peak-to-peak amplitude;
  tier2    the mission's: sigma = (5-scale-height H2 modulation) / 7 per target, ExoSim2 shape
           for the host's Teff — an absolute level that does not know the planet's real amplitude.
And for Tier 1: the tier-1 screen (tier_screen.py) on the 7-bin layout at the same absolute sigma.

Reported: accuracy overall, inside vs outside the training box, by host band, by tier-2
transit count (a proxy for how hard the target is for Ariel); the achieved relative SNR
(actual amplitude / sigma) distribution; and the share of targets the clean-data decline
thresholds of trust_detect.py would decline before any mismatch is applied.

Usage: python mcs_eval.py
Writes results/ariel_mcs.txt / .csv
"""
import json, os, sys
import joblib, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, centres, configs, load_split, metrics  # noqa: E402
from noise import sigma_matrix, nsr_shapes, EXOSIM_NPZ  # noqa: E402
from bin_spectra import bin_native  # noqa: E402

BANDS = [("M (<4000 K)", 0, 4000), ("K (4000-5300)", 4000, 5300), ("G (5300-6000)", 5300, 6000), ("F+ (>6000)", 6000, 1e9)]


def exosim_sigma(sig_level, tstar, cen):
    teffs, shapes = nsr_shapes(np.asarray(cen, float), EXOSIM_NPZ)
    node = teffs[np.argmin(np.abs(teffs[None, :] - np.asarray(tstar)[:, None]), axis=1)]
    return sig_level[:, None] * np.vstack([shapes[int(t)] for t in node])


def main():
    P = pd.read_parquet(os.path.join(DATA, "mcs", "mcs_params.parquet"))
    Xn = np.load(os.path.join(DATA, "mcs", "mcs_native.npy")).astype(np.float64)
    ok = np.all(np.isfinite(Xn), axis=1); P = P[ok].reset_index(drop=True); Xn = Xn[ok]
    wl = np.load(os.path.join(DATA, "native_wl.npy")); y = P.label_co.to_numpy(); tst = P["s temperature"].to_numpy()
    best = json.load(open(os.path.join(RESULTS, "ariel_best.json")))["best"]
    pipes = {"ariel": joblib.load(os.path.join(MODELS, f"ariel_{best}.joblib"))}
    for cfg in ("tier2", "tier1"):
        p = os.path.join(MODELS, f"{cfg}_norm_xgb.joblib")
        if os.path.exists(p): pipes[cfg] = joblib.load(p)
    rng = np.random.default_rng(SEED + 303)
    rows, L = [], [f"The screen on {len(P)} known Ariel targets (MCS 2026-05-11), label C/O > 1 drawn as in the grid", "",
                   f"inside the training box on every bulk parameter: {P.in_box.mean():.1%}", ""]

    def report(tag, cfg, X, sig_note):
        f, m = pipes[cfg]["features"], pipes[cfg]["model"]
        p = m.predict_proba(f.transform(X))[:, 1]; c = ((p >= 0.5).astype(int) == y)
        r = dict(case=tag, config=cfg, n=len(y), accuracy=c.mean(), acc_in_box=c[P.in_box].mean(), acc_out_box=c[~P.in_box].mean() if (~P.in_box).any() else np.nan)
        for b, lo, hi in BANDS:
            mb = (tst >= lo) & (tst < hi); r[f"acc_{b.split()[0]}"] = c[mb].mean() if mb.any() else np.nan
        q = np.nanpercentile(P.tier2_transits, [33, 66])
        easy, hard = P.tier2_transits <= q[0], P.tier2_transits > q[1]
        r["acc_easy_targets"], r["acc_hard_targets"] = c[easy].mean(), c[hard].mean()
        rows.append(r)
        L.append(f"{tag:<34}{cfg:<7} acc {c.mean()*100:6.2f}%  in-box {r['acc_in_box']*100:6.2f}%  out-of-box {r['acc_out_box']*100:6.2f}%  "
                 + " ".join(f"{b.split()[0]} {r[f'acc_{b.split()[0]}']*100:5.1f}" for b, _, _ in BANDS)
                 + f"  easy/hard(tier-2 transits) {r['acc_easy_targets']*100:5.1f}/{r['acc_hard_targets']*100:5.1f}  {sig_note}")
        return p

    # grid convention at Tier-3 binning
    cen = centres("ariel"); Xb = bin_native(Xn, wl, np.array(configs()["ariel"]["edges"]))
    Xg = Xb + rng.normal(0, 1, Xb.shape) * sigma_matrix(Xb, tst, cen, SNR, "ariel")
    p_grid = report("grid noise (SNR 15 on own amplitude)", "ariel", Xg, "")
    # mission convention: absolute sigma from the tier-2 requirement, ExoSim2 shape
    sig2 = P.sigma_tier2.to_numpy()
    S = exosim_sigma(sig2, tst, cen); Xt2 = Xb + rng.normal(0, 1, Xb.shape) * S
    amp = np.ptp(Xb, axis=1); rel = amp / np.median(S, axis=1)
    p_t2 = report("tier-2 sigma, ExoSim2 shape, 102 bins", "ariel", Xt2, f"achieved SNR on own amplitude: median {np.median(rel):.1f}, 10-90% {np.percentile(rel,10):.1f}-{np.percentile(rel,90):.1f}")
    if "tier2" in pipes:
        cen2 = centres("tier2"); Xb2 = bin_native(Xn, wl, np.array(configs()["tier2"]["edges"]))
        X2 = Xb2 + rng.normal(0, 1, Xb2.shape) * exosim_sigma(sig2, tst, cen2); report("tier-2 sigma, tier-2 binning (51)", "tier2", X2, "")
    if "tier1" in pipes:
        cen1 = centres("tier1"); Xb1 = bin_native(Xn, wl, np.array(configs()["tier1"]["edges"]))
        X1 = Xb1 + rng.normal(0, 1, Xb1.shape) * exosim_sigma(sig2, tst, cen1); report("tier-1 sigma, tier-1 binning (7)", "tier1", X1, "")

    # how many real targets would the clean-data decline rules reject before any mismatch?
    f, m = pipes["ariel"]["features"], pipes["ariel"]["model"]
    Xtr, _, _ = load_split("train", "ariel"); Ztr = f.transform(Xtr)
    Xc = np.vstack([load_split(t, "ariel")[0] for t in TESTS]); Zc = f.transform(Xc)
    knn = NearestNeighbors(n_neighbors=10).fit(Ztr); mu = Ztr.mean(0); Ci = np.linalg.inv(np.cov(Ztr, rowvar=False) + 1e-3 * np.eye(Ztr.shape[1]))
    def scores(Z, p):
        d = Z - mu
        return {"margin": 1 - np.abs(2 * p - 1), "mahalanobis": np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d)), "knn": knn.kneighbors(Z)[0].mean(1)}
    pc = m.predict_proba(Zc)[:, 1]; S0 = scores(Zc, pc)
    L += ["", "Share of real targets the clean-fixed decline thresholds (10 % of clean grid planets declined) would decline:"]
    for tag, X in (("grid noise", Xg), ("tier-2 sigma", Xt2)):
        Z = f.transform(X); p = m.predict_proba(Z)[:, 1]; Sx = scores(Z, p)
        L.append(f"  {tag:<14}" + "  ".join(f"{k}: {(Sx[k] > np.quantile(S0[k], 0.9)).mean():5.1%}" for k in Sx))
    L += ["", "Caveats: C/O is drawn, not known, for these targets, so this measures the screen's behaviour on the mission's",
          "population of bulk parameters and noise levels, not its accuracy on the real planets' chemistry."]
    open(os.path.join(RESULTS, "ariel_mcs.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "ariel_mcs.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
