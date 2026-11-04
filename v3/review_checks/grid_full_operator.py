"""The held-out grids through the FULL JWST-to-Ariel operator (reader's question, review round 3).

On the two held-out grids (results/goyal_test/grid_{local,crossfield}.npz) every design is scored through a shortcut
operator: the reduction's coverage mask with independent Gaussian noise per Ariel bin at the reduction's per-bin error
(round3_learner.run, goyal_test.run_frozen). The real JWST inputs instead go through jwst_to_ariel: native points per
instrument mode, overlap-fraction x inverse-variance binning (bin_segment / combine), fitted inter-mode offsets
(align_offsets, reference = the mode with most points; WASP-39 b: SOSS order 1 and MIRI levelled through the benchmark
model as in wasp39_assemble.convert), bins covered below 0.5 flat-filled, and Tier-1 noise added per bin afterwards.

Here a seeded subset of NMOD source models per reduction and grid is taken back to native resolution (the same helpers
as crossfield_test.grid_for / goyal_test.grid_for; binning them with ML.bin_native reproduces the stored rows exactly),
observed like the reduction's own segments (jwst_to_ariel.sample_clean: the model averaged over each point's interval,
exactly as bin_segment treats the point widths), scattered with each point's JWST error (DRAWS draws), aligned, combined
and filled as the real pipeline does; Tier-1 adds the payload sigma per bin on top. The same subset and draw count go
through the shortcut operator for a like-for-like baseline. Designs: the matched ensemble (C+B-mlp, refitted with the
seeds of round3_learner.run and checked against the stored votes), the routing method and the re-implemented classifier
(panel_tier1.score as in goyal_test.run_frozen).

Usage: OMP_NUM_THREADS=1 python review_checks/grid_full_operator.py [--jobs N] [--stage convert|ensemble|frozen|summary|all]
Writes results/round3/grid_full_operator_{spectra.npz, votes.csv.gz, scores.csv, checks.csv, noise.csv} and grid_full_operator.txt
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS, SEED  # noqa: E402
import crossfield_test as CT  # noqa: E402
import goyal_test as GT  # noqa: E402
import jwst_panel as JP  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import matched_learner as ML  # noqa: E402
import round3_learner as R3  # noqa: E402

OUT = os.path.join(RESULTS, "round3"); PFX = os.path.join(OUT, "grid_full_operator_"); GRIDS = ("local", "crossfield"); OPS = ("shortcut", "full"); PRECS = ("JWST", "Tier-1")
MOLS = ML.MOLS; NMOD = 50; DRAWS = 20; NB = 2000; ROOT = os.path.join(HERE, "data", "jwst", "panel"); CAND = "C+B-mlp"; ENS, RT, PUB = "C+B-mlp", "the method", "published"
EDGES, _ = J.tier3_layout(); CEN = 0.5 * (EDGES[1:] + EDGES[:-1])


# ---------------------------------------------------------------- native source models (the stored grids' own construction)
def native_models(grid, i, j, M, rows):
    """(wavelength grid, native depth spectra for the selected rows, stored-row reproduction max abs diff) of one reduction."""
    batch, pl, labs, x, err, filled, s, md = j; m = M.loc[ML.BASE(pl)]; out = []
    if grid == "local":
        T = GT.table("local"); rp = m.p_radius * 0.08921; tp = float(m["atm temperature"]); gp = 2479.0 * (m.p_mass / 317.8) / rp ** 2
        g0 = GT.GRAV[np.argmin(np.abs(np.log(GT.GRAV) - np.log(gp / 100)))]; q = T[(np.abs(T["T"] - tp) <= 200) & (T.g == g0)]
        if len(q) == 0: q = T[(T["T"] == T["T"].iloc[np.argmin(np.abs(T["T"] - tp))]) & (T.g == g0)]
        q = q.sample(min(GT.N, len(q)), random_state=SEED + 900 + i)
        for k in rows:
            r = q.iloc[k]; c, mmm = GT.chem("local", r.file); w, d = GT.rescale("local", r, mmm, float(m["s radius"]), rp, gp, tp); out.append((w, d, r.file))
    else:
        T = CT.table(); rp = m.p_radius * CT.RE; rs = float(m["s radius"]) * CT.RSUN; tp = float(m["atm temperature"]); gp = 6.674e-8 * m.p_mass * 5.972e27 / rp ** 2; q = T[np.abs(T["T"] - tp) <= 200]
        if len(q) == 0: q = T[T["T"] == T["T"].iloc[np.argmin(np.abs(T["T"] - tp))]]
        q = q.sample(min(CT.N, len(q)), random_state=SEED + 950 + i)
        for k in rows:
            r = q.iloc[k]; w, d, c, mmw = CT.load(r.file); h1 = CT.KB * r["T"] / (mmw * CT.MH * CT.G0); h2 = CT.KB * tp / (mmw * CT.MH * gp); z = np.sqrt(d) * CT.RS0 - CT.RP0
            out.append((w, ((z * h2 / h1 + rp) / rs) ** 2, r.file))
    return out


# ---------------------------------------------------------------- the full operator
def segments_for(batch, pl):
    """(segments, reference mode, benchmark model or None) exactly as the real pipeline chose them."""
    if pl == "WASP-39b":
        import wasp39_assemble as WA
        return WA.segments(), "SOSS-o1", WA.benchmark_model()
    S, _ = (JP.segments if batch == "panel" else JP.fourth_segments)(ROOT, pl)
    return S, max(S, key=lambda k: len(S[k]["wl"])), None


def convert(obs, ref, model):
    """jwst_panel.run_planet / fourth_batch.convert / wasp39_assemble.convert on one realization of the segments."""
    off = J.align_offsets(obs, ref)
    if model is not None and "MIRI" in obs and len(obs) > 1:
        off, _ = J.align_to_model(obs, {k: v for k, v in off.items() if k != "MIRI"}, model[0], model[1], ["MIRI"])
    d, e, cov = J.combine(obs, EDGES, off)
    filled = ~np.isfinite(d) | (cov.covered.to_numpy() < 0.5); d_f = d.copy(); d_f[filled] = np.nanmean(d[~filled])
    return d_f, e, filled, off


def lag_corr(N, meas, lags=(1, 2, 5)):
    """Mean Pearson correlation of the noise between measured bins `lag` apart (over rows), and the number of pairs."""
    C = np.corrcoef(N[:, meas].T); idx = np.where(meas)[0]; out = {}
    for l in lags:
        pairs = [(a, b) for a in range(len(idx)) for b in range(len(idx)) if idx[b] - idx[a] == l]
        out[l] = (float(np.mean([C[a, b] for a, b in pairs])) if pairs else np.nan, len(pairs))
    return out


def convert_one(i, j, M, G):
    """All four spectra sets (2 operators x 2 precisions) of both grids for one reduction, plus the checks."""
    t0 = time.time(); batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; e_m = np.nan_to_num(err[meas]); e_t = np.sqrt(e_m ** 2 + s[meas] ** 2)
    S, ref, model = segments_for(batch, pl); off_real = J.align_offsets(S, ref)
    spec, checks, noise = {}, [], []
    for gi, grid in enumerate(GRIDS):
        n = len(G[grid][pl]); rows = np.sort(np.random.default_rng(SEED + 8100 + i).choice(n, min(NMOD, n), replace=False)); nat = native_models(grid, i, j, M, rows)
        B = np.array([ML.bin_native(d[None, :], w, EDGES)[0] for w, d, _ in nat]); rep = float(np.abs(B - G[grid][pl][rows]).max())
        # full operator: JWST point noise through alignment, binning and filling; Tier-1 sigma added per bin afterwards
        rJ = np.random.default_rng(SEED + 8200 + 10 * i + gi); rT = np.random.default_rng(SEED + 8300 + 10 * i + gi); XF = np.zeros((len(rows) * DRAWS, 104)); EF = np.zeros((len(rows) * DRAWS, 104)); offs = []; bad = 0
        for k, (w, d, _) in enumerate(nat):
            clean = J.sample_clean(w, d, S)
            for q in range(DRAWS):
                obs = {m_: dict(s_, depth=s_["depth"] + rJ.normal(0, 1, len(s_["depth"])) * s_["err"]) for m_, s_ in clean.items()}
                d_f, e_f, f2, off = convert(obs, ref, model); bad += int(not np.array_equal(f2, filled)); XF[k * DRAWS + q] = d_f; EF[k * DRAWS + q] = np.where(f2, np.nan, e_f); offs.append([off[m_] for m_ in S])
        XFT = XF + rT.normal(0, 1, XF.shape) * s[None, :]
        # shortcut operator on the same subset: flat fill at the clean mean, independent noise per measured bin at the stored error
        rS = np.random.default_rng(SEED + 8400 + 10 * i + gi); rS2 = np.random.default_rng(SEED + 8405 + 10 * i + gi)
        XS = np.repeat(G[grid][pl][rows].astype(float), DRAWS, axis=0); XS[:, filled] = XS[:, meas].mean(1, keepdims=True); XS[:, meas] += rS.normal(0, 1, (len(XS), meas.sum())) * e_m
        XST = np.repeat(G[grid][pl][rows].astype(float), DRAWS, axis=0); XST[:, filled] = XST[:, meas].mean(1, keepdims=True); XST[:, meas] += rS2.normal(0, 1, (len(XST), meas.sum())) * e_t; XST[:, filled] += rS2.normal(0, 1, (len(XST), filled.sum())) * s[filled]
        spec[grid] = {"rows": rows, "files": [f for _, _, f in nat], ("shortcut", "JWST"): XS, ("shortcut", "Tier-1"): XST, ("full", "JWST"): XF, ("full", "Tier-1"): XFT}
        # checks and the noise structure (JWST precision; noise = draw minus the per-model mean over draws)
        ef = np.nanmean(EF, axis=0); e_dev = float(np.nanmax(np.abs(ef[meas] - e_m) / e_m)); NF = XF - XF.reshape(len(rows), DRAWS, 104).mean(1).repeat(DRAWS, axis=0); NS = XS - XS.reshape(len(rows), DRAWS, 104).mean(1).repeat(DRAWS, axis=0)
        cf, cs = lag_corr(NF, meas), lag_corr(NS, meas); offs = np.array(offs)
        checks.append(dict(grid=grid, planet=pl, batch=batch, n_models=len(rows), bin_reproduction_maxabs=rep, mask_mismatch_draws=bad, err_vector_max_rel_dev=e_dev, n_modes=len(S), modes="+".join(S), reference=ref,
                           n_fitted_offsets=int(sum(1 for m_ in S if m_ != ref and not (m_ in ("NRS1", "NRS2") and ref in ("NRS1", "NRS2")))), real_offsets_ppm="/".join(f"{off_real[m_] * 1e6:+.0f}" for m_ in S),
                           offset_scatter_ppm="/".join(f"{v:.0f}" for v in offs.std(0) * 1e6), rms_full_noise_ppm=float(np.sqrt(np.mean(NF[:, meas] ** 2) * DRAWS / (DRAWS - 1))) * 1e6, rms_shortcut_noise_ppm=float(np.sqrt(np.mean(NS[:, meas] ** 2) * DRAWS / (DRAWS - 1))) * 1e6,
                           rms_stored_err_ppm=float(np.sqrt(np.mean(e_m ** 2))) * 1e6, seconds=round(time.time() - t0, 1)))
        noise.append(dict(grid=grid, planet=pl, n_modes=len(S), **{f"{op}_lag{l}": c[l][0] for op, c in (("full", cf), ("shortcut", cs)) for l in (1, 2, 5)}, **{f"pairs_lag{l}": cf[l][1] for l in (1, 2, 5)}))
    print(pl, f"{time.time() - t0:.0f}s", flush=True); return pl, spec, checks, noise


def stage_convert(a, L, G):
    M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name")
    R = Parallel(n_jobs=a.jobs, verbose=5)(delayed(convert_one)(i, j, M, G) for i, j in enumerate(L))
    np.savez_compressed(PFX + "spectra.npz", **{f"{g}|{pl}|rows": sp[g]["rows"] for pl, sp, _, _ in R for g in GRIDS}, **{f"{g}|{pl}|{op}|{p}": sp[g][(op, p)].astype(np.float32) for pl, sp, _, _ in R for g in GRIDS for op in OPS for p in PRECS})
    pd.DataFrame([c for _, _, cs, _ in R for c in cs]).to_csv(PFX + "checks.csv", index=False); pd.DataFrame([c for _, _, _, ns in R for c in ns]).to_csv(PFX + "noise.csv", index=False)


def load_spectra():
    Z = np.load(PFX + "spectra.npz"); return {k: Z[k] for k in Z.files}


def agg(grid, op, batch, pl, prec, design, rows, pr, ab, prob=None):
    """Per-row means over the draws of present / absent (and probability)."""
    n = len(rows); P = pr.reshape(n, DRAWS).mean(1); A = ab.reshape(n, DRAWS).mean(1); Q = prob.reshape(n, DRAWS).mean(1) if prob is not None else np.full(n, np.nan)
    return pd.DataFrame(dict(grid=grid, operator=op, batch=batch, planet=pl, precision=prec, design=design, row=rows, present=P, absent=A, prob=Q))


# ---------------------------------------------------------------- the matched ensemble, refitted with the published seeds
def ensemble_one(i, j, prec, X, P, G, Z):
    batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; e = np.nan_to_num(err[meas]); e = np.sqrt(e ** 2 + s[meas] ** 2) if prec == "Tier-1" else e; noise = float(np.sqrt(np.mean(e ** 2))); seed = SEED + 5000 + 10 * i + (0 if prec == "JWST" else 3)
    t0 = time.time(); decide, t = R3.fit_mlp(X, P, meas, e, md, noise, seed); out, rep = [], []
    for grid in GRIDS:
        # reproduction of the stored votes: the stored grid through the shortcut with round3_learner.run's own draw
        r2 = np.random.default_rng(SEED + 1700 + i + (0 if prec == "JWST" else 5000)); Te = G[grid][pl].astype(float).copy(); Te[:, meas] += r2.normal(0, 1, (len(Te), meas.sum())) * e; res = decide(ML.features(Te, meas, noise, True))
        rep.append(dict(grid=grid, planet=pl, precision=prec, **{f"prob_{m}": res[m][2] for m in MOLS}, **{f"present_{m}": res[m][0] for m in MOLS}, **{f"absent_{m}": res[m][1] for m in MOLS}))
        rows = Z[f"{grid}|{pl}|rows"]
        for op in OPS:
            res = decide(ML.features(Z[f"{grid}|{pl}|{op}|{prec}"].astype(float), meas, noise, True))
            for m in MOLS: out.append(agg(grid, op, batch, pl, prec, ENS, rows, res[m][0], res[m][1], res[m][2]).assign(molecule=m))
    print(pl, prec, f"{time.time() - t0:.0f}s", flush=True); return pd.concat(out), rep


def stage_ensemble(a, L, G):
    X, P = R3.population(CAND); Z = load_spectra(); print("population", X.shape, flush=True)
    R = Parallel(n_jobs=a.jobs, verbose=5)(delayed(ensemble_one)(i, j, prec, X, P, G, Z) for prec in PRECS for i, j in enumerate(L))
    pd.concat([r[0] for r in R]).to_csv(PFX + "votes_ensemble.csv.gz", index=False)
    # compare with the stored votes
    rows = []
    for grid in GRIDS:
        V = pd.read_csv(os.path.join(OUT, f"votes_{CAND}_{grid}.csv.gz"))
        for r in (q for res in R for q in res[1] if q["grid"] == grid):
            v = V[(V.planet == r["planet"]) & (V.precision == r["precision"])]
            for m in MOLS:
                vm = v[v.molecule == m].sort_values("row"); rows.append(dict(grid=grid, planet=r["planet"], precision=r["precision"], molecule=m, n=len(vm), max_abs_prob_diff=float(np.abs(vm.prob.to_numpy() - r[f"prob_{m}"]).max()),
                                                                            vote_mismatch=int((vm.present.to_numpy() != r[f"present_{m}"]).sum() + (vm.absent.to_numpy() != r[f"absent_{m}"]).sum())))
    pd.DataFrame(rows).to_csv(PFX + "ensemble_reproduction.csv", index=False)


# ---------------------------------------------------------------- the routing method and the re-implemented classifier
_CTX = None


def frozen_one(i, j, Z, F, Fl):
    global _CTX
    import panel_tier1 as PT
    if _CTX is None: _CTX = PT.Context()
    t0 = time.time(); batch, pl, labs, x, err, filled, s, md = j; meas = ~filled; hi, lo = (F.hi_clear_max[pl], F.lo_own_p01[pl]) if batch == "panel" else (Fl.hi[pl], Fl.lo[pl])
    j2 = float(np.nanmean(err[meas] ** 2)) * 1e12; t2 = float(np.mean(s[meas] ** 2)) * 1e12; out = []
    for grid in GRIDS:
        rows = Z[f"{grid}|{pl}|rows"]
        for prec, noise2 in (("JWST", j2), ("Tier-1", j2 + t2)):
            for op in OPS:
                res = PT.score(_CTX, Z[f"{grid}|{pl}|{op}|{prec}"].astype(float), meas, ML.BASE(pl), noise2, hi, lo)
                for m in MOLS:
                    votes, low, up = res[m]
                    for d in (PUB, RT):
                        pr = votes[d] >= 3; ab = votes[d] <= 1
                        if d == RT: pr = pr & ~low; ab = ab & ~low
                        out.append(agg(grid, op, batch, pl, prec, d, rows, pr.astype(float), ab.astype(float)).assign(molecule=m))
    print(pl, f"{time.time() - t0:.0f}s", flush=True); return pd.concat(out)


def stage_frozen(a, L):
    Z = load_spectra(); F = pd.read_csv(os.path.join(RESULTS, "jwst_panel", "amplitude_flag.csv")).set_index("planet"); Fl = pd.read_csv(os.path.join(RESULTS, "jwst_panel_fourth", "fourth_batch_flags.csv")).set_index("planet")
    R = Parallel(n_jobs=a.jobs, verbose=5)(delayed(frozen_one)(i, j, Z, F, Fl) for i, j in enumerate(L))
    pd.concat(R).to_csv(PFX + "votes_frozen.csv.gz", index=False)


# ---------------------------------------------------------------- scores, intervals, report
def balanced(W, cols, w=None):
    """Balanced credit of each column: mean of the pooled credit on truly present and on truly absent labels. w: replicate weights per unit (R x U)."""
    g = W.groupby(["u", "y"])[cols].agg(["sum", "count"]); units = sorted(W.u.unique()); ui = {u: k for k, u in enumerate(units)}
    out = {}
    for c in cols:
        S = np.zeros((2, 2, len(units)))   # (y, sum|count, unit)
        for (u, y), r in g[c].iterrows(): S[int(y), 0, ui[u]] = r["sum"]; S[int(y), 1, ui[u]] = r["count"]
        ww = np.ones((1, len(units))) if w is None else w
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = 0.5 * ((ww @ S[1, 0]) / (ww @ S[1, 1]) + (ww @ S[0, 0]) / (ww @ S[0, 1]))
    return out


def stage_summary(a, L):
    V = pd.concat([pd.read_csv(PFX + "votes_ensemble.csv.gz"), pd.read_csv(PFX + "votes_frozen.csv.gz")]); C = pd.read_csv(PFX + "checks.csv"); N = pd.read_csv(PFX + "noise.csv"); E = pd.read_csv(PFX + "ensemble_reproduction.csv")
    Y = pd.concat([pd.read_csv(os.path.join(RESULTS, "goyal_test", f"truth_{g}.csv")).assign(grid=g) for g in GRIDS])
    Yl = Y.melt(id_vars=["grid", "planet", "row", "file"], value_vars=[f"v_{m}" for m in MOLS], var_name="molecule", value_name="v"); Yl["molecule"] = Yl.molecule.str[2:]
    A = V.merge(Yl, on=["grid", "planet", "row", "molecule"]); assert len(A) == len(V); A["y"] = A.v > -4; A["c"] = np.where(A.y, A.present, A.absent) + 0.5 * (1 - A.present - A.absent); A["u"] = A.planet.map(ML.BASE); A["fp"] = (~A.y) * A.present; A["split"] = 1 - A.present - A.absent
    A.to_csv(PFX + "rows.csv.gz", index=False)
    designs = [ENS, RT, PUB]; cols = [f"{d}|{op}" for d in designs for op in OPS]; rng = np.random.default_rng(4051); scores, lines = [], []
    L_ = [f"The held-out grids through the full JWST-to-Ariel operator against the shortcut (mask + independent per-bin noise): {NMOD} source models per reduction and grid (seeded subset), {DRAWS} noise draws per model and precision.",
          f"Balanced credit = mean of the pooled credit on truly present (v > -4) and truly absent labels; right 1, split or withheld 0.5, wrong 0; draws averaged per label. 95 % planet-bootstrap percentile intervals ({NB} replicates; planets = reductions with -alt/-hu/-hm stripped, resampled with replacement; differences paired within replicate).",
          f"Full operator: native source model averaged over each JWST point's interval (jwst_to_ariel.sample_clean), Gaussian noise per point at its error, align_offsets with the pipeline's reference mode (most points; WASP-39 b: SOSS order 1 with MIRI levelled on the benchmark model), combine (overlap fraction x inverse variance), bins covered < 0.5 flat-filled at the measured mean; Tier-1: the payload sigma per bin added to the converted spectrum. The ensemble is refitted with round3_learner's seeds (checked below); routing and the re-implemented classifier use panel_tier1.score as goyal_test.run_frozen does.", ""]
    L_ += ["== 1. Reproduction checks",
           f"  stored grid rows from the native source models (ML.bin_native on the model's own wavelength grid): max |diff| over all {int(C.n_models.sum())} rows x 104 bins = {C.bin_reproduction_maxabs.max():.2e} (stored as float32; 0 = identical)",
           f"  coverage mask through the full operator identical to the stored mask in all draws: {int(C.mask_mismatch_draws.sum())} mismatching draws of {int(C.n_models.sum()) * DRAWS}; combine's error vector vs stored err: max relative deviation {C.err_vector_max_rel_dev.max():.1e}",
           f"  ensemble vs stored votes_{CAND}_<grid>.csv.gz (stored grid, round3_learner.run's own draw): max |prob diff| {E.max_abs_prob_diff.max():.2e} (stored probabilities are rounded to 4 decimals), vote mismatches {int(E.vote_mismatch.sum())} of {int(2 * E.n.sum())} present/absent entries over {len(E)} planet x precision x molecule x grid cells",
           f"  reductions: {len(C.planet.unique())}; with more than one instrument mode: {int((C[C.grid == 'local'].n_modes > 1).sum())}; with at least one fitted offset (non-reference mode, NRS1+NRS2 counted once): {int((C[C.grid == 'local'].n_fitted_offsets > 0).sum())}",
           f"  rms noise on the measured bins (ppm), median over reductions: full {C.rms_full_noise_ppm.median():.0f}, shortcut {C.rms_shortcut_noise_ppm.median():.0f}, stored err {C.rms_stored_err_ppm.median():.0f}", ""]
    for grid in GRIDS:
        for prec in PRECS:
            q = A[(A.grid == grid) & (A.precision == prec)]; W = q.pivot_table(index=["planet", "row", "molecule"], columns=["design", "operator"], values="c"); W.columns = [f"{d}|{o}" for d, o in W.columns]; W = W.reset_index()
            W = W.merge(q[(q.design == ENS) & (q.operator == "full")][["planet", "row", "molecule", "y", "u"]], on=["planet", "row", "molecule"]); assert W[cols].notna().all().all()
            units = sorted(W.u.unique()); ix = rng.integers(0, len(units), (NB, len(units))); w = np.zeros((NB, len(units))); np.add.at(w, (np.repeat(np.arange(NB), len(units)), ix.ravel()), 1)
            pt = balanced(W, cols); bt = balanced(W, cols, w); ci = lambda d: tuple(np.percentile(d[np.isfinite(d)], [2.5, 97.5]))
            fpr = {c: float(q[(q.design == c.split("|")[0]) & (q.operator == c.split("|")[1]) & ~q.y].present.mean()) for c in cols}; spl = {c: float(q[(q.design == c.split("|")[0]) & (q.operator == c.split("|")[1])].split.mean()) for c in cols}
            lines.append(f"== {grid} grid, {prec}: {len(W)} labels ({int(W.y.sum())} present, {int((~W.y).sum())} absent) on {W.planet.nunique()} reductions of {len(units)} planets")
            lines.append(f"  {'design':<12} {'shortcut':>9} {'full':>9} {'full-shortcut':>14} {'95 % interval':>18}   {'false-present rate (shortcut / full)':>36}  {'split or withheld (shortcut / full)':>36}")
            for d in designs:
                dd = bt[f"{d}|full"] - bt[f"{d}|shortcut"]; lo, hi = ci(dd)
                lines.append(f"  {d:<12} {pt[f'{d}|shortcut'][0]:9.3f} {pt[f'{d}|full'][0]:9.3f} {pt[f'{d}|full'][0] - pt[f'{d}|shortcut'][0]:+14.3f} ({lo:+.3f},{hi:+.3f})   {fpr[f'{d}|shortcut']:17.3f} / {fpr[f'{d}|full']:<17.3f} {spl[f'{d}|shortcut']:17.3f} / {spl[f'{d}|full']:<17.3f}")
                for op in OPS: scores.append(dict(grid=grid, precision=prec, design=d, operator=op, balanced_credit=pt[f"{d}|{op}"][0], credit_present=float(W[W.y][f"{d}|{op}"].mean()), credit_absent=float(W[~W.y][f"{d}|{op}"].mean()), false_present_rate=fpr[f"{d}|{op}"], split_rate=spl[f"{d}|{op}"], n_labels=len(W), full_minus_shortcut=pt[f"{d}|full"][0] - pt[f"{d}|shortcut"][0], full_minus_shortcut_lo=lo, full_minus_shortcut_hi=hi))
            for op in OPS:
                dd = bt[f"{ENS}|{op}"] - bt[f"{RT}|{op}"]; lo, hi = ci(dd); d2 = bt[f"{ENS}|{op}"] - bt[f"{PUB}|{op}"]; lo2, hi2 = ci(d2)
                lines.append(f"  ensemble minus routing, {op:<8}: {pt[f'{ENS}|{op}'][0] - pt[f'{RT}|{op}'][0]:+.3f} ({lo:+.3f},{hi:+.3f})   ensemble minus re-implemented: {pt[f'{ENS}|{op}'][0] - pt[f'{PUB}|{op}'][0]:+.3f} ({lo2:+.3f},{hi2:+.3f})")
                scores.append(dict(grid=grid, precision=prec, design="ensemble minus routing", operator=op, balanced_credit=pt[f"{ENS}|{op}"][0] - pt[f"{RT}|{op}"][0], full_minus_shortcut_lo=lo, full_minus_shortcut_hi=hi, n_labels=len(W)))
            dg = (bt[f"{ENS}|full"] - bt[f"{RT}|full"]) - (bt[f"{ENS}|shortcut"] - bt[f"{RT}|shortcut"]); lo, hi = ci(dg)
            lines.append(f"  change of the gain, full minus shortcut: {(pt[f'{ENS}|full'][0] - pt[f'{RT}|full'][0]) - (pt[f'{ENS}|shortcut'][0] - pt[f'{RT}|shortcut'][0]):+.3f} ({lo:+.3f},{hi:+.3f})")
            # per gas
            for m in MOLS:
                Wm = W[W.molecule == m]; pm = balanced(Wm, cols); lines.append(f"    {m:<4} " + "  ".join(f"{d} {pm[f'{d}|shortcut'][0]:.3f}->{pm[f'{d}|full'][0]:.3f}" for d in designs) + f"   gain {pm[f'{ENS}|shortcut'][0] - pm[f'{RT}|shortcut'][0]:+.3f}->{pm[f'{ENS}|full'][0] - pm[f'{RT}|full'][0]:+.3f}")
            lines.append("")
    L_ += ["== 2. Scores (balanced credit) per grid x precision, shortcut operator vs full operator on the same subset"] + lines
    # noise structure
    nl = N[N.grid == "local"]; multi = nl[nl.n_modes > 1]
    L_ += ["== 3. Noise structure of the full operator at JWST precision (noise = draw minus the per-model mean over draws; correlation over the 1,000 rows of each reduction and grid, averaged over pairs of measured bins at the lag)",
           f"  lag 1: full operator median {N.full_lag1.median():+.3f} (min {N.full_lag1.min():+.3f}, max {N.full_lag1.max():+.3f}; pair-weighted mean {np.average(N.full_lag1, weights=N.pairs_lag1):+.3f}); shortcut median {N.shortcut_lag1.median():+.3f} (min {N.shortcut_lag1.min():+.3f}, max {N.shortcut_lag1.max():+.3f})",
           f"  lag 2: full median {N.full_lag2.median():+.3f} (max {N.full_lag2.max():+.3f}); lag 5: full median {N.full_lag5.median():+.3f} (max {N.full_lag5.max():+.3f}); shortcut lag 2 / 5 medians {N.shortcut_lag2.median():+.3f} / {N.shortcut_lag5.median():+.3f}",
           f"  reductions with lag-1 correlation above 0.1 (local grid): {int((nl.full_lag1 > 0.1).sum())} of {len(nl)}; above 0.3: {int((nl.full_lag1 > 0.3).sum())}; single-mode reductions median {nl[nl.n_modes == 1].full_lag1.median():+.3f}, multi-mode median {multi.full_lag1.median():+.3f}",
           "  per reduction (local grid): " + ", ".join(f"{r.planet} {r.full_lag1:+.2f}" for r in nl.sort_values("full_lag1", ascending=False).itertuples()),
           f"  fitted-offset scatter across draws (ppm, modes in order; local grid): " + "; ".join(f"{r.planet} [{r.modes}] {r.offset_scatter_ppm}" for r in C[(C.grid == 'local') & (C.n_fitted_offsets > 0)].itertuples()), ""]
    L_ += ["== 4. Per-reduction checks (local grid)", C[C.grid == "local"][["planet", "n_modes", "modes", "reference", "n_fitted_offsets", "real_offsets_ppm", "bin_reproduction_maxabs", "mask_mismatch_draws", "err_vector_max_rel_dev", "rms_full_noise_ppm", "rms_shortcut_noise_ppm", "rms_stored_err_ppm"]].to_string(index=False), ""]
    L_ += ["== 5. Ensemble reproduction, worst cells", E.sort_values("max_abs_prob_diff", ascending=False).head(8).to_string(index=False)]
    pd.DataFrame(scores).to_csv(PFX + "scores.csv", index=False); txt = "\n".join(L_); open(os.path.join(OUT, "grid_full_operator.txt"), "w").write(txt + "\n"); print(txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--stage", default="all", choices=["convert", "ensemble", "frozen", "summary", "all"]); a = ap.parse_args()
    L = ML.jobs_list(); G = {g: dict(np.load(os.path.join(RESULTS, "goyal_test", f"grid_{g}.npz"))) for g in GRIDS}; t0 = time.time()
    if a.stage in ("convert", "all"): stage_convert(a, L, G); print("convert done", f"{time.time() - t0:.0f}s", flush=True)
    if a.stage in ("ensemble", "all"): stage_ensemble(a, L, G); print("ensemble done", f"{time.time() - t0:.0f}s", flush=True)
    if a.stage in ("frozen", "all"): stage_frozen(a, L); print("frozen done", f"{time.time() - t0:.0f}s", flush=True)
    if a.stage in ("summary", "all"): stage_summary(a, L)
    print("total", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
