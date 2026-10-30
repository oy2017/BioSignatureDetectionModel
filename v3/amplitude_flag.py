"""The amplitude flag through each reduction's own coverage (review of 2026-10-30; rule and expectations F in REVISION_EXPECTATIONS.md).

The first form of the flag compared a real dispersion over the measured bins with a population statistic over all 104 bins. Here the
population is read through the same bins, and every reduction with a measured dispersion gets a ratio, including GJ 1214 b and L 98-59 d,
which have no clear twin and were skipped by accident before. Primary rule: upper threshold, the largest ratio of any clear POP-I planet through
the reduction's measured bins; lower cut, the first percentile of POP-I with its own decks through those bins. Sensitivities, not candidates:
clear 99th percentile, own-deck 99th percentile, the AIRS range, and the full-bin threshold of the first form.

Writes results/jwst_panel/amplitude_flag.{txt,csv} and amplitude_flag_router.csv; caches data/alfnoor_faithful/pop1_clear_binned.npy.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import alfnoor_faithful as AF  # noqa: E402
import forward_model as FM  # noqa: E402
import generate_grid as G  # noqa: E402
import jwst_panel as JP  # noqa: E402
import jwst_to_ariel as J  # noqa: E402
import router_real as RR  # noqa: E402
from bin_spectra import bin_native  # noqa: E402
from muting_router import RETRAINED, verdict  # noqa: E402

G.FILL_GAS = AF.FILL
OUT = "results/jwst_panel"; MOLS = ["CH4", "H2O", "CO2", "NH3"]; BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")


def populations():
    src = AF.DIRS["faithful"]; wl = np.load("data/native_wl.npy"); edges, _ = J.tier3_layout(); P = pd.read_parquet(os.path.join(src, "pop1_params.parquet"))
    Y = np.load(os.path.join(src, "pop1_native.npy")).astype(float); ok = np.isfinite(Y).all(1); f = os.path.join(src, "pop1_clear_binned.npy")
    if os.path.exists(f):
        C = np.load(f)
    else:
        C = np.full((len(P), len(edges) - 1), np.nan)
        for i in np.where(ok)[0]:
            r = P.iloc[i].to_dict(); r.pop("cloud_pressure", None); r = {k: v for k, v in r.items() if k not in ("atm HCN", "atm C2H2")}
            y = FM.spectrum(r, wl, G.GASES, G.FILL_GAS, cloud_pressure=None)
            if y is not None: C[i] = bin_native(np.asarray(y, float)[None, :], wl, edges)[0]
        np.save(f, C)
    good = ok & np.isfinite(C).all(1); own = bin_native(Y[good], wl, edges); mod = P["modulation_5H"].to_numpy(float)[good] * 1e6
    # check against the census's own clear dispersions
    Pm = pd.read_csv(os.path.join(OUT, "muting_population.csv")).set_index("index"); idx = np.where(good)[0]; common = [i for i in idx if i in Pm.index]; pos = {i: k for k, i in enumerate(idx)}
    d = np.array([C[i].std() * 1e6 for i in common]); chk = float(np.nanmedian(np.abs(d / Pm.loc[common, "clear"].to_numpy() - 1)))
    return C[good], own, mod, chk, edges


def main():
    Xc, Xo, mod, chk, edges = populations(); cen = 0.5 * (edges[1:] + edges[:-1]); M = pd.read_parquet("data/mcs/mcs_params.parquet").drop_duplicates("name").set_index("name")
    either = pd.read_csv(os.path.join(OUT, "router_real.csv")); boot = pd.read_csv(os.path.join(OUT, "muting_bootstrap.csv")).set_index("planet"); pr = []
    ratio = lambda X, m: mod / (X[:, m].std(1) * 1e6)
    for pl, d in RR.PANEL.items():
        cov = pd.read_csv(d["coverage"]); meas = ~cov.filled.to_numpy(bool); depth, err = cov.depth.to_numpy(), cov.err.to_numpy(); md = float(M.loc[BASE(pl), "modulation_5H"]) * 1e6
        def real(m):
            s = np.std(depth[m]) * 1e6; n = float(np.sqrt(np.nanmean(err[m] ** 2)) * 1e6); c = np.sqrt(max(s ** 2 - n ** 2, 0.0)); return md / c if c > 0 else np.inf
        air = meas & (cen > 1.95); rc, ro = ratio(Xc, meas), ratio(Xo, meas); rca = ratio(Xc, air) if air.sum() > 2 else np.full(len(Xc), np.nan); roa = ratio(Xo, air) if air.sum() > 2 else np.full(len(Xc), np.nan)
        pr.append(dict(planet=pl, bins=int(meas.sum()), ratio=real(meas), hi_clear_max=rc.max(), hi_clear_p99=np.percentile(rc, 99), hi_own_p99=np.percentile(ro, 99), lo_own_p01=np.percentile(ro, 1),
                       pop_up_rate=100 * np.mean(ro > rc.max()), pop_low_rate=100 * np.mean(ro < np.percentile(ro, 1)), ratio_airs=real(air) if air.sum() > 2 else np.nan, hi_airs=np.nanmax(rca) if air.sum() > 2 else np.nan,
                       lo_airs=np.nanpercentile(roa, 1) if air.sum() > 2 else np.nan, census_beyond=bool(boot.p_beyond_prior.get(pl, 0) > 0.95)))
    F = pd.DataFrame(pr).set_index("planet"); full_hi = ratio(Xc, np.ones(Xc.shape[1], bool)).max(); full_lo = np.percentile(ratio(Xo, np.ones(Xc.shape[1], bool)), 1)
    fin = lambda r: np.isfinite(r)
    RULES = {"mask, clear maximum (the rule)": (lambda f: fin(f.ratio) & (f.ratio > f.hi_clear_max), lambda f: f.ratio < f.lo_own_p01),
             "mask, clear 99th": (lambda f: fin(f.ratio) & (f.ratio > f.hi_clear_p99), lambda f: f.ratio < f.lo_own_p01),
             "mask, own-deck 99th": (lambda f: fin(f.ratio) & (f.ratio > f.hi_own_p99), lambda f: f.ratio < f.lo_own_p01),
             "AIRS range, clear maximum": (lambda f: fin(f.ratio_airs) & (f.ratio_airs > f.hi_airs), lambda f: f.ratio_airs < f.lo_airs),
             "full bins, first form": (lambda f: fin(f.ratio) & (f.ratio > full_hi), lambda f: f.ratio < full_lo)}
    cols, flagged = {}, {}
    base_v = {}
    for pl, d in RR.PANEL.items():
        V0 = pd.read_csv(d["verdicts"]); VR = {k: pd.read_csv(os.path.join(p, f"{pl}_verdicts.csv")) for k, p in RETRAINED.items() if os.path.exists(os.path.join(p, f"{pl}_verdicts.csv"))}
        for m in MOLS:
            lab = d["labels"].get(m); e = either[(either.planet == pl) & (either.molecule == m) & (either.router == "either")]
            base_v[(pl, m)] = dict(lab=lab, base=verdict(V0, "all", m), airs=verdict(V0, "airs_only", m), either=e.verdict.iloc[0] if len(e) else verdict(V0, "all", m), **{k: verdict(VR[k], "all", m) for k in VR})
    for rule, (up, low) in RULES.items():
        U, W = up(F), low(F); flagged[rule] = (list(F.index[U]), list(F.index[W]))
        for ts in ("haze", "hazedeck"):
            for two in (False, True):
                name = f"{rule} -> {ts}" + (", two-sided" if two else "")
                cols[name] = {k: ("split" if (two and W[k[0]]) else (v.get(ts, v["either"]) if U[k[0]] else v["either"])) for k, v in base_v.items()}
    cols["published"] = {k: v["base"] for k, v in base_v.items()}; cols["AIRS-only everywhere"] = {k: v["airs"] for k, v in base_v.items()}; cols["shape flag -> AIRS-only"] = {k: v["either"] for k, v in base_v.items()}
    V = pd.DataFrame(cols); V.index = pd.MultiIndex.from_tuples(V.index, names=["planet", "molecule"]); labs = pd.Series({k: v["lab"] for k, v in base_v.items()}); labs.index = V.index
    lab = labs.notna(); credit = pd.DataFrame({c: (V.loc[lab, c] == labs[lab]).astype(float).where(V.loc[lab, c] != "split", 0.5) for c in V}); V.assign(label=labs).to_csv(os.path.join(OUT, "amplitude_flag_router.csv"))
    plan = np.array([BASE(p) for p in credit.index.get_level_values(0)]); units = sorted(set(plan)); per = credit.groupby(plan).sum().loc[units]; cnt = pd.Series(plan).value_counts().loc[units].to_numpy()
    rng = np.random.default_rng(405); idx = rng.integers(0, len(units), (20000, len(units))); den = cnt[idx].sum(1)
    def vs(c, ref):
        dif = (per[c].to_numpy()[idx].sum(1) - per[ref].to_numpy()[idx].sum(1)) / den * len(credit); return credit[c].sum() - credit[ref].sum(), *np.percentile(dif, [2.5, 97.5])
    L = [f"Amplitude flag through each reduction's coverage (amplitude_flag.py). Population check: clear dispersions match the census within {100 * chk:.2f} % (median). Full-bin thresholds {full_hi:.2f}, {full_lo:.2f}.", "",
         f"{'reduction':<12} {'bins':>4} {'ratio':>7} {'hi max':>7} {'hi p99':>7} {'own p99':>7} {'lo p01':>7} {'pop>hi %':>8} {'AIRS ratio':>10} {'AIRS hi':>8} census  flag"]
    U0 = RULES["mask, clear maximum (the rule)"][0](F); W0 = RULES["mask, clear maximum (the rule)"][1](F)
    for pl, r in F.iterrows():
        L.append(f"{pl:<12} {int(r.bins):>4d} {r.ratio:7.2f} {r.hi_clear_max:7.2f} {r.hi_clear_p99:7.2f} {r.hi_own_p99:7.2f} {r.lo_own_p01:7.2f} {r.pop_up_rate:8.1f} {r.ratio_airs:10.2f} {r.hi_airs:8.2f} {'beyond' if r.census_beyond else '      '}  {'UP' if U0[pl] else ('LOW' if W0[pl] else '')}")
    L += ["", f"POP-I own-deck spectra flagged upper through the panel's masks: median {F.pop_up_rate.median():.1f} %, range {F.pop_up_rate.min():.1f}-{F.pop_up_rate.max():.1f} %", "", "flags under each rule:"]
    for rule, (u, w) in flagged.items():
        cen_ = set(F.index[F.census_beyond]); L.append(f"  {rule:<34} up {len(u):>2} ({len(set(u) & cen_)} of {len(cen_)} census giants); low {w}; up, not beyond by census: {sorted(set(u) - cen_)}; census giants not flagged: {sorted(cen_ - set(u))}")
    L += ["", f"fixed denominator, {len(credit)} labelled cells, {len(units)} planets; right 1, split or withheld 0.5, wrong 0; 95 % planet bootstrap; descriptive", f"{'pipeline':<52} {'score':>6} {'wrong':>5} {'split':>5} {'CH4 p/s/a':>9} {'vs published':>22} {'vs AIRS-only':>22}"]
    f_ = lambda t: f"{t[0]:+5.1f} ({t[1]:+5.1f},{t[2]:+5.1f})"; rows = []
    for c in credit.columns:
        ch4 = V.loc[[(p, "CH4") for p in RR.METHANE if (p, "CH4") in V.index], c].value_counts(); a, b = vs(c, "published"), vs(c, "AIRS-only everywhere")
        L.append(f"{c:<52} {credit[c].sum():6.1f} {int((credit[c] == 0).sum()):>5d} {int((credit[c] == 0.5).sum()):>5d} {ch4.get('present', 0)}/{ch4.get('split', 0)}/{ch4.get('absent', 0):<5d} {f_(a):>22} {f_(b):>22}")
        rows.append(dict(pipeline=c, score=credit[c].sum(), wrong=int((credit[c] == 0).sum()), split=int((credit[c] == 0.5).sum()), vs_published=a[0], vs_pub_lo=a[1], vs_pub_hi=a[2], vs_airs=b[0], vs_airs_lo=b[1], vs_airs_hi=b[2]))
    F.reset_index().to_csv(os.path.join(OUT, "amplitude_flag.csv"), index=False); pd.DataFrame(rows).to_csv(os.path.join(OUT, "amplitude_flag_scores.csv"), index=False)
    txt = "\n".join(L); open(os.path.join(OUT, "amplitude_flag.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
