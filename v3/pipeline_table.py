"""Table 8 and Appendix D of the manuscript: every design on the three tiers of evidence, the amplitude flag per reduction, the flag rates in simulation.

Simulated accuracy: pipeline_sim.py (mean over four learners and four molecules, POP-I at Tier-1 noise, same test cases and noise draw for every design).
Real panel: 26 reductions of 23 planets, the 66 labels the literature decides; right 1, split or withheld one half, wrong 0; differences against the published
classifier with a 95 % bootstrap over planets (reductions of one planet resampled together); descriptive, since the rule was chosen on this panel.
The rule is the one committed in REVISION_EXPECTATIONS.md section F (amplitude_flag.py). The census row uses the muting census of published compositions with the method's own haze-retrained design
and shape flag, and is not simulated. Tier-1 scores come from panel_tier1.py and the mask control from panel_control.py.

Writes results/pipeline_table.{txt,csv,md}.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import router_real as RR  # noqa: E402
from muting_router import RETRAINED, verdict  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel"); MOLS = ["CH4", "H2O", "CO2", "NH3"]; BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
RULE = "mask, clear maximum (the rule) -> haze, two-sided"
SIM = [("clean", "Clean"), ("haze3e7", "Haze 3 × 10⁷"), ("plhaze29", "Confined haze"), ("cloud1e2", "Deck 10² Pa"), ("spots20", "Spots 20 %"), ("white_x2", "Noise × 2")]
THIRD = {"WASP-94Ab", "HIP67522b", "HAT-P-12b"}
SECOND = {"GJ9827d", "WASP-52b", "L98-59d", "GJ1214b", "LTT3780c", "WASP-80b", "HD209458b", "WASP-96b", "WASP-17b"}


def by_set(setname, variant):
    out = {}
    for pl, d in RR.PANEL.items():
        f = d["verdicts"] if setname == "faithful" else os.path.join(RETRAINED[setname], f"{pl}_verdicts.csv")
        V = pd.read_csv(f)
        for m in MOLS:
            out[(pl, m)] = verdict(V, variant, m)
    return pd.Series(out)


def main():
    V = pd.read_csv(os.path.join(OUT, "amplitude_flag_router.csv")).set_index(["planet", "molecule"]); labels = V.label; ps = pd.read_csv(os.path.join(RESULTS, "pipeline_sim.csv"))
    mr = pd.read_csv(os.path.join(OUT, "muting_router.csv")); census = mr[mr.router == "muting -> retrained haze"].set_index(["planet", "molecule"]).verdict
    rows = [("Published classifier, all inputs", "published, all inputs", V["published"]), ("No-optical input everywhere", "no-optical everywhere", by_set("faithful", "no_optical")),
            ("AIRS-only input everywhere", "AIRS-only everywhere", V["AIRS-only everywhere"]), ("Shape flag → AIRS-only input", "shape flag -> AIRS-only", V["shape flag -> AIRS-only"]),
            ("Haze-retrained design everywhere", "retrained (haze) everywhere", by_set("haze", "all")), ("Haze+deck-retrained design everywhere", "retrained (hazedeck) everywhere", by_set("hazedeck", "all")),
            ("The routing method", "pipeline (haze, flag observed), two-sided", V[RULE]), ("Muting census in place of the amplitude flag", None, census)]
    lab = labels.notna(); idx = labels.index[lab]; plan = np.array([BASE(p) for p in idx.get_level_values(0)]); units = sorted(set(plan)); cnt = pd.Series(plan).value_counts().loc[units].to_numpy()
    rng = np.random.default_rng(405); res = rng.integers(0, len(units), (20000, len(units))); den = cnt[res].sum(1)
    credit = {name: (v.reindex(idx) == labels[lab]).astype(float).where(v.reindex(idx) != "split", 0.5) for name, _, v in rows}; ref = credit[rows[0][0]]; per = lambda c: c.groupby(plan).sum().loc[units].to_numpy()
    out, md, ms = [], [], []
    T1 = pd.read_csv(os.path.join(OUT, "panel_tier1.csv")); CTL = pd.read_csv(os.path.join(OUT, "panel_control.csv")).set_index(["planet", "molecule"]).control
    T1NAME = {"Published classifier, all inputs": "published", "No-optical input everywhere": "no-optical", "AIRS-only input everywhere": "AIRS-only", "Shape flag → AIRS-only input": "shape flag", "Haze-retrained design everywhere": "haze-retrained", "Haze+deck-retrained design everywhere": "haze+deck-retrained", "The routing method": "the method", "Muting census in place of the amplitude flag": "census, same haze design"}
    t1 = T1[(T1.precision == "Tier-1") & T1.label.notna()].pivot_table(index=["planet", "molecule"], columns="design", values="credit").reindex(idx); t1j = T1[(T1.precision == "JWST") & T1.label.notna()].pivot_table(index=["planet", "molecule"], columns="design", values="credit").reindex(idx)
    t1ch4 = T1[(T1.precision == "Tier-1") & (T1.molecule == "CH4") & T1.planet.isin(RR.METHANE)].groupby("design").present.sum(); passes = CTL.reindex(idx) == "passes"
    # the census row is like for like with the method: the same haze-retrained design and the shape flag elsewhere, with HIP 67522 b withheld (panel_tier1.py)
    credit[rows[-1][0]] = t1j["census, same haze design"].copy(); census_v = pd.Series(index=idx, dtype=object)
    md.append("| Design | Right / wrong / split or withheld | Score of 66, JWST precision | Gain over published (95 % interval) | Score of 50, labels passing the mask control | Score of 66, Tier-1 noise | Gain over published (95 % interval) | Methane found of 8, JWST / Tier-1 |")
    md.append("| :-- | :-- | --: | :-- | --: | --: | :-- | :-- |")
    nabs = int((labels[lab] == "absent").sum()); nabsc = int(((labels[lab] == "absent") & passes).sum())
    md.append(f"| Calling every label absent | {nabs} / {len(idx) - nabs} / 0 | {nabs:.1f} | {nabs - credit[rows[0][0]].sum():+.1f} | {nabsc:.1f} | {nabs:.1f} | {nabs - t1[T1NAME[rows[0][0]]].sum():+.1f} | 0 / 0 |".replace("-", "−"))
    ms.append("| Design | " + " | ".join(l for _, l in SIM) + " |"); ms.append("| :-- | " + " | ".join("--:" for _ in SIM) + " |")
    for name, sim, v in rows:
        c = credit[name]; r = dict(design=name); cj = t1[T1NAME[name]]; pt1 = cj.groupby(plan).sum().loc[units].to_numpy(); pt0 = t1[T1NAME[rows[0][0]]].groupby(plan).sum().loc[units].to_numpy(); d1 = (pt1[res].sum(1) - pt0[res].sum(1)) / den * len(c); lo1, hi1 = np.percentile(d1, [2.5, 97.5])
        ch4 = v.reindex([(p, "CH4") for p in RR.METHANE]).value_counts()
        # half credit for a withheld label, as on the panel, so that no design gains by withholding
        g = None
        if sim:
            q_ = ps[ps.pipeline == sim]; g = (q_.groupby("case").half_credit if "two-sided" in sim else q_.groupby("case").accuracy).mean()
        for key, l in SIM: r[l] = float(g[key]) if sim and key in g.index else np.nan
        dif = (per(c)[res].sum(1) - per(ref)[res].sum(1)) / den * len(c); lo, hi = np.percentile(dif, [2.5, 97.5])
        first = c[[p not in SECOND | THIRD for p in c.index.get_level_values(0)]]; batch = lambda s: c[[p in s for p in c.index.get_level_values(0)]]
        r.update(score=c.sum(), right=int((c == 1).sum()), wrong=int((c == 0).sum()), split=int((c == .5).sum()), gain=c.sum() - ref.sum(), lo=lo, hi=hi, control=c[passes.to_numpy()].sum(), tier1=cj.sum(), gain_t1=cj.sum() - t1[T1NAME[rows[0][0]]].sum(), lo_t1=lo1, hi_t1=hi1, ch4=f"{ch4.get('present', 0)} / {ch4.get('split', 0)} / {ch4.get('absent', 0)}", ch4_t1=t1ch4.get(T1NAME[name], np.nan), first=first.sum(), n_first=len(first), second=batch(SECOND).sum(), n_second=len(batch(SECOND)), third=batch(THIRD).sum(), n_third=len(batch(THIRD)))
        out.append(r)
        g_ = "" if name == rows[0][0] else f"{r['gain']:+.1f} ({lo:.1f} to {hi:.1f})".replace("-", "−"); g1 = "" if name == rows[0][0] else f"{r['gain_t1']:+.1f} ({lo1:.1f} to {hi1:.1f})".replace("-", "−")
        ch4_j = ch4.get("present", 0) if name != rows[-1][0] else int(round(T1[(T1.precision == "JWST") & (T1.design == T1NAME[name]) & (T1.molecule == "CH4") & T1.planet.isin(RR.METHANE)].present.sum()))
        md.append(f"| {name} | {r['right']} / {r['wrong']} / {r['split']} | {r['score']:.1f} | {g_} | {r['control']:.1f} | {r['tier1']:.1f} | {g1} | {ch4_j} / {r['ch4_t1']:.1f} |")
        if sim: ms.append(f"| {name} | " + " | ".join(f"{r[l]:.1f}" for _, l in SIM) + " |")
    df = pd.DataFrame(out); df.to_csv(os.path.join(RESULTS, "pipeline_table.csv"), index=False)
    # Appendix D: the amplitude flag per reduction, and the flag rates in simulation
    F = pd.read_csv(os.path.join(OUT, "amplitude_flag.csv")).set_index("planet"); C = pd.read_csv(os.path.join(OUT, "muting_census.csv")).set_index("planet"); B = pd.read_csv(os.path.join(OUT, "muting_bootstrap.csv")).set_index("planet")
    d1 = ["| Reduction | Measured bins | Ratio | Upper threshold | Lower cut | Flag | Muting (census) | Most the prior allows | Beyond the prior |", "| :-- | --: | --: | --: | --: | :-- | --: | --: | :-- |"]
    for pl, r in F.iterrows():
        up, low = np.isfinite(r.ratio) and r.ratio > r.hi_clear_max, r.ratio < r.lo_own_p01; mu = C.muting.get(pl, np.nan); mx = C.max_muting_in_prior.get(pl, np.nan); pb = B.p_beyond_prior.get(pl, np.nan)
        bey = "no twin" if not np.isfinite(mx) else ("yes" if pb > 0.95 else ("marginal" if mu > mx else "no"))
        ra = "no dispersion" if not np.isfinite(r.ratio) else f"{r.ratio:.2f}"; mus = "no twin" if not np.isfinite(mx) else ("no dispersion" if not np.isfinite(mu) else f"{mu:.2f}"); mxs = "" if not np.isfinite(mx) else f"{mx:.2f}"
        d1.append(f"| {pl} | {int(r.bins)} | {ra} | {r.hi_clear_max:.2f} | {r.lo_own_p01:.2f} | {'upper' if up else ('lower' if low else 'none')} | {mus} | {mxs} | {bey} |")
    # the selection table: CH4 and H2O targets lost at the 60 % selection
    Sel = pd.read_csv(os.path.join(RESULTS, "pipeline_sim_selection.csv")); Sel = Sel[Sel.slots == 0.6]; SN = {"published, all inputs": "Published classifier, all inputs", "no-optical everywhere": "No-optical input everywhere", "AIRS-only everywhere": "AIRS-only input everywhere", "optical-excess score -> AIRS-only": "Optical-excess score → AIRS-only input", "shape flag -> AIRS-only": "Shape flag → AIRS-only input", "retrained (haze) everywhere": "Haze-retrained design everywhere", "retrained (hazedeck) everywhere": "Haze+deck-retrained design everywhere", "the routing method": "The routing method"}
    CS = [("clean", "Clean"), ("haze3e7", "Haze 3 × 10⁷"), ("plhaze29", "Confined haze"), ("cloud1e2", "Deck 10² Pa"), ("spots20", "Spots 20 %"), ("white_x2", "Noise × 2")]
    sl = ["| Design | " + " | ".join(l for _, l in CS) + " |", "| :-- | " + " | ".join("--:" for _ in CS) + " |"]
    for k, nm in SN.items():
        sl.append(f"| {nm} | " + " | ".join(f"{int(Sel[(Sel.design == k) & (Sel.molecule == 'CH4') & (Sel.case == c)].lost.iloc[0])} / {int(Sel[(Sel.design == k) & (Sel.molecule == 'H2O') & (Sel.case == c)].lost.iloc[0])}" for c, _ in CS) + " |")
    L = ["Designs on the real panel", *md, "", "Designs in simulation", *ms, "", "Targets lost at the 60 % selection, CH4 / H2O", *sl, "", "Table D1", *d1, ""]
    cases = ["clean", "haze2e6", "haze3e7", "hazetop1", "plhaze29", "plhaze28", "spots20", "compound", "cloud1e2", "white_x2", "white_x3", "cia"]; cases = [c for c in cases if c in set(ps.case)]
    s = ps[ps.pipeline == "shape flag -> AIRS-only"].groupby("case")[[c for c in ("shape_flagged", "disagree_flagged", "d1_flagged", "d2_flagged") if c in ps.columns]].mean()
    rates = pd.read_csv(os.path.join(RESULTS, "pipeline_sim_rates.csv")).set_index("case") if os.path.exists(os.path.join(RESULTS, "pipeline_sim_rates.csv")) else None
    d2 = ["| Spectra | Optical-excess score | Input disagreement | First level of the nest | Second level of the nest | Shape flag | Amplitude flag, upper | Amplitude flag, lower |", "| :-- | --: | --: | --: | --: | --: | --: | --: |"]
    NAMES = {"clean": "Clean", "haze2e6": "Column-filling haze, 2 × 10⁶ m⁻³", "haze3e7": "Column-filling haze, 3 × 10⁷ m⁻³", "hazetop1": "Lee-Mie haze below 1 Pa", "plhaze29": "Power-law haze, log₁₀ a ≈ 3.1", "plhaze28": "Power-law haze, log₁₀ a ≈ 4.1", "spots20": "Spots 20 %", "compound": "Spots 20 % + haze 3 × 10⁷ m⁻³", "cloud1e2": "Deck 10² Pa", "white_x2": "Noise × 2", "white_x3": "Noise × 3", "cia": "Collision-induced absorption"}
    if rates is not None and "d1_flagged" in s.columns:
        for c in cases:
            d2.append(f"| {NAMES[c]} | {rates.optical_flag_pct[c]:.0f} | {s.disagree_flagged[c]:.0f} | {s.d1_flagged[c]:.0f} | {s.d2_flagged[c]:.0f} | {s.shape_flagged[c]:.0f} | {rates['observed up %'][c]:.1f} | {rates['observed low %'][c]:.1f} |")
    L += ["Table D2", *d2, "", "Flag rates", s.loc[cases].round(1).to_string(), "", "Batches", df[["design", "first", "n_first", "second", "n_second", "third", "n_third"]].to_string(index=False), ""]
    txt = "\n".join(L); open(os.path.join(RESULTS, "pipeline_table.md"), "w").write(txt + "\n"); open(os.path.join(RESULTS, "pipeline_table.txt"), "w").write(df.round(2).to_string(index=False) + "\n"); print(txt)


if __name__ == "__main__":
    main()
