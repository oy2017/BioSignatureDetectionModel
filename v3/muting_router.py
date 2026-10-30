"""A prior-extension router on the real JWST panel: a planet whose spectrum is muted beyond anything the training prior can
draw at its bulk parameters (the muting census, jwst_muting_census.py / muting_bootstrap.py) is sent to the design retrained
with aerosols beyond the prior (alfnoor_retrain.py: hazed, or hazed and deep-deck training planets); every other planet keeps
the published design. Compared with no routing, with the either-flag routing to the AIRS-only input (router_real.py), and
with the two combined (a cell the either flag routes takes the AIRS-only verdict; the remaining cells of a muted planet take
the retrained verdict). Scored against the literature labels as router_real.py scores them.

Usage: python muting_router.py   -> results/jwst_panel/muting_router.{txt,csv}
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import jwst_panel as JP  # noqa: E402
import router_real as RR  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel"); MOLS = ["CH4", "H2O", "CO2", "NH3"]; LEARNERS = ["KNN", "MLP", "RFC", "SVC"]
RETRAINED = {"haze": os.path.join(RESULTS, "jwst_panel_haze"), "hazedeck": os.path.join(RESULTS, "jwst_panel_hazedeck"), "union": os.path.join(RESULTS, "jwst_panel_union")}


def verdict(V, variant, m):
    v = V[np.isclose(V.threshold.astype(float), 1e-4) & (V.variant == variant) & (V.molecule == m)]
    n = int(v.present.sum()); return JP.P if n >= 3 else (JP.A if n <= 1 else "split")


def main():
    census = pd.read_csv(os.path.join(OUT, "muting_census.csv")).set_index("planet")
    boot = pd.read_csv(os.path.join(OUT, "muting_bootstrap.csv")).set_index("planet")
    either = pd.read_csv(os.path.join(OUT, "router_real.csv"))
    rows, flags = [], {}
    for pl, d in RR.PANEL.items():
        if not os.path.exists(d["verdicts"]):
            continue
        V0 = pd.read_csv(d["verdicts"]); labs = d["labels"]
        mu = float(census.muting.get(pl, np.nan)); mx = float(census.max_muting_in_prior.get(pl, np.nan)); pb = float(boot.p_beyond_prior.get(pl, np.nan))
        f_mut = bool(np.isfinite(mu) and np.isfinite(mx) and mu > mx)            # point estimate beyond the prior's most muted twin
        f_mut_strict = bool(np.isfinite(pb) and pb > 0.95)                        # beyond it in > 95 % of resamples
        flags[pl] = (mu, mx, pb, f_mut, f_mut_strict)
        VR = {k: pd.read_csv(os.path.join(p, f"{pl}_verdicts.csv")) for k, p in RETRAINED.items() if os.path.exists(os.path.join(p, f"{pl}_verdicts.csv"))}
        for m in MOLS:
            base = verdict(V0, "all", m); lab = labs.get(m)
            e = either[(either.planet == pl) & (either.molecule == m) & (either.router == "either")]
            v_either = e.verdict.iloc[0] if len(e) else base; n_either = int(e.flagged_learners.iloc[0]) if len(e) else 0
            out = {"none": base, "either -> AIRS-only": v_either}
            for k in RETRAINED:
                vr = verdict(VR[k], "all", m) if k in VR else None
                out[f"muting -> retrained {k}"] = (vr if (f_mut and vr is not None) else base)
                out[f"muting (strict) -> retrained {k}"] = (vr if (f_mut_strict and vr is not None) else base)
                out[f"either, then muting -> retrained {k}"] = (v_either if n_either > 0 else (vr if (f_mut and vr is not None) else base))
            for k in ("hazedeck", "union"):
                out[f"retrained {k} everywhere"] = verdict(VR[k], "all", m) if k in VR else base
                # census first: a muted planet takes the retrained verdict, every other planet the either-flag routing (needs the census)
                out[f"muting -> retrained {k}, else either -> AIRS-only"] = (verdict(VR[k], "all", m) if (f_mut and k in VR) else v_either)
                # no census: the retrained design everywhere, with the either flag routing to its own AIRS-only input (pipeline-computable)
                if k in VR:
                    out[f"retrained {k} everywhere, either -> its AIRS-only"] = verdict(VR[k], "airs_only", m) if n_either > 0 else verdict(VR[k], "all", m)
                else:
                    out[f"retrained {k} everywhere, either -> its AIRS-only"] = v_either
            for router, v in out.items():
                rows.append(dict(planet=pl, molecule=m, router=router, verdict=v, label=lab, muting_flag=f_mut, muting_flag_strict=f_mut_strict,
                                 retrained_available="hazedeck" in VR, correct=None if lab is None or v == "split" else bool(v == lab)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "muting_router.csv"), index=False)
    giants = [p for p in flags if p in ("HAT-P-26b", "WASP-107b", "WASP-39b", "GJ3470b", "WASP-166b", "WASP-17b", "WASP-80b", "HD209458b", "WASP-96b", "WASP-52b")]
    L = [f"Prior-extension routing on the real panel ({df.planet.nunique()} reductions): a planet muted beyond the training prior's most muted twin is sent to the retrained design",
         f"  muting flag fires on {sum(f[3] for f in flags.values())} reductions (point estimate), {sum(f[4] for f in flags.values())} with P(beyond prior) > 0.95; retrained verdicts exist for {int(df.groupby('planet').retrained_available.first().sum())} of {df.planet.nunique()}", ""]
    L.append(f"{'router':<58} {'scored correct':>15} {'on muting-flagged planets':>26} {'on the giants':>14} {'methane present/split/absent':>30}")
    for r in df.router.unique():
        d = df[df.router == r]; s = d[d.correct.notna()]; sf = s[s.muting_flag]; sg = s[s.planet.isin(giants)]
        ch4 = d[(d.molecule == "CH4") & d.planet.isin(RR.METHANE)]
        L.append(f"{r:<58} {int(s.correct.sum()):>7}/{len(s):<7} {int(sf.correct.sum()):>12}/{len(sf):<13} {int(sg.correct.sum()):>6}/{len(sg):<7} {int((ch4.verdict == JP.P).sum()):>10} / {int((ch4.verdict == 'split').sum())} / {int((ch4.verdict == JP.A).sum())}")
    L.append(""); L.append("per planet: muting, max in prior, P(beyond prior), flag; labels that change between 'none' and 'muting -> retrained hazedeck' [label]")
    for pl, (mu, mx, pb, f, fs) in flags.items():
        ch = []
        for m in MOLS:
            a = df[(df.planet == pl) & (df.molecule == m) & (df.router == "none")].verdict.iloc[0]; b = df[(df.planet == pl) & (df.molecule == m) & (df.router == "muting -> retrained hazedeck")].verdict.iloc[0]
            lab = RR.PANEL[pl]["labels"].get(m)
            if a != b:
                ch.append(f"{m}: {a} -> {b} [{lab if lab else 'amb.'}]")
        L.append(f"  {pl:<12} muting {mu:5.2f}  prior max {mx:4.2f}  P {pb:4.2f}  {'FLAG' if f else '    '}{'*' if fs else ' '}  " + ("; ".join(ch) if ch else "-"))
    open(os.path.join(OUT, "muting_router.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
