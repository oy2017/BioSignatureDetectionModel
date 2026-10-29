"""The routed pipeline on the real JWST panel: each (molecule, learner) cell takes the AIRS-only learner's verdict when a
flag fires on the all-input spectrum (the optical-excess score, the input disagreement, or either, each at the threshold
that declines 10 % of clean POP-I planets), and the all-input verdict otherwise; the planet's verdict is the majority of
the four learners as in jwst_panel.py. Scored against the literature labels of jwst_panel.LABELS (ambiguous cells not
scored), for every reduction of the panel.

Usage: python router_real.py   -> results/jwst_panel/router_real.{txt,csv}
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402
import alfnoor_diagnostic as D, jwst_panel as JP  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel"); ROUTERS = ["none", "optical", "disagree", "either", "AIRS-only everywhere"]
# WASP-39 b, the external-validation target of Section 3.5, scored here too from its benchmark assembly (wasp39_assemble.py).
# Labels at the 1e-4 threshold from free-chemistry retrievals of the same JWST data: H2O log -2.08 +/- 0.09 (Deka et al. 2025,
# NEXOTRANS, 0.6-12 um) -> P; CO2 -3.74 +/- 0.10 with patchy clouds but -4.33 +/- 0.09 with Mie aerosols (same paper) -> ambiguous;
# CH4 VMR < 5e-6 (Rustamkulov et al. 2023, PRISM) -> A; NH3 unconstrained in the published free retrievals -> ambiguous.
EXTRA = {"WASP-39b": dict(labels={"CH4": JP.A, "H2O": JP.P, "CO2": None, "NH3": None},
                          coverage=os.path.join(RESULTS, "wasp39", "coverage_benchmark.csv"), verdicts=os.path.join(RESULTS, "wasp39", "verdicts_benchmark.csv"))}
PANEL = {pl: dict(labels=labs, coverage=os.path.join(OUT, f"{pl}_coverage.csv"), verdicts=os.path.join(OUT, f"{pl}_verdicts.csv")) for pl, labs in JP.LABELS.items()}
PANEL.update(EXTRA)
METHANE = [p for p, d in PANEL.items() if d["labels"].get("CH4") == JP.P]


def main():
    cases, Pte, e, sig = D.test_cases(); sc = D.Scorer(e); S0, _, _ = sc.scores(cases["clean"])
    q = lambda s: np.quantile(s[np.isfinite(s)], 1 - D.DECLINE)
    thr_opt = q(S0[("optical", None)]); thr_dis = {(m, k): q(S0[("disagree", (m, k))]) for m in D.MOLS for k in D.LEARNERS}
    rows = []
    for pl, d in PANEL.items():
        labs = d["labels"]
        if not os.path.exists(d["coverage"]):
            print(f"{pl}: no coverage file, skipped", flush=True); continue
        cov = pd.read_csv(d["coverage"]); x = cov.depth.to_numpy().copy(); filled = cov.filled.to_numpy(bool); x[filled] = np.nanmean(x[~filled])
        S, _, _ = sc.scores(x[None, :]); V = pd.read_csv(d["verdicts"]); V = V[np.isclose(V.threshold.astype(float), 1e-4)]
        pres = {(v, m, k): int(V[(V.variant == v) & (V.molecule == m) & (V.learner == k)].present.iloc[0]) for v in ("all", "airs_only") for m in D.MOLS for k in D.LEARNERS}
        f_opt = bool(S[("optical", None)][0] > thr_opt)
        for m in D.MOLS:
            for router in ROUTERS:
                n_present = 0; n_flag = 0
                for k in D.LEARNERS:
                    f_dis = bool(S[("disagree", (m, k))][0] > thr_dis[(m, k)])
                    flag = {"none": False, "optical": f_opt, "disagree": f_dis, "either": f_opt or f_dis, "AIRS-only everywhere": True}[router]
                    n_flag += flag; n_present += pres[("airs_only" if flag else "all", m, k)]
                verdict = JP.P if n_present >= 3 else (JP.A if n_present <= 1 else "split"); lab = labs.get(m)
                rows.append(dict(planet=pl, molecule=m, router=router, flagged_learners=n_flag, learners_present=n_present, verdict=verdict, label=lab,
                                 correct=None if lab is None or verdict == "split" else bool(verdict == lab)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "router_real.csv"), index=False)
    L = [f"The routed pipeline on the real panel ({df.planet.nunique()} reductions, WASP-39 b included; flag thresholds decline 10 % of clean POP-I planets; verdict = majority of four learners; scored labels only)", ""]
    L.append(f"{'router':<22} {'scored correct':>15} {'CH4 on the six methane reductions: present / split / absent':>62} {'cells routed':>13}")
    for r in ROUTERS:
        d = df[df.router == r]; sc_ = d[d.correct.notna()]; ch4 = d[(d.molecule == "CH4") & d.planet.isin(METHANE)]
        L.append(f"{r:<22} {int(sc_.correct.sum()):>7}/{len(sc_):<7} {int((ch4.verdict == JP.P).sum()):>22} / {int((ch4.verdict == 'split').sum())} / {int((ch4.verdict == JP.A).sum()):<28} {int(d.flagged_learners.sum()):>6}/{len(d)*4}")
    L.append(""); L.append("CH4 on the methane reductions, learners routed and verdict (all-input -> routed), per router:")
    for pl in METHANE:
        cells = []
        for r in ROUTERS[1:]:
            d = df[(df.planet == pl) & (df.molecule == "CH4") & (df.router == r)].iloc[0]; d0 = df[(df.planet == pl) & (df.molecule == "CH4") & (df.router == "none")].iloc[0]
            cells.append(f"{r}: {d.flagged_learners}/4 routed, {d0.verdict} -> {d.verdict}")
        L.append(f"  {pl:<12} " + "; ".join(cells))
    L.append(""); L.append("labels that change under each router (planet molecule: all-input -> routed [label]):")
    for r in ROUTERS[1:]:
        ch = []
        for (pl, m), g in df.groupby(["planet", "molecule"], sort=False):
            a = g[g.router == "none"].iloc[0]; b = g[g.router == r].iloc[0]
            if a.verdict != b.verdict: ch.append(f"{pl} {m}: {a.verdict} -> {b.verdict} [{a.label or 'amb.'}]")
        L.append(f"  {r}: " + ("; ".join(ch) if ch else "none"))
    txt = "\n".join(L); print(txt); open(os.path.join(OUT, "router_real.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
