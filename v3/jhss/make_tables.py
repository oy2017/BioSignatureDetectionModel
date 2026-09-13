"""Generate the data tables of the JHSS manuscript from the committed result files and insert them
between <!-- table:NAME --> ... <!-- /table:NAME --> markers in manuscript.md, so the tables can never
drift from the numbers. Run after any results change: python make_tables.py"""
import os, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
LAB = {"clean": "Clean", "cloud_1e5Pa": "Cloud deck, 10⁵ Pa", "cloud_1e4Pa": "Cloud deck, 10⁴ Pa", "cloud_1e3Pa": "Cloud deck, 10³ Pa", "cloud_1e2Pa": "Cloud deck, 10² Pa (out of range)",
       "cloud_1e1Pa": "Cloud deck, 10 Pa (out of range)", "haze_2p0e5": "Haze, 2 × 10⁵ m⁻³", "haze_2p0e6": "Haze, 2 × 10⁶ m⁻³", "haze_3p0e7": "Haze, 3 × 10⁷ m⁻³", "haze_2p4e8": "Haze, 2.4 × 10⁸ m⁻³",
       "haze_1p0e10": "Haze, 10¹⁰ m⁻³ (out of range)", "tlse_spots02": "Star spots, 2 %", "tlse_spots05": "Star spots, 5 %", "tlse_spots10": "Star spots, 10 %", "tlse_spots20": "Star spots, 20 %",
       "tlse_mixed": "Spots + faculae (out of range)", "tlse_fac10": "Faculae, 10 % (out of range)", "quenched": "Quenched chemistry", "exotransmit": "Other radiative-transfer code", "exomol": "Other opacity tables",
       "compound_spots10_haze3e7": "Spots 10 % + haze 3 × 10⁷", "compound_spots20_haze3e7": "Spots 20 % + haze 3 × 10⁷", "absorbers": "HCN + C₂H₂ omitted", "absorbers_quenched": "HCN + C₂H₂ omitted, quenched",
       "white_snr12": "White noise, SNR 12", "white_snr8": "White noise, SNR 8", "white_snr5": "White noise, SNR 5", "correlated_snr12": "Correlated noise, SNR 12", "correlated_snr8": "Correlated noise, SNR 8", "correlated_snr5": "Correlated noise, SNR 5"}
ORDER = list(LAB)


def envelope(cfg):
    e = pd.read_csv(os.path.join(RES, f"{cfg}_trust_envelope.csv")); e = e.set_index(["case", "score"])
    cases = [c for c in ORDER if (c, "ensemble") in e.index]
    L = ["| Mismatch | All | Ensemble | Margin | Mahalanobis | k-NN |", "| :-- | --: | :-- | :-- | :-- | :-- |"]
    for c in cases:
        cells = []
        for sc in ("ensemble", "margin", "mahalanobis", "knn"):
            r = e.loc[(c, sc)]; cells.append(f"{r.accuracy_accepted*100:.1f} ({r.coverage*100:.0f} %; {r.credit*100:+.1f})")
        L.append(f"| {LAB[c]} | {e.loc[(c, 'ensemble')].accuracy_all*100:.1f} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def ceilings():
    L = ["| Mismatch | Tier 3: clean-trained | Tier 3: randomized | Tier 3: held-out | Tier 3: ceiling | Tier 1: clean-trained | Tier 1: randomized | Tier 1: ceiling |",
         "| :-- | --: | --: | --: | --: | --: | --: | --: |"]
    R = {cfg: pd.read_csv(os.path.join(RES, f"{cfg}_trust_randomized.csv")).set_index("case") for cfg in ("ariel", "tier1")}
    O = {cfg: pd.read_csv(os.path.join(RES, f"{cfg}_oracle.csv")).set_index("case") for cfg in ("ariel", "tier1")}
    E = {cfg: pd.read_csv(os.path.join(RES, f"{cfg}_trust_envelope.csv")) for cfg in ("ariel", "tier1")}
    E = {k: v[v.score == "ensemble"].set_index("case") for k, v in E.items()}
    f = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.1f}"
    for c in ORDER:
        if c not in R["ariel"].index and c not in E["ariel"].index: continue
        vals = []
        for cfg in ("ariel", "tier1"):
            fr = R[cfg].loc[c, "frozen"] if c in R[cfg].index else (E[cfg].loc[c, "accuracy_all"] if c in E[cfg].index and c.startswith("absorbers") else np.nan)
            rd = E[cfg].loc[c, "accuracy_all"] if c in E[cfg].index else np.nan
            ho = R[cfg].loc[c, "held_out"] if c in R[cfg].index else np.nan
            oc = O[cfg].loc[c, "oracle"] if c in O[cfg].index else np.nan
            vals += [f(fr), f(rd)] + ([f(ho)] if cfg == "ariel" else []) + [f(oc)]
        L.append(f"| {LAB[c]} | " + " | ".join(vals) + " |")
    return "\n".join(L)


def consortium():
    a = pd.read_csv(os.path.join(RES, "alfnoor_screen.csv")); a = a[a.threshold == "1e-4"]
    theirs = {"CH4": "82–87", "H2O": "71–78", "CO2": "79–83", "NH3": "82–87"}
    cols = ["clean", "haze3e7", "cloud1e2", "absorbers", "spots10", "spots20", "noise_x2", "noise_x3"]
    heads = ["Clean", "Haze 3 × 10⁷", "Cloud 10² Pa", "HCN + C₂H₂", "Spots 10 %", "Spots 20 %", "Noise ×2", "Noise ×3"]
    L = ["| Molecule | Classifier | Their Table 6 | " + " | ".join(heads) + " |", "| :-- | :-- | --: | " + " | ".join("--:" for _ in heads) + " |"]
    for mol in ("CH4", "H2O", "CO2", "NH3"):
        for m in ("KNN", "MLP", "RFC", "SVC", "vote+ensemble", "vote+knn"):
            r = a[(a.molecule == mol) & (a.model == m)]
            if r.empty: continue
            r = r.iloc[0]; name = {"vote+ensemble": "vote, ensemble rule", "vote+knn": "vote, k-NN rule"}.get(m, m)
            L.append(f"| {mol.replace('CH4','CH₄').replace('H2O','H₂O').replace('CO2','CO₂').replace('NH3','NH₃')} | {name} | {theirs[mol] if m == 'KNN' else ''} | " + " | ".join(f"{r[c]*100:.1f}" for c in cols) + " |")
    return "\n".join(L)


def axes():
    rows = [("Cloud deck", "grey, optically thick cloud top at 10⁵, 10⁴, 10³, 10², 10 Pa", "aerosols hide the lower atmosphere", "10³–10⁵ Pa, 50 % of planets"),
            ("Haze", "Lee et al. Mie haze, 0.1 µm, at 2 × 10⁵, 2 × 10⁶, 3 × 10⁷, 2.4 × 10⁸, 10¹⁰ m⁻³", "photochemical haze mutes and slopes the spectrum", "10⁵–3 × 10⁸ m⁻³, 60 %"),
            ("Stellar contamination", "unocculted spots at 2, 5, 10, 20 % coverage (contrast 0.85); faculae 5, 10 %; mixed", "the transit light-source effect imprints the star on the planet", "spots 0–20 %, 70 %"),
            ("Noise level and colour", "white and time-correlated (σ = 3 bins) at effective SNR 12, 10, 8, 5; a gain ramp ×0.25–2", "the deployed noise differs from the training noise", "SNR 5–15, all planets"),
            ("Radiative-transfer code", "the same planets rendered by Exo-Transmit with the same opacities", "code-to-code differences in the forward model", "never (held out)"),
            ("Opacity database", "ExoMol cross sections for H₂O, CH₄, CO₂, CO in place of Exo-Transmit's tables", "line-list differences between databases", "never (held out)"),
            ("Quenched chemistry", "carbon–oxygen partition frozen at the quench level, K<sub>zz</sub> = 10⁷–10¹¹ cm² s⁻¹", "vertical mixing drives the photosphere out of equilibrium", "50 % of planets"),
            ("Omitted absorbers", "HCN and C₂H₂ added at their FastChem equilibrium or quenched abundances", "species the training forward model does not contain", "never (held out)"),
            ("Compounds", "spots 10–20 % × haze 2 × 10⁶–3 × 10⁷ × SNR 8–10 on the same planet", "real planets are off on several axes at once", "jointly, through the draws above"),
            ("Binning and targets", "Tier-1 (7 points) and Tier-2 (51) layouts; the 965 known Mission Candidate Sample planets under the mission's noise definition", "the decision is made at Tier 1, on real targets", "—")]
    L = ["| Axis | What is varied | Concern it represents | In the randomized grid |", "| :-- | :-- | :-- | :-- |"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return "\n".join(L)


def main():
    p = os.path.join(HERE, "manuscript.md"); s = open(p).read()
    for name, fn in (("axes", axes), ("consortium", consortium), ("envelope3", lambda: envelope("ariel")), ("envelope1", lambda: envelope("tier1")), ("ceilings", ceilings)):
        pat = re.compile(rf"<!-- table:{name} -->.*?<!-- /table:{name} -->", re.S)
        assert pat.search(s), name
        s = pat.sub(f"<!-- table:{name} -->\n{fn()}\n<!-- /table:{name} -->", s)
    open(p, "w").write(s); print("tables inserted")


if __name__ == "__main__":
    main()
