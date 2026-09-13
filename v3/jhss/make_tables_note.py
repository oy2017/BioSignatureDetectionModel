"""Fill the data tables of the technical note from committed result files, between
<!-- table:NAME --> ... <!-- /table:NAME --> markers in note_src.md. Run before render_note.py."""
import os, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
PUB = {"KNN": {"CH4": (79, 83, 85), "CO2": (77, 79, 82), "H2O": (64, 71, 82), "NH3": (75, 82, 84)},
       "MLP": {"CH4": (78, 85, 87), "CO2": (77, 81, 83), "H2O": (70, 76, 84), "NH3": (80, 86, 87)},
       "RFC": {"CH4": (77, 82, 87), "CO2": (76, 79, 83), "H2O": (69, 74, 82), "NH3": (78, 85, 87)},
       "SVC": {"CH4": (79, 86, 89), "CO2": (79, 83, 84), "H2O": (69, 78, 84), "NH3": (81, 87, 87)}}
MOL = {"CH4": "CH₄", "H2O": "H₂O", "CO2": "CO₂", "NH3": "NH₃"}; TH = ["1e-5", "1e-4", "1e-3"]


def repro():
    d = pd.read_csv(os.path.join(RES, "alfnoor_faithful_faithful_tier3_r20_radiometric.csv"), dtype={"threshold": str})
    d = d[~d.model.str.startswith("vote")]
    L = ["| Classifier | Molecule | > 10⁻⁵: this work | > 10⁻⁵: published | > 10⁻⁴: this work | > 10⁻⁴: published | > 10⁻³: this work | > 10⁻³: published |",
         "| :-- | :-- | --: | --: | --: | --: | --: | --: |"]
    for c in PUB:
        for m in ("CH4", "H2O", "CO2", "NH3"):
            cells = []
            for k, th in enumerate(TH):
                r = d[(d.model == c) & (d.molecule == m) & (d.threshold == th)].iloc[0]
                cells += [f"{r.clean*100:.1f}", f"{PUB[c][m][k]}"]
            L.append(f"| {c} | {MOL[m]} | " + " | ".join(cells) + " |")
    return "\n".join(L)


CASES = [("cloud1e3", "Grey cloud deck, 10³ Pa", "yes"), ("cloud1e2", "Grey cloud deck, 10² Pa", "no"),
         ("haze2e6", "Haze, 2 × 10⁶ m⁻³", "no"), ("haze3e7", "Haze, 3 × 10⁷ m⁻³", "no"), ("haze2p4e8", "Haze, 2.4 × 10⁸ m⁻³", "no"),
         ("spots10", "Unocculted spots, 10 %", "no"), ("spots20", "Unocculted spots, 20 %", "no"), ("compound", "Spots 20 % + haze 3 × 10⁷ m⁻³", "no"),
         ("white_x2", "White noise, 2 × σ", "no"), ("white_x3", "White noise, 3 × σ", "no"), ("corr_x2", "Correlated noise, 2 × σ", "no"),
         ("ramp_x2", "Gain ramp, 2 noise levels", "no"), ("exomol", "ExoMol opacity database", "no"),
         ("exotransmit", "Exo-Transmit radiative transfer", "no"), ("absorbers", "HCN and C₂H₂ added", "no")]


def consortium():
    fz = pd.read_csv(os.path.join(RES, "alfnoor_trust_frozen.csv"), dtype={"threshold": str}); fz = fz[fz.threshold == "1e-4"]
    m = fz.groupby("molecule").mean(numeric_only=True) * 100; c0 = m["clean"].mean()
    ce = pd.read_csv(os.path.join(RES, "alfnoor_trust_ceiling.csv")).groupby("case").ceiling.mean() * 100
    rd = pd.read_csv(os.path.join(RES, "alfnoor_trust_randomized.csv")); rf = rd[rd.variant == "full"].groupby("molecule").mean(numeric_only=True) * 100
    det = pd.read_csv(os.path.join(RES, "alfnoor_trust_detect.csv")).groupby(["rule", "case"]).mean(numeric_only=True)
    cal = pd.read_csv(os.path.join(RES, "alfnoor_trust_calibration.csv")).groupby("case").mean(numeric_only=True)
    L = [f"| Mismatch | Accuracy (%) | Loss | Irreducible | Loss after randomized training | Margin rule keeps (%) | Margin AUROC (error) | Distance AUROC (shift) | Conformal coverage (%) |",
         "| :-- | --: | --: | --: | --: | --: | --: | --: | --: |",
         f"| None (clean) | {c0:.1f} | — | — | {c0 - rf['clean'].mean():.1f} | {det.loc[('margin','clean')].coverage*100:.0f} | {det.loc[('margin','clean')].auroc_error:.2f} | — | {cal.loc['clean','coverage']*100:.0f} |"]
    for k, lab, _ in CASES:
        irr = f"{c0 - ce[k]:.1f}" if k in ce.index else "—"
        cov = f"{cal.loc[k,'coverage']*100:.0f}" if k in cal.index else "—"
        L.append(f"| {lab} | {m[k].mean():.1f} | {c0 - m[k].mean():.1f} | {irr} | {c0 - rf[k].mean():.1f} | {det.loc[('margin',k)].coverage*100:.0f} | "
                 f"{det.loc[('margin',k)].auroc_error:.2f} | {det.loc[('mahalanobis',k)].auroc_shift:.2f} | {cov} |")
    return "\n".join(L)


def variants():
    d = pd.read_csv(os.path.join(RES, "alfnoor_haze_mechanism.csv")); g = d.groupby(["variant", "case"])[["accuracy", "recall"]].mean() * 100
    V = [("A published", "All 104 bins (published)"), ("B ir_stats", "All bins, normalised with the bins above 1.1 µm"), ("C no_optical", "Photometric points removed"), ("D airs_only", "AIRS bins only (above 1.95 µm)")]
    cols = [("clean", "Clean"), ("haze2e6", "Haze 2 × 10⁶"), ("haze3e7", "Haze 3 × 10⁷"), ("haze2p4e8", "Haze 2.4 × 10⁸"), ("spots20", "Spots 20 %"), ("compound", "Spots + haze")]
    L = ["| Classifier input | " + " | ".join(c[1] for c in cols) + " |", "| :-- | " + " | ".join("--:" for _ in cols) + " |"]
    for v, lab in V:
        L.append(f"| {lab} | " + " | ".join(f"{g.loc[(v, c), 'accuracy']:.1f} ({g.loc[(v, c), 'recall']:.0f})" for c, _ in cols) + " |")
    return "\n".join(L)


def main():
    p = os.path.join(HERE, "note_src.md"); s = open(p, encoding="utf-8").read()
    for name, fn in (("repro", repro), ("consortium", consortium), ("variants", variants)):
        s, n = re.subn(rf"(<!-- table:{name} -->\n).*?(<!-- /table:{name} -->)", lambda mm: mm.group(1) + fn() + "\n" + mm.group(2), s, flags=re.S)
        print(name, "filled" if n else "MARKER MISSING")
    open(p, "w", encoding="utf-8").write(s)


if __name__ == "__main__":
    main()
