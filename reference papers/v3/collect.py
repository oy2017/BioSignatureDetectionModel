"""Collect the v3 reference set: verify each arXiv id against its expected title, download the PDF
(unless already in reference papers/), and emit references.bib + INDEX.md with 'why cited'."""
import os, re, sys, time, json, urllib.request, urllib.parse, xml.etree.ElementTree as ET
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NS = {"a": "http://www.w3.org/2005/Atom"}
# key, arxiv id, expected title fragment (lowercase), topic, why cited, existing file in reference papers/ (or None)
REFS = [
 # --- Ariel mission and its screens
 ("tinetti2018",   None, None, "Ariel", "the mission, its science goals and the tier concept (Exp. Astron. 46, 135; no arXiv; file ../s10686-018-9598-x.pdf)", "s10686-018-9598-x.pdf"),
 ("edwards2019",   "1905.04959", "updated study of potential targets for ariel", "Ariel", "the target list, tier definitions by SNR, number of transits", "Edwards_2019_AJ_157_242.pdf"),
 ("edwards2022",   "2205.05073", "ariel target list", "Ariel", "current MCS; the list our real-target test set is built from", "Ariel_Mission_Reference_Sample_Edwards_2022.pdf"),
 ("mugnai2020",    "2009.07824", "arielrad", "Ariel", "the radiometric model behind the tier SNR requirement", "Mugnai_2020_ArielRad.pdf"),
 ("mugnai2025",    "2501.12809", "exosim 2", "Ariel", "the consortium simulator whose noise shape we adopt", None),
 ("changeat2020",  "2003.01839", "alfnoor", "Ariel", "population retrievals of the target list with TauREx + ArielRad", "Changeat_2020_AJ_160_80.pdf"),
 ("mugnai2021",    "2110.00503", "alfnoor", "Ariel", "THE consortium Tier-1 screen we rebuild: selection strategy, band metric, four ML classifiers, in-simulator validation", None),
 ("changeat2023",  "2206.14633", "ariel data challenge", "Ariel", "the ABC training grid: H2O, CH4, CO, CO2, NH3 only (sec 2.2)", "ESA_Ariel_Data_Challenge_Changeat_Yip_2023.pdf"),
 ("aubin2023",     "2309.09337", "ariel data challenge 2023", "Ariel", "ADC 2023 winner; seven targets = R, T and five abundances", None),
 ("mugnai2026",    "2605.03719", "public dataset of ariel simulated observations", "Ariel", "instrument-only train/test shift; the nearest shift dataset", "Mugnai_2026_Ariel_simulated_observations_dataset.pdf"),
 ("tiers2026",     "2604.07598", "information content of ariel transmission spectra", "Ariel", "tier binning R ~1/3/1 and 10/50/10; what each tier constrains", None),
 ("yakubu2026",    "2606.23766", "machine learning and deep learning for exoplanet detection", "Ariel", "2026 review: robustness under mismatch listed as open, physics-side mismatch absent", None),
 # --- ML for exoplanet atmospheres (what exists, how it is validated)
 ("marquezneila2018", "1806.03944", "supervised machine learning for analysing spectra", "ML-exo", "random forest retrieval; included HCN for WFC3", "1806.03944v1.pdf"),
 ("zingales2018",  "1806.02906", "exogan", "ML-exo", "GAN retrieval; in-simulator validation", None),
 ("cobb2019",      "1905.10659", "ensemble of bayesian neural networks", "ML-exo", "ensembles for retrieval", None),
 ("nixon2020",     "2004.10755", "assessment of supervised machine learning", "ML-exo", "random-forest retrieval assessment; no OOD test", None),
 ("yip2021",       "2011.11284", "peeking inside the black box", "ML-exo", "sensitivity of DNN retrievals to spectral features", None),
 ("ardevol2022",   "2203.01236", "convolutional neural networks as an alternative", "ML-exo", "CLOSEST PRECEDENT: CNN tested on added/removed species and star spots; no remedy, no detection", None),
 ("vasist2023",    "2301.06575", "neural posterior estimation for exoplanetary", "ML-exo", "NPE retrieval; coverage diagnostics in-simulator", None),
 ("gebhard2024",   "2410.21477", "flow matching for atmospheric retrieval", "ML-exo", "FMPE retrieval with noise-level conditioning; state of the art in ML retrieval", None),
 ("yip2023",       None, None, "Ariel", "lessons learned from ADC 2022 (PMLR 220; file ../Yip_2023_Lessons_Learned_ADC2022_PMLR_v220.pdf)", "Yip_2023_Lessons_Learned_ADC2022_PMLR_v220.pdf"),
 ("hayes2020",     "1909.00718", "unsupervised machine-learning classification", "ML-exo", "PCA/k-means classes as retrieval priors", "Hayes_2020_unsupervised_retrieval_classification.pdf"),
 ("duque2025",     "2407.19167", "machine-assisted classification of potential biosignatures", "ML-exo", "the MultiREx framework and a low-SNR classification screen", "2407.19167v2.pdf"),
 ("ares2024",      "2401.03809", "ares vi", "ML-exo", "1D-retrieval biases from 3D effects: forward-model misspecification in classical retrieval", None),
 ("barstow2020",   "2002.01063", "comparison of exoplanet spectroscopic retrieval tools", "ML-exo", "retrieval-code intercomparison; forward-model choice priced for classical retrieval", None),
 # --- chemistry of the label and the omitted species
 ("madhusudhan2012", "1109.3183", "carbon-rich giant planets", "chemistry", "C/O > 1: HCN, C2H2 major constituents; tracers of C/O above 800 K", None),
 ("moses2013",     "1211.2996", "chemical consequences of the c/o ratio", "chemistry", "C/O chemistry with quenching and photochemistry; disequilibrium enhances HCN, C2H2", None),
 ("zahnle2014",    "1408.6283", "methane, carbon monoxide, and ammonia", "chemistry", "the CO/CH4 quench timescale used in Axis 8", None),
 ("stock2018",     "1804.05010", "fastchem", "chemistry", "the equilibrium chemistry code behind the label", None),
 ("kawashima2021", "2110.13443", "disequilibrium chemistry", "chemistry", "equilibrium assumption biases retrieved C/O (ARCiS)", None),
 ("baeyens2025",   "2506.12806", "hot-jupiter atmospheres with disequilibrium", "chemistry", "re-analysis of ten hot Jupiters with disequilibrium retrieval", None),
 # --- stellar contamination and aerosols
 ("rackham2018",   "1711.05691", "transit light source effect", "physics", "contamination largest for M dwarfs: the expected host dependence", None),
 ("pinhas2018",    "1811.00011", "stellar heterogeneity", "physics", "retrieval of stellar heterogeneity with the spectrum", None),
 ("lee2013",       "1307.1404", "hr 8799", "physics", "the Mie haze prescription (Lee et al. 2013)", "Lee_2013_HR8799b_retrieval.pdf"),
 # --- forward models and opacities
 ("alrefaie2021",  "1912.07759", "taurex 3", "codes", "the radiative-transfer core under MultiREx", "Al-Refaie_2021_ApJ_917_37.pdf"),
 ("kempton2017",   "1611.03871", "exo-transmit", "codes", "the independent code and the source of the CO/NH3/HCN/C2H2 tables", "Kempton_2017_ExoTransmit.pdf"),
 ("chubb2021",     "2009.00687", "exomolop", "codes", "the ExoMol cross sections of the opacity axis", "Chubb_2021_ExoMolOP.pdf"),
 ("freedman2014",  "1409.0026", "gaseous mean opacities", "codes", "the opacity compilation behind Exo-Transmit", "Freedman_2014_gaseous_mean_opacities.pdf"),
 ("lupu2014",      "1401.1499", "atmospheres of earthlike planets after giant impact", "codes", "line-list sources of the Exo-Transmit tables (Table 2)", "Lupu_2014_atmospheres_after_giant_impact.pdf"),
 # --- the ML methods we apply (all standard; cited as origins)
 ("bishop1995",    None, None, "ML-methods", "training with noise = Tikhonov regularization (why noise augmentation cannot undo a draw)", None),
 ("chen2020",      "1907.10905", "group-theoretic framework for data augmentation", "ML-methods", "augmentation as group invariance", None),
 ("tobin2017",     "1703.06907", "domain randomization", "ML-methods", "the sim-to-real recipe we apply to spectra", None),
 ("hendrycks2017", "1610.02136", "baseline for detecting misclassified", "ML-methods", "the softmax-confidence baseline", None),
 ("lee2018",       "1807.03888", "mahalanobis", "ML-methods", "Mahalanobis OOD score", None),
 ("sun2022",       "2204.06507", "nearest neighbors", "ML-methods", "k-NN distance OOD score", None),
 ("lakshminarayanan2017", "1612.01474", "deep ensembles", "ML-methods", "ensemble disagreement as uncertainty", None),
 ("geifman2017",   "1705.08500", "selective classification", "ML-methods", "selective prediction / accuracy-coverage", None),
 ("jaeger2023",    "2211.15259", "failure detection", "ML-methods", "OOD detection != failure detection; FD-Shifts; our detection inversion reproduces it", None),
 ("hendrycks2019", "1903.12261", "common corruptions", "ML-methods", "held-out-corruption evaluation protocol", None),
 ("rusak2020",     "2001.06057", "robust against diverse image corruptions", "ML-methods", "noise augmentation and variance tuning", None),
 ("angelopoulos2021", "2107.07511", "conformal prediction", "ML-methods", "split conformal prediction (our baseline)", None),
 ("tibshirani2019", "1904.06019", "conformal prediction under covariate shift", "ML-methods", "why the conformal guarantee needs the deployment distribution", None),
 # --- misspecification for simulation-based inference (the cosmology precedent)
 ("schmitt2021",   "2112.08866", "detecting model misspecification in amortized", "SBI", "misspecification detection for amortized inference", None),
 ("cannon2022",    "2209.01845", "model misspecification in neural simulation-based", "SBI", "impact of misspecification on neural SBI", None),
 ("huang2023",     "2305.15871", "robust statistics for simulation-based inference", "SBI", "robust summaries under misspecification", None),
 ("sbi2025a",      "2507.13495", "deep ensembles", "SBI", "ensemble-based misspecification detection (cosmology)", None),
 ("sbi2025b",      "2508.05744", "misspecification in cosmology", "SBI", "flow-based OOD detection for cosmological SBI", None),
]

def meta(ids):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    r = ET.fromstring(urllib.request.urlopen(url, timeout=120).read()); out = {}
    for e in r.findall("a:entry", NS):
        i = e.find("a:id", NS).text.split("/abs/")[-1]; i = re.sub(r"v\d+$", "", i)
        out[i] = dict(title=e.find("a:title", NS).text.replace("\n", " ").strip(), year=e.find("a:published", NS).text[:4],
                      authors=[a.find("a:name", NS).text for a in e.findall("a:author", NS)])
    return out

ids = [r[1] for r in REFS if r[1]]
M = {}
for k in range(0, len(ids), 20):
    M.update(meta(ids[k:k+20])); time.sleep(8)
rows, bib = [], []
for key, aid, frag, topic, why, existing in REFS:
    if aid is None:
        if existing:
            rows.append((topic, key, key, why, "../" + existing)); bib.append(f"@misc{{{key}, note={{{why}}}}}"); continue
        rows.append((topic, key, "Bishop 1995, Neural Computation 7, 108 (no arXiv)", why, "—")); 
        bib.append(f"@article{{{key},\n  author={{Bishop, Christopher M.}}, title={{Training with noise is equivalent to Tikhonov regularization}},\n  journal={{Neural Computation}}, volume={{7}}, pages={{108--116}}, year={{1995}}}}")
        continue
    m = M.get(aid); ok = m is not None and frag in m["title"].lower()
    status = "OK" if ok else ("NO METADATA" if m is None else f"TITLE MISMATCH: {m['title'][:70]}")
    fname = existing or f"{key}_{aid}.pdf"; dest = os.path.join(ROOT, existing) if existing else os.path.join(HERE, fname)
    if ok and not os.path.exists(dest):
        try:
            urllib.request.urlretrieve(f"https://arxiv.org/pdf/{aid}", dest); time.sleep(2)
        except Exception as e:
            status += f" (download failed: {e})"
    if m:
        first = m["authors"][0].split()[-1]; etal = " et al." if len(m["authors"]) > 2 else ""
        bib.append(f"@article{{{key},\n  author={{{' and '.join(m['authors'])}}},\n  title={{{m['title']}}},\n  journal={{arXiv e-prints}}, eprint={{{aid}}}, year={{{m['year']}}}}}")
        rows.append((topic, key, f"{first}{etal} {m['year']}: {m['title'][:80]}", why, status if not ok else ("../" + existing if existing else fname)))
    else:
        rows.append((topic, key, aid, why, status))
open(os.path.join(HERE, "references.bib"), "w").write("\n\n".join(bib) + "\n")
L = ["# References for the v3 paper — what each is cited for", "", "Files: this directory (new) or `../` (already in reference papers/). Status column flags any id whose arXiv title did not match the expected paper.", ""]
for topic in ("Ariel", "ML-exo", "chemistry", "physics", "codes", "ML-methods", "SBI"):
    L += [f"## {topic}", "", "| key | paper | cited for | file / status |", "|---|---|---|---|"]
    L += [f"| {k} | {p} | {w} | {s} |" for t, k, p, w, s in rows if t == topic]; L.append("")
open(os.path.join(HERE, "INDEX.md"), "w").write("\n".join(L))
print("\n".join(f"{k:<22} {s}" for t, k, p, w, s in rows))
