"""Copy the result files and figures cited by the technical note from v3/ into this directory, so a reader
finds them in one place. The sources stay in v3/ (where the scripts run); re-run after any recompute:
    python ariel_tier1_trust/collect.py"""
import os, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); V3 = os.path.join(ROOT, "v3"); HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "results/reproduction": ["alfnoor_faithful_faithful_tier3_r20_radiometric.csv", "alfnoor_faithful_faithful_tier3_r20_radiometric.txt"],
    "results/consortium_stress_test": ["alfnoor_trust.txt", "alfnoor_trust_frozen.csv", "alfnoor_trust_ceiling.csv", "alfnoor_trust_randomized.csv",
                                       "alfnoor_trust_detect.csv", "alfnoor_trust_calibration.csv", "alfnoor_trust_host.csv", "alfnoor_trust_tradeoff.csv", "alfnoor_trust_fallbacks.csv"],
    "results/haze_mechanism": ["alfnoor_haze_mechanism.txt", "alfnoor_haze_mechanism.csv", "haze_generality_photnoise.txt", "haze_generality_photnoise.csv",
                               "haze_generality_particles.txt", "haze_generality_particles.csv", "haze_generality_carbonrich.txt", "haze_generality_carbonrich.csv"],
    "results/carbon_rich_classifier": ["ariel_trust_randomized.txt", "ariel_trust_detect.txt", "ariel_trust_envelope.txt", "ariel_oracle.txt", "ariel_conformal_calibration.txt",
                                       "ariel_mechanism_bands.txt", "tier_screen.txt", "tier1_trust_detect.txt", "mcs_absorbers.txt", "ariel_mcs.txt", "ariel_sensitivity.txt"],
    "results/forward_model_checks": ["check_pressure_units.txt"],
}
for sub, names in FILES.items():
    os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    for n in names:
        src = os.path.join(V3, "results", n)
        if os.path.exists(src): shutil.copy2(src, os.path.join(HERE, sub, n))
        else: print("missing", n)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
for n in sorted(os.listdir(os.path.join(V3, "results", "figures"))):
    if n.startswith("note_fig"): shutil.copy2(os.path.join(V3, "results", "figures", n), os.path.join(HERE, "figures", n))
for n in ("alfnoor_trust_expectations.md",):
    shutil.copy2(os.path.join(V3, n), os.path.join(HERE, n))
print("collected")
