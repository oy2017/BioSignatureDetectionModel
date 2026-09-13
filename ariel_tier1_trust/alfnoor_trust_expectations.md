# The consortium screen under the full trust procedure: expectations written before the run

Written 2026-09-13, before `alfnoor_trust.py` was run. Screen: the Mugnai et al. (2021) Tier-1
molecular screen as re-implemented in `alfnoor_faithful.py` (Tier-3 binning R = 20/100/30, Tier-1
noise from the payload model at each target's integer Tier-1 transit count, POP-III training, POP-I
test, KNN/MLP/RFC/SVC at scikit-learn defaults). Reported at the 1e-4 abundance threshold, mean over
the four classifiers, unless stated. "Loss" is in accuracy points against the screen's own clean
accuracy.

**Already observed** (reproduction runs of 2026-09-12/13, so not predictions): clean accuracy per
molecule (CH4 85, H2O 75, CO2 80, NH3 84); haze 3e7 m^-3 costs 13–23; a grey deck at 1e2 Pa costs
6–14; spots 10/20 % cost 4–11 / 7–15; noise doubled costs 6–11; HCN + C2H2 at 1e-7–1e-4 cost ≤ 0.5.
Under haze the four classifiers agree on the wrong answer (ensemble keeps ~100 %), and the k-NN distance
declines almost everything.

## Predictions

1. **Ceilings.** Aerosol losses are mostly irreducible: a screen trained at haze 3e7 or at a 1e2 Pa deck
   recovers < 50 % of the loss, because the features are physically removed. Stellar contamination is
   mostly reducible (> 70 %), because it is a smooth multiplicative distortion. HCN + C2H2 has nothing
   to recover.
2. **Randomized training grid** (haze on 60 %, spots on 70 %, noise ×1–3 on POP-III; the random grey
   deck is already part of the published design). Clean cost ≤ 2 points. Recovers ≥ 50 % of the
   spot loss and < 50 % of the haze loss. Held-out ingredients transfer little (< 30 %).
3. **Opacity database and radiative-transfer code.** The ExoMol swap costs ≥ 5 points (the carbon-rich
   screen lost 9); Exo-Transmit's own code costs ≤ 2 (the carbon-rich screen lost 0.7).
4. **Compound** spots 20 % + haze 3e7 is sub-additive, as on the carbon-rich screen.
5. **Correlated noise** at the same per-bin variance costs ≥ 2 points more than white noise.
6. **Decline rules.** Confidence (probability margin, ensemble spread) ranks errors on clean data with
   AUROC ≥ 0.75 but loses that ranking under haze and clouds; distance rules (Mahalanobis, k-NN, PCA
   reconstruction) separate shifted from clean spectra (shift AUROC ≥ 0.9 for haze and deck) but rank
   errors near chance; no rule has positive credit against the clean selective baseline on any
   aerosol case.
7. **Calibration.** Expected calibration error rises ≥ 3× under haze 3e7; split-conformal coverage
   calibrated at 90 % on clean spectra falls below 85 % under haze.
8. **Host dependence.** The spot loss is ≥ 1.5× larger for M-dwarf hosts than for FGK hosts.
9. **Trade-off.** Randomizing haze and spots lowers the Mahalanobis shift AUROC for those cases by
   ≥ 0.2, as on the carbon-rich screen.

A prediction is reported as met or not met in the results file; misses are reported as misses.
