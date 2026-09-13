Subject: Rebuilding the Tier-1 classifiers of Mugnai et al. (2021) — four questions

Dear Dr Mugnai and co-authors,

I am a high-school student working on how machine-learning screens trained on simulated Ariel spectra behave when the simulator's physics is wrong. As part of that I rebuilt the Tier-1 molecular classifiers of your Alfnoor paper (AJ 162, 288, §II.5 and §III.3) from the text, since Alfnoor itself is not public, so that your design could be tested alongside mine. I would be grateful for four clarifications, because my rebuild reproduces your Table 6 only in part and the shortfall is most likely on my side.

What I did: the 965 known planets of the 2026 Mission Candidate Sample; POP-III training (each planet ×4, T in 0.7–1.05 T_p, CH4/H2O/CO2/NH3 log-uniform 1e-9–1e-2, grey cloud deck log-uniform 5e2–1e6 Pa, H2/He) and POP-I test (1e-7–1e-2); seven Tier-1 points (three photometric, NIRSpec ×1, AIRS-CH0 ×2 split at 2.76 µm, AIRS-CH1 ×1); Gaussian noise per target at the Tier-1 requirement (SNR 7 on the 5-scale-height modulation) with the ExoSim 2 noise shape; zero-mean unit-variance inputs; scikit-learn KNN/MLP/RFC/SVC at defaults; MultiREx over TauREx 3 with Exo-Transmit opacity tables.

What I get at the 1e-4 threshold: CH4 70–72 %, H2O 63–69 %, CO2 60 % (the majority rate), NH3 76–78 %, against your 82–87, 71–78, 79–83 and 82–87 %. Reducing my noise to about a third of the requirement brings CH4, H2O and NH3 into your ranges; CO2 does not move until the noise is a quarter, and moving the CH0 split to 2.5 or 3.0 µm changes nothing.

The questions:
1. Noise: is the Tier-1 noise in your POP-I/POP-III spectra the ArielRad estimate for the *integer* number of transits each target needs (so that bright targets have SNR well above 7), and is the SNR-7 requirement applied per Tier-1 bin or to the spectrum as a whole?
2. Binning: where does your AIRS-CH0 split fall, and is AIRS-CH1 a single point at Tier 1?
3. CO2: with a single 3.9–7.8 µm point, the 4.3 µm band is heavily diluted in my layout; is there something in your binning or opacities (ExoMol k-tables) that keeps CO2 detectable at Tier 1?
4. Would you be willing to share the trained classifiers or the POP-I/POP-III spectra, so that the test can be run on your actual screen rather than a reconstruction?

In case it is useful: my rebuild also shows that adding HCN and C2H2 at their equilibrium abundances — which the ABC database and the free-chemistry grids omit — leaves your molecule-presence classifiers almost unchanged but puts a carbon-rich (C/O > 1) screen trained on the same five gases at chance on carbon-rich planets. The code and every result are public at https://github.com/oy2017/BioSignatureDetectionModel (directory v3).

Thank you for your time.

[name, school, e-mail]
