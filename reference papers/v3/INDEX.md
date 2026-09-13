# References for the v3 paper — what each is cited for

Files: this directory (new downloads, key_arxivid.pdf) or `../` (already in reference papers/). Every arXiv id was checked against its title (Semantic Scholar metadata or the PDF's first page).

## Ariel

| key | paper | cited for | file |
|---|---|---|---|
| tinetti2018 | tinetti2018 | the mission, its science goals and the tier concept (Exp. Astron. 46, 135; no arXiv; file ../s10686-018-9598-x.pdf) | ../s10686-018-9598-x.pdf |
| edwards2019 | Edwards et al. 2019: An Updated Study of Potential Targets for Ariel | the target list, tier definitions by SNR, number of transits | ../Edwards_2019_AJ_157_242.pdf |
| edwards2022 | Edwards 2022: The Ariel Target List: The Impact of TESS and the Potential for Characterizing Multip | current MCS; the list our real-target test set is built from | ../Ariel_Mission_Reference_Sample_Edwards_2022.pdf |
| mugnai2020 | Mugnai et al. 2020: ArielRad: the Ariel radiometric model | the radiometric model behind the tier SNR requirement | ../Mugnai_2020_ArielRad.pdf |
| mugnai2025 | Mugnai et al. 2025: ExoSim 2: the new exoplanet observation simulator applied to the Ariel space mission | the consortium simulator whose noise shape we adopt | mugnai2025_2501.12809.pdf |
| changeat2020 | Changeat et al. 2020: Alfnoor: A Retrieval Simulation of the Ariel Target List | population retrievals of the target list with TauREx + ArielRad | ../Changeat_2020_AJ_160_80.pdf |
| mugnai2021 | Mugnai et al. 2021: Alfnoor: Assessing the Information Content of Ariel's Low-resolution Spectra with Pla | THE consortium Tier-1 screen we rebuild: selection strategy, band metric, four ML classifiers, in-simulator validation | mugnai2021_2110.00503.pdf |
| changeat2023 | Changeat 2022: ESA-Ariel Data Challenge NeurIPS 2022: Introduction to exo-atmospheric studies and pr | the ABC training grid: H2O, CH4, CO, CO2, NH3 only (sec 2.2) | ../ESA_Ariel_Data_Challenge_Changeat_Yip_2023.pdf |
| aubin2023 | Aubin et al. 2023: Simulation-based Inference for Exoplanet Atmospheric Retrieval: Insights from winning | ADC 2023 winner; seven targets = R, T and five abundances | aubin2023_2309.09337.pdf |
| mugnai2026 | Mugnai et al. 2026: A public dataset of Ariel simulated observations for developing exoplanetary atmosphe | instrument-only train/test shift; the nearest shift dataset | ../Mugnai_2026_Ariel_simulated_observations_dataset.pdf |
| tiers2026 | Radica et al. 2026: On the Information Content of Ariel Transmission Spectra: Reassessing the Tier System | tier binning R ~1/3/1 and 10/50/10; what each tier constrains | tiers2026_2604.07598.pdf |
| yakubu2026 | Yakubu 2026: Machine Learning and Deep Learning for Exoplanet Detection and Atmospheric Characteri | 2026 review: robustness under mismatch listed as open, physics-side mismatch absent | yakubu2026_2606.23766.pdf |
| yip2023 | yip2023 | lessons learned from ADC 2022 (PMLR 220; file ../Yip_2023_Lessons_Learned_ADC2022_PMLR_v220.pdf) | ../Yip_2023_Lessons_Learned_ADC2022_PMLR_v220.pdf |

## ML-exo

| key | paper | cited for | file |
|---|---|---|---|
| marquezneila2018 | Márquez-Neila et al. 2018: Supervised machine learning for analysing spectra of exoplanetary atmospheres | random forest retrieval; included HCN for WFC3 | ../1806.03944v1.pdf |
| zingales2018 | Zingales 2018: ExoGAN: Retrieving Exoplanetary Atmospheres Using Deep Convolutional Generative Adver | GAN retrieval; in-simulator validation | zingales2018_1806.02906.pdf |
| cobb2019 | Cobb et al. 2019: An Ensemble of Bayesian Neural Networks for Exoplanetary Atmospheric Retrieval | ensembles for retrieval | cobb2019_1905.10659.pdf |
| nixon2020 | Nixon 2020: Assessment of supervised machine learning for atmospheric retrieval of exoplanets | random-forest retrieval assessment; no OOD test | nixon2020_2004.10755.pdf |
| yip2021 | Yip et al. 2020: Peeking inside the Black Box: Interpreting Deep-learning Models for Exoplanet Atmosph | sensitivity of DNN retrievals to spectral features | yip2021_2011.11284.pdf |
| ardevol2022 | Martinez et al. 2022: Convolutional neural networks as an alternative to Bayesian retrievals | CLOSEST PRECEDENT: CNN tested on added/removed species and star spots; no remedy, no detection | ardevol2022_2203.01236.pdf |
| vasist2023 | Vasist et al. 2023: Neural posterior estimation for exoplanetary atmospheric retrieval | NPE retrieval; coverage diagnostics in-simulator | vasist2023_2301.06575.pdf |
| gebhard2024 | Gebhard et al. 2024: Flow Matching for Atmospheric Retrieval of Exoplanets: Where Reliability meets Adapti | FMPE retrieval with noise-level conditioning; state of the art in ML retrieval | gebhard2024_2410.21477.pdf |
| hayes2020 | Hayes et al. 2019: Optimizing exoplanet atmosphere retrieval using unsupervised machine-learning classif | PCA/k-means classes as retrieval priors | ../Hayes_2020_unsupervised_retrieval_classification.pdf |
| duque2025 | Duque-Castano et al. 2024: Machine-assisted classification of potential biosignatures in Earth-like exoplanets u | the MultiREx framework and a low-SNR classification screen | ../2407.19167v2.pdf |
| ares2024 | Jaziri et al. 2024: ARES. VI. Viability of one-dimensional retrieval models for transmission spectroscopy | 1D-retrieval biases from 3D effects: forward-model misspecification in classical retrieval | TITLE MISMATCH: ARES. VI. Viability of one-dimensional retrieval models for  |
| barstow2020 | arXiv:2002.01063 | retrieval-code intercomparison; forward-model choice priced for classical retrieval | barstow2020_2002.01063.pdf |

## chemistry

| key | paper | cited for | file |
|---|---|---|---|
| madhusudhan2012 | arXiv:1109.3183 | C/O > 1: HCN, C2H2 major constituents; tracers of C/O above 800 K | madhusudhan2012_1109.3183.pdf |
| moses2013 | Moses et al. 2012: CHEMICAL CONSEQUENCES OF THE C/O RATIO ON HOT JUPITERS: EXAMPLES FROM WASP-12b, CoRoT | C/O chemistry with quenching and photochemistry; disequilibrium enhances HCN, C2H2 | moses2013_1211.2996.pdf |
| zahnle2014 | Zahnle 2014: METHANE, CARBON MONOXIDE, AND AMMONIA IN BROWN DWARFS AND SELF-LUMINOUS GIANT PLANETS | the CO/CH4 quench timescale used in Axis 8 | zahnle2014_1408.6283.pdf |
| stock2018 | arXiv:1804.05010 | the equilibrium chemistry code behind the label | stock2018_1804.05010.pdf |
| kawashima2021 | Kawashima 2021: Implementation of disequilibrium chemistry to spectral retrieval 
code ARCiS and appl | equilibrium assumption biases retrieved C/O (ARCiS) | kawashima2021_2110.13443.pdf |
| baeyens2025 | Bardet et al. 2025: Re-analysis of 10 Hot-Jupiter Atmospheres with disequilibrium chemistry retrieval | re-analysis of ten hot Jupiters with disequilibrium retrieval | baeyens2025_2506.12806.pdf |

## physics

| key | paper | cited for | file |
|---|---|---|---|
| rackham2018 | arXiv:1711.05691 | contamination largest for M dwarfs: the expected host dependence | rackham2018_1711.05691.pdf |
| pinhas2018 | Pinhas et al. 2018: H2O abundances and cloud properties in ten hot giant exoplanets | retrieval of stellar heterogeneity with the spectrum | pinhas2018_1811.00011.pdf |
| lee2013 | Lee et al. 2013: ATMOSPHERIC RETRIEVAL ANALYSIS OF THE DIRECTLY IMAGED EXOPLANET HR 8799b | the Mie haze prescription (Lee et al. 2013) | ../Lee_2013_HR8799b_retrieval.pdf |

## codes

| key | paper | cited for | file |
|---|---|---|---|
| alrefaie2021 | Alrefaie et al. 2019: TauREx 3: A Fast, Dynamic, and Extendable Framework for Retrievals | the radiative-transfer core under MultiREx | ../Al-Refaie_2021_ApJ_917_37.pdf |
| kempton2017 | Kempton et al. 2016: Exo-Transmit: An Open-Source Code for Calculating Transmission Spectra for Exoplanet  | the independent code and the source of the CO/NH3/HCN/C2H2 tables | ../Kempton_2017_ExoTransmit.pdf |
| chubb2021 | Chubb et al. 2020: The ExoMolOP database: Cross sections and k-tables for molecules of interest in high- | the ExoMol cross sections of the opacity axis | ../Chubb_2021_ExoMolOP.pdf |
| freedman2014 | Freedman et al. 2014: GASEOUS MEAN OPACITIES FOR GIANT PLANET AND ULTRACOOL DWARF ATMOSPHERES OVER A RANGE  | the opacity compilation behind Exo-Transmit | ../Freedman_2014_gaseous_mean_opacities.pdf |
| lupu2014 | Lupu et al. 2014: THE ATMOSPHERES OF EARTHLIKE PLANETS AFTER GIANT IMPACT EVENTS | line-list sources of the Exo-Transmit tables (Table 2) | ../Lupu_2014_atmospheres_after_giant_impact.pdf |

## ML-methods

| key | paper | cited for | file |
|---|---|---|---|
| bishop1995 | bishop1995 | training with noise = Tikhonov regularization (why noise augmentation cannot undo a draw) | — |
| chen2020 | Chen et al. 2019: Invariance reduces Variance: Understanding Data Augmentation in Deep Learning and Bey | augmentation as group invariance | TITLE MISMATCH: Invariance reduces Variance: Understanding Data Augmentation |
| tobin2017 | Tobin et al. 2017: Domain randomization for transferring deep neural networks from simulation to the rea | the sim-to-real recipe we apply to spectra | tobin2017_1703.06907.pdf |
| hendrycks2017 | Hendrycks 2016: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Net | the softmax-confidence baseline | hendrycks2017_1610.02136.pdf |
| lee2018 | Lee et al. 2018: A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial  | Mahalanobis OOD score | lee2018_1807.03888.pdf |
| sun2022 | Sun et al. 2022: Out-of-distribution Detection with Deep Nearest Neighbors | k-NN distance OOD score | sun2022_2204.06507.pdf |
| lakshminarayanan2017 | Lakshminarayanan et al. 2016: Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | ensemble disagreement as uncertainty | lakshminarayanan2017_1612.01474.pdf |
| geifman2017 | Geifman 2017: Selective Classification for Deep Neural Networks | selective prediction / accuracy-coverage | geifman2017_1705.08500.pdf |
| jaeger2023 | Jaeger et al. 2022: A Call to Reflect on Evaluation Practices for Failure Detection in Image Classificati | OOD detection != failure detection; FD-Shifts; our detection inversion reproduces it | jaeger2023_2211.15259.pdf |
| hendrycks2019 | Hendrycks 2019: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations | held-out-corruption evaluation protocol | hendrycks2019_1903.12261.pdf |
| rusak2020 | arXiv:2001.06057 | noise augmentation and variance tuning | rusak2020_2001.06057.pdf |
| angelopoulos2021 | Angelopoulos 2021: A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quant | split conformal prediction (our baseline) | angelopoulos2021_2107.07511.pdf |
| tibshirani2019 | Tibshirani et al. 2019: Conformal Prediction Under Covariate Shift | why the conformal guarantee needs the deployment distribution | tibshirani2019_1904.06019.pdf |

## SBI

| key | paper | cited for | file |
|---|---|---|---|
| schmitt2021 | Schmitt et al. 2021: Detecting Model Misspecification in Amortized Bayesian Inference with Neural Networks | misspecification detection for amortized inference | schmitt2021_2112.08866.pdf |
| cannon2022 | Cannon et al. 2022: Investigating the Impact of Model Misspecification in Neural Simulation-based Inferen | impact of misspecification on neural SBI | cannon2022_2209.01845.pdf |
| huang2023 | Huang et al. 2023: Learning Robust Statistics for Simulation-based Inference under Model Misspecificatio | robust summaries under misspecification | huang2023_2305.15871.pdf |
| sbi2025a | Alvey et al. 2025: Simulation-based inference with deep ensembles: evaluating calibration uncertainty an | ensemble-based misspecification detection (cosmology) | sbi2025a_2507.13495.pdf |
| sbi2025b | Akhmetzhanova et al. 2025: Detecting model misspecification in cosmology with scale-dependent normalizing flows | flow-based OOD detection for cosmological SBI | sbi2025b_2508.05744.pdf |
