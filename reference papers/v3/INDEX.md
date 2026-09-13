# References for the v3 paper — what each is cited for

Files: this directory (new downloads, named key_arxivid.pdf) or `../` (already in reference papers/). Every arXiv id was checked against its title (API or first page).

## Ariel

| key | paper | cited for | file |
|---|---|---|---|
| tinetti2018 | tinetti2018 | the mission, its science goals and the tier concept (Exp. Astron. 46, 135; no arXiv; file ../s10686-018-9598-x.pdf) | ../s10686-018-9598-x.pdf |
| edwards2019 | arXiv:1905.04959 | the target list, tier definitions by SNR, number of transits | ../Edwards_2019_AJ_157_242.pdf |
| edwards2022 | arXiv:2205.05073 | current MCS; the list our real-target test set is built from | ../Ariel_Mission_Reference_Sample_Edwards_2022.pdf |
| mugnai2020 | arXiv:2009.07824 | the radiometric model behind the tier SNR requirement | ../Mugnai_2020_ArielRad.pdf |
| mugnai2025 | arXiv:2501.12809 | the consortium simulator whose noise shape we adopt | mugnai2025_2501.12809.pdf |
| changeat2020 | arXiv:2003.01839 | population retrievals of the target list with TauREx + ArielRad | ../Changeat_2020_AJ_160_80.pdf |
| mugnai2021 | arXiv:2110.00503 | THE consortium Tier-1 screen we rebuild: selection strategy, band metric, four ML classifiers, in-simulator validation | mugnai2021_2110.00503.pdf |
| changeat2023 | arXiv:2206.14633 | the ABC training grid: H2O, CH4, CO, CO2, NH3 only (sec 2.2) | ../ESA_Ariel_Data_Challenge_Changeat_Yip_2023.pdf |
| aubin2023 | arXiv:2309.09337 | ADC 2023 winner; seven targets = R, T and five abundances | aubin2023_2309.09337.pdf |
| mugnai2026 | arXiv:2605.03719 | instrument-only train/test shift; the nearest shift dataset | ../Mugnai_2026_Ariel_simulated_observations_dataset.pdf |
| tiers2026 | arXiv:2604.07598 | tier binning R ~1/3/1 and 10/50/10; what each tier constrains | tiers2026_2604.07598.pdf |
| yakubu2026 | arXiv:2606.23766 | 2026 review: robustness under mismatch listed as open, physics-side mismatch absent | yakubu2026_2606.23766.pdf |
| yip2023 | yip2023 | lessons learned from ADC 2022 (PMLR 220; file ../Yip_2023_Lessons_Learned_ADC2022_PMLR_v220.pdf) | ../Yip_2023_Lessons_Learned_ADC2022_PMLR_v220.pdf |

## ML-exo

| key | paper | cited for | file |
|---|---|---|---|
| marquezneila2018 | arXiv:1806.03944 | random forest retrieval; included HCN for WFC3 | ../1806.03944v1.pdf |
| zingales2018 | arXiv:1806.02906 | GAN retrieval; in-simulator validation | zingales2018_1806.02906.pdf |
| cobb2019 | arXiv:1905.10659 | ensembles for retrieval | cobb2019_1905.10659.pdf |
| nixon2020 | arXiv:2004.10755 | random-forest retrieval assessment; no OOD test | nixon2020_2004.10755.pdf |
| yip2021 | arXiv:2011.11284 | sensitivity of DNN retrievals to spectral features | yip2021_2011.11284.pdf |
| ardevol2022 | arXiv:2203.01236 | CLOSEST PRECEDENT: CNN tested on added/removed species and star spots; no remedy, no detection | ardevol2022_2203.01236.pdf |
| vasist2023 | arXiv:2301.06575 | NPE retrieval; coverage diagnostics in-simulator | vasist2023_2301.06575.pdf |
| gebhard2024 | arXiv:2410.21477 | FMPE retrieval with noise-level conditioning; state of the art in ML retrieval | gebhard2024_2410.21477.pdf |
| hayes2020 | arXiv:1909.00718 | PCA/k-means classes as retrieval priors | ../Hayes_2020_unsupervised_retrieval_classification.pdf |
| duque2025 | arXiv:2407.19167 | the MultiREx framework and a low-SNR classification screen | ../2407.19167v2.pdf |
| ares2024 | arXiv:2401.03809 | 1D-retrieval biases from 3D effects: forward-model misspecification in classical retrieval | ares2024_2401.03809.pdf |
| barstow2020 | arXiv:2002.01063 | retrieval-code intercomparison; forward-model choice priced for classical retrieval | barstow2020_2002.01063.pdf |

## chemistry

| key | paper | cited for | file |
|---|---|---|---|
| madhusudhan2012 | arXiv:1109.3183 | C/O > 1: HCN, C2H2 major constituents; tracers of C/O above 800 K | madhusudhan2012_1109.3183.pdf |
| moses2013 | arXiv:1211.2996 | C/O chemistry with quenching and photochemistry; disequilibrium enhances HCN, C2H2 | moses2013_1211.2996.pdf |
| zahnle2014 | arXiv:1408.6283 | the CO/CH4 quench timescale used in Axis 8 | zahnle2014_1408.6283.pdf |
| stock2018 | arXiv:1804.05010 | the equilibrium chemistry code behind the label | stock2018_1804.05010.pdf |
| kawashima2021 | arXiv:2110.13443 | equilibrium assumption biases retrieved C/O (ARCiS) | kawashima2021_2110.13443.pdf |
| baeyens2025 | arXiv:2506.12806 | re-analysis of ten hot Jupiters with disequilibrium retrieval | baeyens2025_2506.12806.pdf |

## physics

| key | paper | cited for | file |
|---|---|---|---|
| rackham2018 | arXiv:1711.05691 | contamination largest for M dwarfs: the expected host dependence | rackham2018_1711.05691.pdf |
| pinhas2018 | arXiv:1811.00011 | retrieval of stellar heterogeneity with the spectrum | pinhas2018_1811.00011.pdf |
| lee2013 | arXiv:1307.1404 | the Mie haze prescription (Lee et al. 2013) | ../Lee_2013_HR8799b_retrieval.pdf |

## codes

| key | paper | cited for | file |
|---|---|---|---|
| alrefaie2021 | arXiv:1912.07759 | the radiative-transfer core under MultiREx | ../Al-Refaie_2021_ApJ_917_37.pdf |
| kempton2017 | arXiv:1611.03871 | the independent code and the source of the CO/NH3/HCN/C2H2 tables | ../Kempton_2017_ExoTransmit.pdf |
| chubb2021 | arXiv:2009.00687 | the ExoMol cross sections of the opacity axis | ../Chubb_2021_ExoMolOP.pdf |
| freedman2014 | arXiv:1409.0026 | the opacity compilation behind Exo-Transmit | ../Freedman_2014_gaseous_mean_opacities.pdf |
| lupu2014 | arXiv:1401.1499 | line-list sources of the Exo-Transmit tables (Table 2) | ../Lupu_2014_atmospheres_after_giant_impact.pdf |

## ML-methods

| key | paper | cited for | file |
|---|---|---|---|
| bishop1995 | bishop1995 | training with noise = Tikhonov regularization (why noise augmentation cannot undo a draw) | — |
| chen2020 | arXiv:1907.10905 | augmentation as group invariance | chen2020_1907.10905.pdf |
| tobin2017 | arXiv:1703.06907 | the sim-to-real recipe we apply to spectra | tobin2017_1703.06907.pdf |
| hendrycks2017 | arXiv:1610.02136 | the softmax-confidence baseline | hendrycks2017_1610.02136.pdf |
| lee2018 | arXiv:1807.03888 | Mahalanobis OOD score | lee2018_1807.03888.pdf |
| sun2022 | arXiv:2204.06507 | k-NN distance OOD score | sun2022_2204.06507.pdf |
| lakshminarayanan2017 | arXiv:1612.01474 | ensemble disagreement as uncertainty | lakshminarayanan2017_1612.01474.pdf |
| geifman2017 | arXiv:1705.08500 | selective prediction / accuracy-coverage | geifman2017_1705.08500.pdf |
| jaeger2023 | arXiv:2211.15259 | OOD detection != failure detection; FD-Shifts; our detection inversion reproduces it | jaeger2023_2211.15259.pdf |
| hendrycks2019 | arXiv:1903.12261 | held-out-corruption evaluation protocol | hendrycks2019_1903.12261.pdf |
| rusak2020 | arXiv:2001.06057 | noise augmentation and variance tuning | rusak2020_2001.06057.pdf |
| angelopoulos2021 | arXiv:2107.07511 | split conformal prediction (our baseline) | angelopoulos2021_2107.07511.pdf |
| tibshirani2019 | arXiv:1904.06019 | why the conformal guarantee needs the deployment distribution | tibshirani2019_1904.06019.pdf |

## SBI

| key | paper | cited for | file |
|---|---|---|---|
| schmitt2021 | arXiv:2112.08866 | misspecification detection for amortized inference | schmitt2021_2112.08866.pdf |
| cannon2022 | arXiv:2209.01845 | impact of misspecification on neural SBI | cannon2022_2209.01845.pdf |
| huang2023 | arXiv:2305.15871 | robust summaries under misspecification | huang2023_2305.15871.pdf |
| sbi2025a | arXiv:2507.13495 | ensemble-based misspecification detection (cosmology) | sbi2025a_2507.13495.pdf |
| sbi2025b | arXiv:2508.05744 | flow-based OOD detection for cosmological SBI | sbi2025b_2508.05744.pdf |
