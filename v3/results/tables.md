## Table 2
| Model, features | Accuracy (%) | F1 (%) | Brier | ECE | AUC |
| :-- | --: | --: | --: | --: | --: |
| XGBoost, raw bins | 86.50 ± 0.80 | 86.91 ± 0.94 | 0.0935 | 0.037 | 0.947 |
| Random Forest, raw bins | 86.59 ± 0.93 | 87.11 ± 0.96 | 0.0908 | 0.042 | 0.948 |
| XGBoost, PCA | 91.83 ± 0.66 | 92.03 ± 0.68 | 0.0578 | 0.020 | 0.978 |
| Random Forest, PCA | 90.14 ± 0.79 | 90.27 ± 0.92 | 0.0944 | 0.145 | 0.969 |
| XGBoost, per-spectrum normalized | 96.05 ± 0.59 | 96.11 ± 0.57 | 0.0292 | 0.010 | 0.994 |
| Random Forest, per-spectrum normalized | 95.34 ± 0.59 | 95.44 ± 0.60 | 0.0379 | 0.028 | 0.990 |
| MLP, per-spectrum normalized | 95.43 ± 0.53 | 95.52 ± 0.53 | 0.0338 | 0.009 | 0.992 |
| MLP, PCA whitened | 86.80 ± 0.66 | 85.75 ± 0.66 | 0.1059 | 0.132 | 0.962 |
| MLP, PCA unwhitened | 84.98 ± 0.62 | 83.82 ± 0.74 | 0.1019 | 0.116 | 0.961 |

Feature dimensions: norm 102, raw 102, pca 71, pcaw 71. n_train = 18040. Best: norm_xgb.

## Resolution ladder
| Configuration | Bins | XGBoost, normalized (%) | MLP, normalized (%) | XGBoost, PCA (%) | Brier (XGBoost, normalized) | ECE |
| :-- | --: | --: | --: | --: | --: | --: |
| Ariel delivered (photometry + R 15 / 100 / 30) | 102 | 96.05 ± 0.59 | 95.43 ± 0.53 | 91.83 ± 0.66 | 0.0292 | 0.010 |

## Table 3
| Axis | Case | Δ accuracy (pts) | Δ Brier | Predicted positive rate | Amplitude ratio |
| :-- | :-- | --: | --: | --: | --: |
| clean | reference | +0.0 | +0.000 | 0.516 | 1.00 |
| absorbers | absorbers | -23.7 | +0.224 | 0.266 | 0.99 |
| absorbers | absorbers_quenched | -37.2 | +0.362 | 0.091 | 0.87 |
| cloud | cloud_1e1Pa | -32.1 | +0.227 | 0.575 | 0.03 |
| cloud | cloud_1e2Pa | -11.9 | +0.086 | 0.541 | 0.29 |
| cloud | cloud_1e3Pa | -5.5 | +0.042 | 0.552 | 0.59 |
| cloud | cloud_1e4Pa | -1.2 | +0.010 | 0.537 | 0.84 |
| cloud | cloud_1e5Pa | -0.1 | +0.001 | 0.519 | 0.96 |
| compound | compound_spots10_haze2e6_snr10 | -7.5 | +0.069 | 0.520 | 0.94 |
| compound | compound_spots10_haze3e7 | -8.2 | +0.075 | 0.542 | 1.01 |
| compound | compound_spots10_haze3e7_snr8 | -8.2 | +0.075 | 0.542 | 1.01 |
| compound | compound_spots20_haze3e7 | -10.9 | +0.098 | 0.518 | 1.29 |
| compound | compound_spots20_haze3e7_snr8 | -10.9 | +0.098 | 0.518 | 1.29 |
| exomol | exomol | -9.0 | +0.064 | 0.423 | 0.87 |
| exotransmit | exotransmit | -0.7 | +0.004 | 0.521 | 0.89 |
| haze | haze_1p0e10 | -34.4 | +0.270 | 0.280 | 0.45 |
| haze | haze_2p0e5 | -0.2 | +0.001 | 0.520 | 0.99 |
| haze | haze_2p0e6 | -1.3 | +0.011 | 0.538 | 0.93 |
| haze | haze_2p4e8 | -7.3 | +0.064 | 0.605 | 0.74 |
| haze | haze_3p0e7 | -5.1 | +0.047 | 0.584 | 0.85 |
| quenched | quenched | -14.9 | +0.150 | 0.321 | 0.88 |
| quenched | quenched_kzz10 | -16.9 | +0.172 | 0.298 | 0.86 |
| quenched | quenched_kzz11 | -20.3 | +0.202 | 0.261 | 0.84 |
| quenched | quenched_kzz7 | -11.5 | +0.116 | 0.360 | 0.91 |
| quenched | quenched_kzz8 | -12.9 | +0.131 | 0.345 | 0.89 |
| tlse | tlse_fac05 | -1.2 | +0.009 | 0.498 | 1.02 |
| tlse | tlse_fac10 | -2.4 | +0.019 | 0.485 | 1.04 |
| tlse | tlse_mixed | -6.7 | +0.062 | 0.510 | 0.95 |
| tlse | tlse_spots02 | -2.0 | +0.018 | 0.523 | 0.98 |
| tlse | tlse_spots05 | -4.5 | +0.040 | 0.519 | 0.95 |
| tlse | tlse_spots10 | -7.1 | +0.065 | 0.510 | 0.96 |
| tlse | tlse_spots20 | -10.7 | +0.098 | 0.495 | 1.07 |
| white noise | snr12 | -1.5 | +0.011 | 0.514 | 1.00 |
| white noise | snr10 | -2.5 | +0.018 | 0.514 | 1.00 |
| white noise | snr8 | -4.0 | +0.030 | 0.514 | 1.00 |
| white noise | snr5 | -6.6 | +0.048 | 0.500 | 1.00 |
| correlated noise | snr12 | -2.5 | +0.021 | 0.514 | 1.00 |
| correlated noise | snr10 | -4.4 | +0.038 | 0.509 | 1.00 |
| correlated noise | snr8 | -6.9 | +0.055 | 0.495 | 1.00 |
| correlated noise | snr5 | -11.6 | +0.096 | 0.492 | 1.00 |
| gain ramp | x0.25 | +0.0 | +0.001 | 0.514 | 1.00 |
| gain ramp | x0.5 | -0.3 | +0.002 | 0.516 | 1.00 |
| gain ramp | x1.0 | -1.0 | +0.007 | 0.520 | 1.01 |
| gain ramp | x2.0 | -2.6 | +0.021 | 0.524 | 1.03 |
| baseline offset | x0.25 | +0.0 | +0.000 | 0.516 | 1.00 |
| baseline offset | x0.5 | +0.0 | +0.000 | 0.516 | 1.00 |
| baseline offset | x1.0 | +0.0 | +0.000 | 0.516 | 1.00 |
| baseline offset | x2.0 | +0.0 | +0.000 | 0.516 | 1.00 |
| absolute noise | 20 ppm | -3.2 | +0.022 | 0.528 | 1.00 |
| absolute noise | 50 ppm | -5.0 | +0.036 | 0.533 | 1.00 |
| absolute noise | 100 ppm | -7.7 | +0.054 | 0.540 | 1.00 |
| absolute noise | 200 ppm | -10.5 | +0.074 | 0.546 | 1.00 |
| noise colouring | white snr15 | -1.8 | +0.014 | 0.507 | 1.00 |
| noise colouring | ariel snr15 | -0.4 | +0.003 | 0.512 | 1.00 |
| noise colouring | exosim snr15 | -0.3 | +0.002 | 0.514 | 1.00 |
| noise colouring | white snr10 | -3.0 | +0.023 | 0.510 | 1.00 |
| noise colouring | ariel snr10 | -0.9 | +0.007 | 0.519 | 1.00 |
| noise colouring | exosim snr10 | -1.0 | +0.007 | 0.519 | 1.00 |
| noise colouring | white snr7 | -4.9 | +0.036 | 0.504 | 1.00 |
| noise colouring | ariel snr7 | -2.6 | +0.019 | 0.520 | 1.00 |
| noise colouring | exosim snr7 | -2.7 | +0.020 | 0.520 | 1.00 |
| noise colouring | white snr5 | -6.4 | +0.047 | 0.505 | 1.00 |
| noise colouring | ariel snr5 | -3.8 | +0.029 | 0.528 | 1.00 |
| noise colouring | exosim snr5 | -4.0 | +0.030 | 0.528 | 1.00 |
| extrapolation | radius>15 vs control | -0.7 | +0.007 | nan | nan |
