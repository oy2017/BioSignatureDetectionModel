## Table 2
| Model, features | Accuracy (%) | F1 (%) | Brier | ECE | AUC |
| :-- | --: | --: | --: | --: | --: |
| XGBoost, raw bins | 85.93 ± 1.06 | 86.50 ± 0.95 | 0.0944 | 0.024 | 0.944 |
| Random Forest, raw bins | 86.19 ± 1.17 | 86.90 ± 1.16 | 0.0946 | 0.038 | 0.943 |
| XGBoost, PCA | 91.56 ± 0.54 | 91.70 ± 0.62 | 0.0589 | 0.017 | 0.977 |
| Random Forest, PCA | 90.22 ± 0.49 | 90.38 ± 0.44 | 0.0931 | 0.140 | 0.969 |
| XGBoost, per-spectrum normalized | 95.84 ± 0.50 | 95.91 ± 0.53 | 0.0304 | 0.012 | 0.993 |
| Random Forest, per-spectrum normalized | 95.53 ± 0.67 | 95.61 ± 0.70 | 0.0374 | 0.032 | 0.990 |
| MLP, per-spectrum normalized | 95.47 ± 0.58 | 95.56 ± 0.63 | 0.0332 | 0.011 | 0.991 |
| MLP, PCA whitened | 86.82 ± 0.73 | 86.43 ± 0.92 | 0.1074 | 0.133 | 0.961 |
| MLP, PCA unwhitened | 85.63 ± 0.62 | 84.66 ± 0.69 | 0.1047 | 0.116 | 0.960 |

Feature dimensions: pcaw 71, raw 102, norm 102, pca 71. n_train = 18040. Best: norm_xgb.

## Resolution ladder
| Configuration | Bins | XGBoost, normalized (%) | MLP, normalized (%) | XGBoost, PCA (%) | Brier (XGBoost, normalized) | ECE |
| :-- | --: | --: | --: | --: | --: | --: |
| Ariel delivered (photometry + R 15 / 100 / 30) | 102 | 95.84 ± 0.50 | 95.47 ± 0.58 | 91.56 ± 0.54 | 0.0304 | 0.012 |

## Table 3
| Axis | Case | Δ accuracy (pts) | Δ Brier | Predicted positive rate | Amplitude ratio |
| :-- | :-- | --: | --: | --: | --: |
| clean | reference | +0.0 | +0.000 | 0.517 | 1.00 |
| cloud | cloud_1e1Pa | -33.8 | +0.228 | 0.394 | 0.03 |
| cloud | cloud_1e2Pa | -12.9 | +0.088 | 0.427 | 0.29 |
| cloud | cloud_1e3Pa | -4.7 | +0.034 | 0.483 | 0.57 |
| cloud | cloud_1e4Pa | -1.1 | +0.009 | 0.502 | 0.80 |
| cloud | cloud_1e5Pa | -0.1 | +0.001 | 0.513 | 0.94 |
| exomol | exomol | -10.7 | +0.078 | 0.396 | 0.86 |
| exotransmit | exotransmit | -5.7 | +0.050 | 0.422 | 0.89 |
| haze | haze_1p0e10 | -38.3 | +0.324 | 0.140 | 0.47 |
| haze | haze_2p0e5 | -0.1 | +0.001 | 0.521 | 0.99 |
| haze | haze_2p0e6 | -0.8 | +0.007 | 0.534 | 0.95 |
| haze | haze_2p4e8 | -4.6 | +0.034 | 0.529 | 0.78 |
| haze | haze_3p0e7 | -2.9 | +0.026 | 0.558 | 0.89 |
| quenched | quenched | +2.4 | -0.015 | 0.514 | 0.99 |
| tlse | tlse_fac05 | -0.7 | +0.007 | 0.509 | 1.02 |
| tlse | tlse_fac10 | -2.0 | +0.018 | 0.499 | 1.04 |
| tlse | tlse_mixed | -6.0 | +0.055 | 0.477 | 0.96 |
| tlse | tlse_spots02 | -1.6 | +0.015 | 0.507 | 0.98 |
| tlse | tlse_spots05 | -3.9 | +0.036 | 0.492 | 0.97 |
| tlse | tlse_spots10 | -6.5 | +0.059 | 0.473 | 0.97 |
| tlse | tlse_spots20 | -9.7 | +0.089 | 0.435 | 1.08 |
| white noise | snr12 | -1.7 | +0.013 | 0.509 | 1.00 |
| white noise | snr10 | -2.6 | +0.021 | 0.498 | 1.00 |
| white noise | snr8 | -4.5 | +0.035 | 0.490 | 1.00 |
| white noise | snr5 | -8.3 | +0.063 | 0.453 | 1.00 |
| correlated noise | snr12 | -2.3 | +0.021 | 0.508 | 1.00 |
| correlated noise | snr10 | -4.4 | +0.039 | 0.499 | 1.00 |
| correlated noise | snr8 | -6.3 | +0.055 | 0.479 | 1.00 |
| correlated noise | snr5 | -12.6 | +0.105 | 0.438 | 1.00 |
| gain ramp | x0.25 | -0.1 | +0.002 | 0.515 | 1.00 |
| gain ramp | x0.5 | -0.8 | +0.005 | 0.513 | 1.01 |
| gain ramp | x1.0 | -2.2 | +0.020 | 0.508 | 1.01 |
| gain ramp | x2.0 | -5.3 | +0.048 | 0.494 | 1.03 |
| baseline offset | x0.25 | +0.0 | +0.000 | 0.517 | 1.00 |
| baseline offset | x0.5 | +0.0 | +0.000 | 0.517 | 1.00 |
| baseline offset | x1.0 | +0.0 | +0.000 | 0.517 | 1.00 |
| baseline offset | x2.0 | +0.0 | +0.000 | 0.517 | 1.00 |
| absolute noise | 20 ppm | -3.9 | +0.026 | 0.512 | 1.00 |
| absolute noise | 50 ppm | -5.5 | +0.038 | 0.500 | 1.00 |
| absolute noise | 100 ppm | -8.1 | +0.053 | 0.492 | 1.00 |
| absolute noise | 200 ppm | -10.7 | +0.072 | 0.483 | 1.00 |
| noise colouring | white snr15 | -2.2 | +0.015 | 0.510 | 1.00 |
| noise colouring | ariel snr15 | -0.1 | +0.000 | 0.516 | 1.00 |
| noise colouring | exosim snr15 | -0.2 | +0.001 | 0.515 | 1.00 |
| noise colouring | white snr10 | -3.3 | +0.027 | 0.498 | 1.00 |
| noise colouring | ariel snr10 | -1.4 | +0.011 | 0.496 | 1.00 |
| noise colouring | exosim snr10 | -1.5 | +0.011 | 0.493 | 1.00 |
| noise colouring | white snr7 | -6.0 | +0.045 | 0.477 | 1.00 |
| noise colouring | ariel snr7 | -3.4 | +0.026 | 0.473 | 1.00 |
| noise colouring | exosim snr7 | -3.5 | +0.028 | 0.469 | 1.00 |
| noise colouring | white snr5 | -8.2 | +0.061 | 0.453 | 1.00 |
| noise colouring | ariel snr5 | -5.4 | +0.041 | 0.454 | 1.00 |
| noise colouring | exosim snr5 | -5.5 | +0.042 | 0.450 | 1.00 |
| extrapolation | radius>15 vs control | -0.4 | +0.005 | nan | nan |
