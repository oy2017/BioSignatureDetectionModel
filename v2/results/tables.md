## Table 2
| Model, features | Accuracy (%) | F1 (%) | Brier | ECE | AUC |
| :-- | --: | --: | --: | --: | --: |
| XGBoost, raw bins | 72.56 ± 0.49 | 73.35 ± 0.78 | 0.1810 | 0.062 | 0.812 |
| Random Forest, raw bins | 72.24 ± 0.96 | 73.25 ± 0.71 | 0.1866 | 0.058 | 0.799 |
| XGBoost, PCA | 83.90 ± 1.03 | 83.99 ± 1.07 | 0.1188 | 0.064 | 0.925 |
| Random Forest, PCA | 80.70 ± 1.07 | 80.89 ± 1.07 | 0.1558 | 0.141 | 0.894 |
| XGBoost, per-spectrum normalized | 90.33 ± 0.42 | 90.29 ± 0.38 | 0.0686 | 0.038 | 0.975 |
| Random Forest, per-spectrum normalized | 89.08 ± 0.36 | 88.89 ± 0.39 | 0.0797 | 0.064 | 0.967 |
| norm_mlp | 90.14 ± 0.34 | 89.91 ± 0.37 | 0.0668 | 0.012 | 0.973 |
| MLP, PCA whitened | 73.58 ± 0.84 | 77.73 ± 0.52 | 0.1657 | 0.115 | 0.871 |
| MLP, PCA unwhitened | 73.71 ± 0.89 | 77.26 ± 0.65 | 0.1631 | 0.111 | 0.876 |

Feature dimensions: pca 76, norm 102, pcaw 76, raw 102. n_train = 18156. Best: norm_xgb.

## Resolution ladder
| Configuration | Bins | XGBoost, normalized (%) | MLP, normalized (%) | XGBoost, PCA (%) | Brier (XGBoost, normalized) | ECE |
| :-- | --: | --: | --: | --: | --: | --: |
| Ariel delivered (photometry + R 15 / 100 / 30) | 102 | 90.33 ± 0.42 | 90.14 ± 0.34 | 83.90 ± 1.03 | 0.0686 | 0.038 |
| uniform R = 100 | 275 | 90.75 ± 0.37 | 90.87 ± 0.54 | 84.62 ± 0.39 | 0.0651 | 0.037 |
| uniform R = 200 (earlier study) | 550 | 92.14 ± 0.42 | 92.09 ± 0.94 | 85.50 ± 0.94 | 0.0573 | 0.027 |

## Table 3
| Axis | Case | Δ accuracy (pts) | Δ Brier | Predicted positive rate | Amplitude ratio |
| :-- | :-- | --: | --: | --: | --: |
| clean | reference | +0.0 | +0.000 | 0.495 | 1.00 |
| cloud | cloud_1e1Pa | -32.2 | +0.324 | 0.089 | 0.04 |
| cloud | cloud_1e2Pa | -20.1 | +0.194 | 0.229 | 0.37 |
| cloud | cloud_1e3Pa | -9.4 | +0.088 | 0.349 | 0.67 |
| cloud | cloud_1e4Pa | -2.5 | +0.021 | 0.446 | 0.87 |
| cloud | cloud_1e5Pa | -0.2 | +0.001 | 0.489 | 0.97 |
| exomol | exomol | -1.7 | +0.011 | 0.540 | 0.92 |
| exomol | exomol_o3 | -15.3 | +0.139 | 0.339 | 0.86 |
| exotransmit | exotransmit | -1.0 | +0.008 | 0.499 | 0.89 |
| haze | haze_1p0e10 | -40.4 | +0.431 | 0.000 | 0.37 |
| haze | haze_2p0e5 | -0.5 | +0.003 | 0.475 | 0.99 |
| haze | haze_2p0e6 | -7.2 | +0.068 | 0.367 | 0.92 |
| haze | haze_2p4e8 | -36.9 | +0.380 | 0.035 | 0.60 |
| haze | haze_3p0e7 | -28.2 | +0.278 | 0.124 | 0.76 |
| tlse | tlse_fac05 | -2.3 | +0.020 | 0.545 | 1.02 |
| tlse | tlse_fac10 | -4.9 | +0.041 | 0.580 | 1.04 |
| tlse | tlse_mixed | -15.9 | +0.157 | 0.271 | 0.95 |
| tlse | tlse_spots02 | -4.1 | +0.040 | 0.420 | 0.98 |
| tlse | tlse_spots05 | -10.6 | +0.102 | 0.335 | 0.96 |
| tlse | tlse_spots10 | -17.8 | +0.177 | 0.249 | 0.94 |
| tlse | tlse_spots20 | -26.3 | +0.265 | 0.155 | 1.05 |
| white noise | snr12 | -2.5 | +0.020 | 0.479 | 1.10 |
| white noise | snr10 | -5.6 | +0.042 | 0.460 | 1.16 |
| white noise | snr8 | -10.4 | +0.078 | 0.407 | 1.27 |
| white noise | snr5 | -25.0 | +0.195 | 0.227 | 1.65 |
| correlated noise | snr12 | -6.7 | +0.058 | 0.473 | 1.10 |
| correlated noise | snr10 | -11.5 | +0.102 | 0.439 | 1.16 |
| correlated noise | snr8 | -18.1 | +0.160 | 0.396 | 1.25 |
| correlated noise | snr5 | -28.3 | +0.259 | 0.272 | 1.62 |
| gain ramp | x0.25 | -0.4 | +0.004 | 0.494 | 1.04 |
| gain ramp | x0.5 | -2.0 | +0.017 | 0.495 | 1.05 |
| gain ramp | x1.0 | -4.9 | +0.047 | 0.495 | 1.07 |
| gain ramp | x2.0 | -15.1 | +0.138 | 0.492 | 1.13 |
| baseline offset | x0.25 | +0.0 | +0.000 | 0.495 | 1.05 |
| baseline offset | x0.5 | +0.0 | +0.000 | 0.495 | 1.05 |
| baseline offset | x1.0 | +0.0 | +0.000 | 0.495 | 1.05 |
| baseline offset | x2.0 | +0.0 | +0.000 | 0.495 | 1.05 |
| absolute noise | 20 ppm | -3.3 | +0.038 | 0.445 | 1.00 |
| absolute noise | 50 ppm | -6.2 | +0.065 | 0.414 | 1.00 |
| absolute noise | 100 ppm | -8.7 | +0.089 | 0.388 | 1.01 |
| absolute noise | 200 ppm | -12.3 | +0.123 | 0.346 | 1.05 |
| noise colouring | white snr15 | -0.8 | +0.007 | 0.496 | 1.04 |
| noise colouring | ariel snr15 | -0.1 | +0.001 | 0.496 | 1.05 |
| noise colouring | white snr10 | -2.7 | +0.021 | 0.486 | 1.10 |
| noise colouring | ariel snr10 | -1.6 | +0.012 | 0.484 | 1.11 |
| noise colouring | white snr7 | -6.9 | +0.051 | 0.447 | 1.20 |
| noise colouring | ariel snr7 | -5.3 | +0.037 | 0.446 | 1.20 |
| noise colouring | white snr5 | -13.2 | +0.100 | 0.371 | 1.35 |
| noise colouring | ariel snr5 | -11.5 | +0.085 | 0.366 | 1.37 |
| extrapolation | radius>15 vs control | -0.6 | +0.005 | nan | nan |
