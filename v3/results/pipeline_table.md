Designs on the real panel
| Design | Right / wrong / split or withheld | Score of 66, JWST precision | Gain over published (95 % interval) | Score of 50, labels passing the mask control | Score of 66, Tier-1 noise | Gain over published (95 % interval) | Methane found of 8, JWST / Tier-1 |
| :-- | :-- | --: | :-- | --: | --: | :-- | :-- |
| Calling every label absent | 38 / 26 / 0 | 38.0 | −5.5 | 34.0 | 38.0 | −1.7 | 0 / 0 |
| Published classifier, all inputs | 39 / 16 / 9 | 43.5 |  | 38.5 | 39.7 |  | 0 / 0.0 |
| No-optical input everywhere | 45 / 14 / 5 | 47.5 | +4.0 (−0.4 to 9.3) | 40.0 | 41.0 | +1.3 (0.3 to 2.6) | 0 / 0.1 |
| AIRS-only input everywhere | 50 / 13 / 1 | 50.5 | +7.0 (0.5 to 13.0) | 43.5 | 41.4 | +1.7 (−0.3 to 3.8) | 7 / 1.7 |
| Margin rule → AIRS-only input | 44 / 15 / 5 | 46.5 | +3.0 (0.0 to 6.2) | 40.0 | 39.7 | +0.1 (−0.4 to 0.5) | 1 / 0.2 |
| Ensemble rule → AIRS-only input | 46 / 17 / 1 | 46.5 | +3.0 (0.4 to 6.2) | 40.0 | 40.3 | +0.7 (−0.1 to 1.5) | 2 / 0.8 |
| k-NN distance rule → AIRS-only input | 42 / 16 / 6 | 45.0 | +1.5 (−1.5 to 5.4) | 40.5 | 40.7 | +1.1 (−0.4 to 2.7) | 3 / 1.5 |
| Mahalanobis rule → AIRS-only input | 44 / 14 / 6 | 47.0 | +3.5 (0.0 to 7.7) | 42.0 | 40.9 | +1.2 (−0.3 to 2.9) | 5 / 1.6 |
| PCA rule → AIRS-only input | 44 / 15 / 5 | 46.5 | +3.0 (−0.6 to 7.4) | 41.5 | 40.5 | +0.8 (−0.0 to 1.9) | 5 / 1.1 |
| Shape flag → AIRS-only input | 47 / 12 / 5 | 49.5 | +6.0 (1.0 to 10.9) | 42.5 | 41.3 | +1.7 (0.3 to 3.2) | 5 / 1.3 |
| Haze-retrained design everywhere | 47 / 14 / 3 | 48.5 | +5.0 (0.0 to 10.3) | 41.5 | 41.1 | +1.5 (0.1 to 2.8) | 2 / 0.3 |
| Haze+deck-retrained design everywhere | 48 / 12 / 4 | 50.0 | +6.5 (1.1 to 12.1) | 42.0 | 41.6 | +2.0 (0.3 to 3.5) | 2 / 0.6 |
| The routing method | 50 / 11 / 3 | 51.5 | +8.0 (2.2 to 13.7) | 43.5 | 41.9 | +2.3 (0.9 to 3.8) | 5 / 1.2 |
| Amplitude flag, AIRS-only input elsewhere | 52 / 12 / 0 | 52.0 | +8.5 (1.8 to 14.8) | 44.0 | 41.9 | +2.3 (0.4 to 4.2) | 7 / 1.6 |
| Muting census in place of the amplitude flag | 51 / 8 / 5 | 53.5 | +10.0 (4.4 to 15.4) | 44.5 | 42.8 | +3.1 (1.5 to 5.0) | 6 / 1.3 |

Designs in simulation
| Design | Clean | Haze 3 × 10⁷ | Confined haze | Deck 10² Pa | Spots 20 % | Noise × 2 | Muted × 3 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Published classifier, all inputs | 81.2 | 63.5 | 67.1 | 71.4 | 70.4 | 71.9 | 67.7 |
| No-optical input everywhere | 77.4 | 71.8 | 74.9 | 72.3 | 74.2 | 69.6 | 66.4 |
| AIRS-only input everywhere | 73.8 | 73.3 | 73.6 | 72.1 | 74.0 | 68.3 | 65.4 |
| Margin rule → AIRS-only input | 79.9 | 63.8 | 68.5 | 71.9 | 70.9 | 71.1 | 67.5 |
| Ensemble rule → AIRS-only input | 80.0 | 66.4 | 71.4 | 72.5 | 72.0 | 71.3 | 67.6 |
| k-NN distance rule → AIRS-only input | 80.9 | 70.3 | 66.9 | 71.1 | 72.4 | 71.1 | 66.4 |
| Mahalanobis rule → AIRS-only input | 80.7 | 70.8 | 67.3 | 71.2 | 72.6 | 71.3 | 66.5 |
| PCA rule → AIRS-only input | 80.6 | 70.9 | 67.8 | 71.3 | 73.6 | 71.4 | 67.2 |
| Shape flag → AIRS-only input | 76.6 | 73.3 | 72.7 | 71.8 | 73.9 | 70.2 | 66.7 |
| Haze-retrained design everywhere | 80.1 | 73.2 | 73.9 | 73.3 | 75.7 | 71.7 | 67.5 |
| Haze+deck-retrained design everywhere | 80.2 | 73.6 | 75.4 | 74.6 | 76.1 | 71.7 | 67.7 |
| The routing method | 76.4 | 72.2 | 72.9 | 71.6 | 73.2 | 69.8 | 66.9 |
| Amplitude flag, AIRS-only input elsewhere | 73.8 | 72.2 | 73.6 | 72.0 | 73.3 | 67.9 | 65.8 |

Targets lost at the 60 % selection, CH4 / H2O
| Design | Clean | Haze 3 × 10⁷ | Confined haze | Deck 10² Pa | Spots 20 % | Noise × 2 | Muted × 3 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Published classifier, all inputs | 20 / 51 | 38 / 78 | 18 / 84 | 29 / 88 | 29 / 75 | 37 / 67 | 56 / 97 |
| No-optical input everywhere | 18 / 62 | 23 / 69 | 17 / 71 | 33 / 86 | 24 / 58 | 44 / 81 | 54 / 111 |
| AIRS-only input everywhere | 27 / 82 | 27 / 85 | 27 / 83 | 34 / 99 | 26 / 81 | 57 / 104 | 61 / 114 |
| Optical-excess score → AIRS-only input | 19 / 56 | 26 / 85 | 28 / 97 | 30 / 87 | 28 / 73 | 37 / 69 | 56 / 95 |
| Shape flag → AIRS-only input | 24 / 67 | 26 / 85 | 27 / 95 | 33 / 88 | 28 / 74 | 38 / 74 | 60 / 93 |
| Haze-retrained design everywhere | 17 / 56 | 21 / 78 | 19 / 78 | 26 / 78 | 23 / 67 | 45 / 74 | 56 / 101 |
| Haze+deck-retrained design everywhere | 21 / 53 | 24 / 79 | 20 / 76 | 28 / 84 | 24 / 67 | 41 / 82 | 58 / 99 |
| The routing method | 20 / 67 | 27 / 84 | 24 / 93 | 32 / 86 | 27 / 71 | 38 / 74 | 57 / 96 |
| Amplitude flag, AIRS-only input elsewhere | 22 / 80 | 28 / 84 | 24 / 82 | 30 / 96 | 26 / 77 | 57 / 104 | 61 / 107 |
| k-NN distance rule → AIRS-only input | 19 / 54 | 33 / 98 | 28 / 83 | 32 / 88 | 26 / 70 | 44 / 82 | 63 / 108 |
| Mahalanobis rule → AIRS-only input | 20 / 54 | 34 / 98 | 26 / 83 | 36 / 88 | 25 / 73 | 39 / 79 | 65 / 102 |

Table D1
| Reduction | Measured bins | Ratio | Upper threshold | Lower cut | Flag | Muting (census) | Most the prior allows | Beyond the prior |
| :-- | --: | --: | --: | --: | :-- | --: | --: | :-- |
| TOI-270d | 87 | 2.84 | 5.74 | 0.49 | none | 0.61 (0.57-0.66) | 1.97 | no |
| K2-18b | 99 | 1.49 | 5.63 | 0.49 | none | 0.72 (0.66-0.83) | 2.62 | no |
| K2-18b-alt | 88 | 3.34 | 5.44 | 0.49 | none | 1.71 (1.22-2.57) | 2.65 | no |
| TOI-421b | 89 | 5.39 | 5.78 | 0.50 | none | 3.39 (2.33-5.40) | 3.67 | no |
| HAT-P-18b | 48 | 7.99 | 6.13 | 0.59 | upper | 3.70 (3.22-4.57) | 3.27 | marginal |
| WASP-107b | 67 | 8.77 | 7.14 | 0.53 | upper | 4.00 (3.91-4.08) | 2.18 | yes |
| HD189733b | 56 | 5.09 | 8.88 | 0.54 | none | 1.73 (1.69-1.79) | 2.25 | no |
| HAT-P-26b | 35 | 5.71 | 8.33 | 0.48 | none | 5.32 (5.12-5.56) | 2.37 | yes |
| GJ3470b | 53 | 11.70 | 8.66 | 0.53 | upper | 3.45 (3.14-3.96) | 1.95 | yes |
| WASP-166b | 90 | 6.20 | 5.80 | 0.50 | upper | 3.07 (2.65-3.81) | 1.78 | yes |
| TOI-1231b | 74 | 5.44 | 5.09 | 0.45 | upper | 1.55 (1.30-2.02) | 1.88 | no |
| K2-18b-hu | 88 | 3.92 | 5.44 | 0.49 | none | 2.00 (1.79-2.36) | 2.65 | no |
| TOI-270d-hm | 41 | 5.48 | 8.36 | 0.48 | none | 1.04 (0.92-1.29) | 1.92 | no |
| GJ9827d | 50 | no dispersion | 6.18 | 0.60 | none | no dispersion | 3.42 | undefined |
| WASP-52b | 37 | 5.39 | 8.46 | 0.48 | none | 2.26 (2.17-2.36) | 2.39 | no |
| L98-59d | 37 | 10.57 | 8.46 | 0.48 | upper | no twin |  | no twin |
| GJ1214b | 38 | 63.44 | 8.31 | 0.48 | upper | no twin |  | no twin |
| LTT3780c | 97 | 2.13 | 5.58 | 0.48 | none | 0.99 (0.77-1.45) | 2.69 | no |
| WASP-80b | 49 | 11.93 | 9.20 | 0.57 | upper | 2.89 (2.52-3.63) | 1.66 | yes |
| HD209458b | 58 | 8.12 | 8.91 | 0.54 | none | 2.35 (2.28-2.44) | 1.33 | yes |
| WASP-96b | 88 | 3.36 | 5.44 | 0.49 | none | 1.49 (1.39-1.63) | 1.94 | no |
| WASP-17b | 63 | 7.23 | 6.07 | 0.57 | upper | 3.99 (3.84-4.17) | 1.96 | yes |
| WASP-94Ab | 83 | 8.87 | 5.85 | 0.48 | upper | 3.30 (3.09-3.59) | 1.47 | yes |
| HIP67522b | 41 | 4.28 | 8.36 | 0.48 | none | no twin |  | no twin |
| HAT-P-12b | 103 | 11.37 | 5.39 | 0.48 | upper | 5.34 (4.92-5.90) | 1.98 | yes |
| WASP-39b | 103 | 6.94 | 5.23 | 0.47 | upper | 3.71 (3.65-3.78) | 2.04 | yes |

Table D2
| Spectra | Optical-excess score | Input disagreement | First level of the nest | Second level of the nest | Shape flag | Amplitude flag, upper | Amplitude flag, lower |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Clean | 10 | 9 | 9 | 9 | 17 | 6.8 | 1.0 |
| Column-filling haze, 2 × 10⁶ m⁻³ | 56 | 20 | 25 | 9 | 60 | 8.6 | 2.1 |
| Column-filling haze, 3 × 10⁷ m⁻³ | 100 | 36 | 35 | 16 | 100 | 3.9 | 4.1 |
| Lee-Mie haze below 1 Pa | 88 | 25 | 30 | 8 | 90 | 10.4 | 0.8 |
| Power-law haze, log₁₀ a ≈ 3.1 | 76 | 20 | 22 | 10 | 79 | 12.9 | 0.8 |
| Power-law haze, log₁₀ a ≈ 4.1 | 97 | 23 | 20 | 15 | 97 | 12.0 | 1.0 |
| Spots 20 % | 66 | 21 | 21 | 12 | 68 | 4.4 | 4.1 |
| Spots 20 % + haze 3 × 10⁷ m⁻³ | 100 | 40 | 35 | 23 | 100 | 0.1 | 8.3 |
| Deck 10² Pa | 1 | 5 | 5 | 8 | 6 | 15.5 | 0.8 |
| Noise × 2 | 3 | 6 | 6 | 7 | 8 | 0.0 | 1.3 |
| Noise × 3 | 2 | 5 | 5 | 7 | 7 | 0.0 | 24.6 |
| Collision-induced absorption | 10 | 9 | 9 | 9 | 17 | 7.1 | 1.0 |

Flag rates
          shape_flagged  disagree_flagged  d1_flagged  d2_flagged
case                                                             
clean              17.3               9.0         9.0         9.1
haze2e6            60.4              20.2        25.4         9.0
haze3e7            99.8              36.2        34.5        16.0
hazetop1           89.8              24.6        30.2         8.3
plhaze29           78.6              20.1        21.5        10.5
plhaze28           97.5              23.2        19.9        15.2
spots20            68.0              21.2        21.5        12.4
compound           99.9              40.2        34.8        23.3
cloud1e2            6.4               5.2         4.7         8.1
white_x2            8.2               5.7         6.0         7.4
white_x3            6.6               4.7         4.8         7.0
cia                17.1               9.1         8.9         8.9

Batches
                                      design  first  n_first  second  n_second  third  n_third
            Published classifier, all inputs   23.5       38    14.5        19    5.5        7
                 No-optical input everywhere   25.5       38    16.0        19    6.0        7
                  AIRS-only input everywhere   29.5       38    16.0        19    5.0        7
               Margin rule → AIRS-only input   26.0       38    15.5        19    5.0        7
             Ensemble rule → AIRS-only input   26.0       38    15.5        19    5.0        7
        k-NN distance rule → AIRS-only input   25.5       38    14.0        19    5.5        7
          Mahalanobis rule → AIRS-only input   26.0       38    15.5        19    5.5        7
                  PCA rule → AIRS-only input   26.0       38    15.5        19    5.0        7
                Shape flag → AIRS-only input   28.5       38    16.0        19    5.0        7
            Haze-retrained design everywhere   27.0       38    15.5        19    6.0        7
       Haze+deck-retrained design everywhere   28.0       38    16.5        19    5.5        7
                          The routing method   29.5       38    16.0        19    6.0        7
   Amplitude flag, AIRS-only input elsewhere   30.0       38    16.0        19    6.0        7
Muting census in place of the amplitude flag   30.5       38    16.5        19    6.5        7

