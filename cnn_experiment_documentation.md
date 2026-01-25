# CNN & ResNet Experiment Documentation
**Date:** January 24, 2026
**Project:** BioSignature Detection Model

## 1. Overview
Convolutional Neural Networks (1D-CNNs) and ResNets were explored as a way to learn features directly from the raw spectral data, bypassing PCA. **These experiments largely failed**, achieving accuracies between 50% (random guess) and 61%.

## 2. Key Findings

### A. The Failure of Raw Spectra
Unlike MLP and XGBoost, which used PCA-compressed features, these models consumed the raw 1D spectral grid (or binned versions).
*   **Result:** The models failed to extract meaningful patterns from the noise.
*   **Diagnosis:** The dataset size (~2,400 training samples) is insufficient for a Deep CNN to learn invariant filters across a high-dimensional input space (>5,000 spectral bins).

### B. Architecture Experiments

#### 1. Standard 1D CNNs (4-Layer & 8-Layer)
**Scripts:** `evaluate_set_layers_cnn.py`, `evaluate_8_layer_cnn.py`
*   **Performance:** **~50% Accuracy** (Random Chance).
*   **Behavior:** The models collapsed, often predicting "Non-Bio" for everything or oscillating wildly.

#### 2. 1D ResNet (Optimized)
**Script:** `evaluate_resnet_cnn_optimized.py`
*   **Performance:** **50% Accuracy**.
*   **Behavior:** Even with residual connections to improve gradient flow, the model could not find a descent path to a solution better than random guessing.

#### 3. Derivative CNN (Feature Engineering)
**Script:** `evaluate_derivative_cnn.py`
*   **Concept:** Instead of raw flux, feed the *1st Derivative* (slope) of the spectrum to highlight absorption lines.
*   **Performance:** **~61% Accuracy**.
*   **Observation:** This was the *only* CNN approach that showed "learning" behavior. It suggests that the *change* in flux is more informative than the absolute flux, but it still falls far short of PCA-based models (~87%).

## 3. Conclusion
Deep Learning from raw spectra is **not viable** with the current dataset size.
*   **Recommendation:** Future CNN attempts would require massive Data Augmentation (e.g., adding noise, shifting spectra) to artificially increase the training set size by 10-100x.
*   **Current Status:** All CNN development is paused in favor of PCA + XGBoost/MLP.
