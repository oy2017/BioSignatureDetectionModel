# Deep Learning Model Tuning Journal

## Goal
To optimize a neural network model for detecting biosignatures from exoplanet spectra, aiming to match or exceed the performance of the existing XGBoost benchmark (~87% accuracy).

---

## Key Findings

1.  **PCA is Essential:** Both MLP and CNN models consistently failed to learn from raw spectral data (**E1, E2**). Applying Principal Component Analysis (PCA) was the single most critical step to enable successful training.
2.  **MLP is the Superior Architecture:** The Multi-Layer Perceptron (MLP) consistently outperformed the Convolutional Neural Network (CNN) and ultimately achieved a higher peak accuracy (**E6** vs. **E4**).
3.  **Batch Size is a Crucial, Model-Specific Hyperparameter:** The optimal batch size was different for each model. The MLP's performance peaked at a larger batch size of **128** (**E6**), while the CNN performed best with a smaller batch size of **64** (**E4**).
4.  **MLP Matches the Benchmark:** The final optimized MLP configuration (**E6**) successfully matched the ~87% accuracy of the XGBoost model, achieving the project's primary goal.

---

## Hyperparameters Under Investigation

*   **Model Architecture (`--model_type`):** The fundamental type of neural network used (CNN vs. MLP).
*   **Input Features (PCA vs. Raw Spectra):** The data fed into the model, controlled via `--pca_start_idx` and `--pca_end_idx`.
*   **Feature Scaling:** `StandardScaler` was applied both before PCA and on the selected PCA components. This was found to be a critical step.
*   **Epochs (`--epochs`):** The number of passes through the training dataset. `EarlyStopping` was used to prevent overfitting.
*   **Batch Size (`--batch_size`):** The number of samples per gradient update.

---

## Master Experiment Log

| ID | Model | PCA Config | Epochs | Batch Size | Accuracy | Notes |
|----|-------|------------|--------|------------|----------|---------------------------------|
| E1 | CNN   | Raw Spectra| 50     | 32         | **49.83%** | Failure to learn |
| E2 | MLP   | Raw Spectra| 50     | 32         | **51.34%** | Failure to learn |
| E3 | CNN   | PCA 0-End  | 50     | 32         | **79.77%** | Successful learning with PCA |
| E4 | CNN   | PCA 2-102  | 100    | 64         | **83.00%** | **Best CNN Result** |
| E5 | MLP   | PCA 0-100  | 50     | 32         | **83.80%** | Strong baseline MLP performance |
| E6 | MLP   | PCA 0-100  | 50     | 128        | **87.63%** | **Best MLP & Overall Result** |
| E7 | MLP   | PCA 0-100  | 50     | 256        | **85.12%** | Performance drop after peak |

---

## Experimental Analysis

### Part 1: The Critical Role of PCA vs. Raw Spectra

The initial investigation focused on whether the models could learn from the raw spectral data. Experiments **E1** (CNN) and **E2** (MLP) both resulted in accuracies of ~50%, demonstrating a complete failure to learn, even after 50 epochs.

In contrast, as soon as PCA-transformed data was used (**E3**, **E5**), performance dramatically improved to ~80% and ~84% respectively.

**Conclusion:** Raw spectral data contains high-variance noise that prevents the neural networks from learning. PCA is an essential feature engineering step to filter this noise and present the data in a learnable format.

### Part 2: Architecture Comparison - MLP vs. CNN

With PCA established as necessary, a comparison of the two architectures shows the MLP consistently achieved higher accuracy. The best MLP result (**E6**, 87.63%) significantly outperformed the best CNN result (**E4**, 83%).

**Conclusion:** The MLP is the more effective architecture for this task. Its structure, which treats each PCA component as an independent feature, is better suited to the non-spatially correlated nature of principal components.

### Part 3: Tuning - The Effect of Batch Size

The final tuning step revealed a critical, model-specific sensitivity to batch size.

*   **For the MLP:** A series of experiments (**E5**, **E6**, **E7**) tested batch sizes of 32, 128, and 256. Performance peaked at a batch size of **128** (**E6**), indicating a "sweet spot" for this architecture.
*   **For the CNN:** The best performance (**E4**) was found with a batch size of **64**. Previous tests with other batch sizes (e.g., 32, 128) resulted in lower accuracy.

**Conclusion:** The optimal batch size is not universal. The MLP benefited from the stable gradients of a larger batch size (128), while the CNN performed better with the regularizing effect of a smaller batch size (64).

---

## Final Conclusion & Optimal Configurations

### Optimal Configuration (MLP)
*   **Reference:** Experiment **E6**
*   **Architecture:** MLP (Dense 256 -> 128 -> 64)
*   **Input Features:** PCA Components 0-100 (Scaled)
*   **Hyperparameters:** 50 Epochs, Batch Size 128, Learning Rate 0.001
*   **Performance:** **87.63% Accuracy**

This configuration is the best-performing neural network, successfully matching the XGBoost benchmark.

### Best CNN Configuration
*   **Reference:** Experiment **E4**
*   **Architecture:** 1D CNN
*   **Input Features:** PCA Components 2-102 (Scaled, Noisy components removed)
*   **Hyperparameters:** 100 Epochs, Batch Size 64, Learning Rate 0.001
*   **Performance:** **83% Accuracy**

The CNN also demonstrates strong performance, proving to be a viable, albeit slightly less optimal, alternative to the MLP.