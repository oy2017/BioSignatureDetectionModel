# Methodology

## Data Generation
*   **Total Spectra Generated:** A training set of **3,000** unique planetary spectra and a test set of **600** spectra were generated for each experiment. For final variance analysis, 5 independent test sets of 600 spectra each were created.
*   **Parameter Ranges (H2-dominated atmospheres):**
    *   **Planet Radius:** 1.0 - 26.0 $R_{\oplus}$
    *   **Planet Mass:** 1.0 - 300.0 $M_{\oplus}$
    *   **Atmospheric Temperature:** 500 - 2500 K
    *   **Stellar Temperature:** 2500 - 7500 K
    *   **Wavelength Range:** 0.5 - 7.8 $\mu m$ (at R=200)
    *   **Signal-to-Noise Ratio (SNR):** 15
    *   **Biosignature Definition:** Log($CH_4$) > -6.0 AND Log($O_3$) > -7.0.

## Model Training & Evaluation
*   **Train/Test Split:** Models were trained on the **3,000-sample** `multirex_spectra_H2_train.parquet` dataset. Final performance was evaluated on a held-out **~600-sample** `multirex_spectra_H2_test.parquet` dataset. Data was shuffled before training to ensure random distribution.
*   **Cross-Validation:** During hyperparameter tuning, **Stratified 3-Fold Cross-Validation** was used on the training set to find the optimal parameters for Random Forest and XGBoost. Deep learning models used a simple 80/20 validation split.
*   **Evaluation Metric:** The primary metric for model performance and hyperparameter selection was **Accuracy**. For final model comparison, **Precision, Recall, and F1-Score** were also calculated across 5 independent test sets to measure variance. Additionally, pairwise **McNemar's Tests** were employed to evaluate the statistical significance of performance differences between models. To assess model reliability and the trustworthiness of confidence levels, **Calibration Curves (Reliability Diagrams)** and **Brier Scores** were utilized.

## Software & Hardware
*   **Software:**
    *   **Language:** Python 3
    *   **Data Generation:** `multirex` (v0.3.1)
    *   **Machine Learning:** `scikit-learn` (for RandomForest, PCA, StandardScaler), `XGBoost`, `TensorFlow` (with Keras API for MLP/CNN).
    *   **Data Handling:** `pandas`, `numpy`
*   **Hardware:**
    *   The experiments were run on a standard local machine without specialized GPU acceleration for the deep learning models. CPU processing was parallelized where possible (e.g., `n_jobs=-1` for XGBoost/RF).
