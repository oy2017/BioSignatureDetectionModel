# CNN Debugging & Optimization Journal
**Date:** February 13, 2026
**Focus:** Reproducing and improving CNN performance for Biosignature Detection.

## 1. Initial Status & Data Integrity
**Goal:** Verify the "Best CNN" configuration (Experiment E10, ~83% accuracy) from previous tuning logs.

### A. The Crash
*   **Incident:** Attempting to run `evaluate_deep_learning.py` resulted in immediate crashes or `NaN` errors during PCA/Scaling.
*   **Diagnosis:** The dataset contained physical impossibilities.
    *   **Observation:** ~64 rows in the training set had transit depths > 1.0 (some as high as `1e209`), causing numerical overflow.
    *   **Root Cause:** Instability in the `multirex` radiative transfer simulation for certain planetary parameters.
*   **Fix:** 
    1.  Updated `generate_multiverse_data_1-15.py` to filter out `NaN`s and values > 1.0 during generation.
    2.  Regenerated the `H2` dataset to ensure a clean baseline.

---

## 2. Reproducibility Crisis (PCA + CNN)
**Goal:** Reproduce the ~83% accuracy of the manual E10 configuration on the new clean dataset.

### A. The Manual Run
*   **Command:** `evaluate_deep_learning.py H2 --pca 2-102 --epochs 100 --batch 64`
*   **Result:** **77.65% Accuracy**.
*   **Analysis:** Slightly lower than the historical 83%, likely due to the new random seed/dataset generation, but a solid baseline.

### B. The Grid Search Failure
*   **Action:** Created `tune_cnn_gridsearch.py` to systematically optimize hyperparameters.
*   **Result:** The "Best" model found by grid search (using E10 parameters) achieved only **54.19%**.
*   **Discrepancy:** Why did the manual script get 77% and the grid search get 54% with the *same* parameters?
*   **Investigation:** 
    *   Architecture was identical.
    *   Hyperparameters were identical.
    *   **The Culprit:** **Data Shuffling**.
    *   The manual script (`evaluate_deep_learning.py`) shuffled the training data (`sklearn.utils.shuffle`) before splitting.
    *   The grid search script relied on Keras's internal validation split, which takes the *last* 20% of data. If data is ordered (e.g., by planet mass), this creates a biased validation set.

### C. The Fix & Breakthrough
*   **Fix:** Added explicit `shuffle(X_train, y_train, random_state=42)` to the grid search script.
*   **Result:** The grid search immediately matched the manual performance and found an even better configuration.
*   **Best CNN Configuration:**
    *   **Accuracy:** **84.92%** (vs 77.65% manual baseline)
    *   **Params:** `filters=32`, `kernel=3`, `dropout=0.3`, `batch=32`, `lr=0.0005`.
    *   **Insight:** A smaller, slower-learning model performed better than the original E10 config.

---

## 3. The Raw Spectra Challenge
**Goal:** Make the CNN learn directly from Raw Spectra (200 channels) without PCA.

### A. Baseline
*   **Performance:** **~50%** (Random Guessing).
*   **Hypothesis:** The raw data is too noisy and high-dimensional for the shallow network.

### B. Grid Search Optimization
*   **Action:** Ran the corrected `tune_cnn_gridsearch.py` on Raw Spectra (100 epochs).
*   **Result:** **70.02% Accuracy**.
*   **Significance:** This was a massive jump from 50%. It proved that with proper shuffling and sufficient epochs, the CNN *can* learn from raw data.
*   **Best Raw Params:** `filters=32`, `kernel=5`, `dropout=0.3`, `batch=64`.

### C. Advanced Optimization Attempts (Failed)
*   **Attempt 1 (Deeper Network):** Added a 3rd Convolutional layer.
    *   **Result:** **54.56%**. The model collapsed.
*   **Attempt 2 (Data Augmentation):** Added `GaussianNoise` layer.
    *   **Result:** **54%**. The added noise likely overwhelmed the weak signal.
*   **Attempt 3 (Binning):** Averaged every 4 spectral channels (200 -> 50 features).
    *   **Result:** **51.21%**. Loss of resolution hurt more than the denoising helped.

---

## 4. Final Verdict

| Approach | Model | Features | Best Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **PCA** | **CNN** | PCA (2-102) | **84.92%** | **Strongest Result** |
| **Raw** | **CNN** | Raw Spectra | **70.02%** | Viable, but inferior |
| **Raw** | **CNN** | Binned/Deep | ~51-54% | Failed |

### Key Takeaways
1.  **PCA is Essential:** It acts as a critical feature extractor that no amount of CNN tuning (so far) can match on raw data.
2.  **Shuffling is Critical:** Omitting data shuffling caused a 20%+ drop in perceived performance.
3.  **Simplicity Wins:** Smaller filters (32 vs 64) and shallower networks (2 layers vs 3) consistently performed better, suggesting the dataset size (~2700 samples) limits the complexity of models we can train.
