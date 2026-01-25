# Grid Search & Early Stopping Analysis
**Date:** January 2, 2026

## 1. Executive Summary
This experiment aimed to optimize the 3-Layer MLP architecture (`256 -> 128 -> 64`) using `GridSearchCV`. The key finding is that **dataset size** played a more critical role than hyperparameter tuning. The original script's strategy of using a single large training split (80%) with Early Stopping consistently outperformed Cross-Validation methods which fragmented the small dataset (~3,000 samples) too aggressively.

## 2. Performance Comparison

| Method | Test Accuracy | Training Samples | Validation Strategy | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Original Script** (`evaluate_mlp_3layer_best.py`) | **87.5%** | **2,400** | Single Split (20%) + Early Stopping | **Best Performer.** Maximized data quantity for training allowed the model to learn robust features. |
| **Simple Grid Search** (`tune_3layer_mlp_simple_grid.py`) | **85.6%** | **2,400** | Single Split (20%) + Early Stopping | **Robust.** Replicated the original data strategy but systematically tuned parameters. Confirmed the architecture is stable. |
| **Standard Grid Search** (`tune_3layer_mlp_gridsearch_cv.py`) | 82.9% | 2,000 | 3-Fold CV (No Early Stopping) | **Good.** Lower accuracy due to reduced training data size (2,000 vs 2,400) and lack of Early Stopping (forced to 100 epochs). |
| **Nested Grid Search** (`tune_3layer_mlp_gridsearch_earlystopping.py`) | 54.5% | 1,600 | 3-Fold CV + Internal Split (20%) | **Failed.** Data starvation. Splitting the 2,000 fold samples further for internal validation left only 1,600 samples, causing model collapse. |

## 3. The Impact of Epochs & Early Stopping
*   **Original / Simple Grid:** Used `EarlyStopping`. The model typically converged around **Epoch 53** (restoring best weights). This prevented overfitting.
*   **Standard Grid Search:** Forced `100 Epochs`. While it found good parameters, the lack of a safety brake meant the model likely overfit the training noise in the later epochs, reducing test accuracy.

## 4. Best Hyperparameters Found
The **Simple Grid Search** (which successfully combined high data volume with tuning) identified the following optimal configuration:

*   **Architecture:** `[256, 128, 64]` (Fixed)
*   **Batch Size:** `128`
*   **Learning Rate:** `0.001`
*   **Dropout Rate:** `0.3` (Slightly lower than the original 0.4, suggesting the model could handle slightly more complexity).
*   **Epochs:** `100` (Upper limit; Early Stopping typically cut this short around epoch 50-60).

## 5. Conclusion
For this specific dataset size (~3,000 samples), **Single-Split Validation is superior to K-Fold Cross Validation**. The loss of training data inherent in K-Fold (and especially nested validation) outweighs the statistical robustness it usually provides. Future improvements should focus on **Data Augmentation** to increase the effective sample size, which would then allow for robust Cross-Validation.
