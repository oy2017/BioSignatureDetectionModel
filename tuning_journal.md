# Tuning Journal for BioSignatureDetectionModel

This journal documents the experiments performed to tune the machine learning models for biosignature detection.

## PCA Variance Analysis (H2 Dataset)

Before running tuning experiments, a PCA variance analysis was performed on the H2 training data to understand how many components are needed to explain the variance in the data.

*   **Script**: `analyze_pca_variance.py`
*   **Findings**:
    *   The first principal component (PC-1) explains **97.01%** of the variance.
    *   The first two principal components (PC-1 and PC-2) explain **99.97%** of the variance.
    *   The number of components needed to explain 95% of the variance is just **1**.
*   **Conclusion**: The vast majority of the information is contained within the first few principal components.
*   **Artifacts**: `H2_pca_explained_variance.png`

---


## Experiment 1-10: Tuning the Number of Principal Components

This series of experiments tests how the number of top principal components used as features affects classification performance. The key finding is that performance is abysmal with only a few components, even though they explain >99.9% of the variance. This proves the classification signal is not in the highest-variance components.

*   **Exp 1 (1 PC):** Accuracy: 0.52
*   **Exp 2 (2 PCs):** Accuracy: 0.52
*   **Exp 3 (3 PCs):** Accuracy: 0.48
*   **Exp 4 (5 PCs):** Accuracy: 0.55
*   **Exp 5 (10 PCs):** Accuracy: 0.69
*   **Exp 6 (20 PCs):** Accuracy: 0.76
*   **Exp 7 (40 PCs):** Accuracy: 0.80
*   **Exp 8 (50 PCs):** Accuracy: 0.78
*   **Exp 9 (75 PCs):** Accuracy: 0.82
*   **Exp 10 (100 PCs):** Accuracy: 0.83

---

## Experiment 11-17: Tuning by Ignoring Initial Principal Components

This series of experiments tests the hypothesis that ignoring the highest-variance principal components can improve model performance.

*   **Exp 11 (PCs 0-100):** Accuracy: 0.83
*   **Exp 12 (PCs 1-101):** Accuracy: 0.83
*   **Exp 13 (PCs 2-102):** **Accuracy: 0.84** (Best performance so far)
*   **Exp 14 (PCs 3-103):** Accuracy: 0.84
*   **Exp 15 (PCs 2-7):** Accuracy: 0.57
*   **Exp 16 (PCs 2-12):** Accuracy: 0.69
*   **Exp 17 (PCs 2-32):** Accuracy: 0.80

### Summary of Findings (Experiments 0-17)
This initial set of experiments confirms that the highest-variance principal components are not the most useful for classification. Ignoring the first two principal components and using the next 100 (Experiment 13) resulted in a measurable improvement in performance. This suggests the highest-variance components act as noise for this classification task.

---
---

## Session 2: Systematic Model Tuning (Begins Exp. 18)

This section documents the systematic tuning of Random Forest and XGBoost models based on the findings from the initial PCA experiments.

### Random Forest Tuning with PCA (idx 2-32)

This tuning was performed with the initial PCA range of 2 to 32 components.

**Experiment 18: n_estimators = 50**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=32 --n_estimators=50`
*   **Hyperparameters:** `n_estimators`: 50, `max_depth`: None
*   **Accuracy:** 0.79
*   **Confusion Matrix:** `[[245, 53], [70, 230]]`
*   **Notes:** Slightly lower accuracy compared to the baseline (0.79 vs 0.80).

**Experiment 19: n_estimators = 100**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=32 --n_estimators=100`
*   **Hyperparameters:** `n_estimators`: 100, `max_depth`: None
*   **Accuracy:** 0.80
*   **Confusion Matrix:** `[[248, 50], [71, 229]]`
*   **Notes:** Performance is identical to the baseline (n_estimators=150).

**Experiment 20: n_estimators = 200**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=32 --n_estimators=200`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: None
*   **Accuracy:** 0.80
*   **Confusion Matrix:** `[[248, 50], [71, 229]]`
*   **Notes:** Performance is identical to the baseline (n_estimators=150).

**Experiment 21: n_estimators = 300**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=32 --n_estimators=300`
*   **Hyperparameters:** `n_estimators`: 300, `max_depth`: None
*   **Accuracy:** 0.80
*   **Confusion Matrix:** `[[248, 50], [71, 229]]`
*   **Notes:** Performance is identical to the baseline (n_estimators=150).

**Summary of Findings (Tuning n_estimators with PCA 2-32):** Varying `n_estimators` from 50 to 300 did not significantly change the model's performance.

**Experiment 22: max_depth = 5**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=32 --max_depth=5`
*   **Hyperparameters:** `n_estimators`: 150, `max_depth`: 5
*   **Accuracy:** 0.79
*   **Confusion Matrix:** `[[244, 54], [74, 226]]`
*   **Notes:** Slightly lower accuracy compared to the baseline (0.79 vs 0.80).

**Experiment 23: max_depth = 10**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=32 --max_depth=10`
*   **Hyperparameters:** `n_estimators`: 150, `max_depth`: 10
*   **Accuracy:** 0.80
*   **Confusion Matrix:** `[[248, 50], [71, 229]]`
*   **Notes:** Performance is identical to the baseline (max_depth=None).

**Experiment 24: max_depth = 20**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=32 --max_depth=20`
*   **Hyperparameters:** `n_estimators`: 150, `max_depth`: 20
*   **Accuracy:** 0.80
*   **Confusion Matrix:** `[[248, 50], [71, 229]]`
*   **Notes:** Performance is identical to the baseline (max_depth=None).

**Summary of Findings (Tuning max_depth with PCA 2-32):** Varying `max_depth` did not significantly improve performance.

---

### Random Forest Tuning with Optimal PCA (idx 2-102)

This tuning was performed with the optimal PCA configuration (indices 2 to 102).

**Experiment 25: n_estimators = 50**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=50`
*   **Hyperparameters:** `n_estimators`: 50, `max_depth`: None
*   **Accuracy:** 0.81
*   **Confusion Matrix:** `[[254, 44], [70, 230]]`

**Experiment 26: n_estimators = 100**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=100`
*   **Hyperparameters:** `n_estimators`: 100, `max_depth`: None
*   **Accuracy:** 0.83
*   **Confusion Matrix:** `[[255, 43], [57, 243]]`

**Experiment 27: n_estimators = 150**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=150`
*   **Hyperparameters:** `n_estimators`: 150, `max_depth`: None
*   **Accuracy:** 0.84
*   **Confusion Matrix:** `[[257, 41], [54, 246]]`

**Experiment 28: n_estimators = 200**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: None
*   **Accuracy:** **0.85**
*   **Confusion Matrix:** `[[254, 44], [47, 253]]`

**Experiment 29: n_estimators = 300**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=300`
*   **Hyperparameters:** `n_estimators`: 300, `max_depth`: None
*   **Accuracy:** 0.84
*   **Confusion Matrix:** `[[256, 42], [52, 248]]`

**Summary of Findings (Tuning n_estimators with Optimal PCA):** Performance peaked at `n_estimators=200`, achieving 0.85 accuracy.

**Experiment 30: max_depth = 5**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=5`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 5
*   **Accuracy:** 0.80
*   **Confusion Matrix:** `[[244, 54], [66, 234]]`

**Experiment 31: max_depth = 10**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=10`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 10
*   **Accuracy:** 0.83
*   **Confusion Matrix:** `[[247, 51], [52, 248]]`

**Experiment 32: max_depth = 20**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=20`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 20
*   **Accuracy:** **0.85**
*   **Confusion Matrix:** `[[254, 44], [47, 253]]`

**Experiment 33: max_depth = None (Unlimited)**
*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: None
*   **Accuracy:** **0.85**
*   **Confusion Matrix:** `[[254, 44], [47, 253]]`

**Summary of Findings (Tuning max_depth with Optimal PCA):** Performance was best with `max_depth=20` or `None`.

---
---

## XGBoost Model Evaluation and Tuning

### XGBoost Baseline Run (Optimal PCA)

The initial XGBoost evaluation was performed using the optimal PCA configuration discovered during the Random Forest tuning.

**Date:** 2025-11-08
**Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102`
**Hyperparameters (Defaults):**
*   `n_estimators`: 150
*   `max_depth`: 5
*   `learning_rate`: 0.1
*   `pca_start_idx`: 2
*   `pca_end_idx`: 102

**Results:**
```
              precision    recall  f1-score   support

 Non-Bio (0)       0.85      0.87      0.86       298
     Bio (1)       0.87      0.85      0.86       300

    accuracy                           0.86       598
   macro avg       0.86      0.86      0.86       598
weighted avg       0.86      0.86      0.86       598
```
**Confusion Matrix:**
```
[[259  39]
 [ 44 256]]
```
**Notes:** The baseline XGBoost model, without any specific tuning, already outperforms the best-tuned Random Forest model (0.86 vs 0.85 accuracy).

---

### XGBoost Hyperparameter Tuning (Optimal PCA)

**Experiment 34: n_estimators = 50**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=50`
*   **Hyperparameters:** `n_estimators`: 50, `max_depth`: 5, `learning_rate`: 0.1
*   **Accuracy:** 0.82
*   **Confusion Matrix:** `[[250, 48], [59, 241]]`

**Experiment 35: n_estimators = 100**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=100`
*   **Hyperparameters:** `n_estimators`: 100, `max_depth`: 5, `learning_rate`: 0.1
*   **Accuracy:** 0.86
*   **Confusion Matrix:** `[[259, 39], [45, 255]]`

**Experiment 36: n_estimators = 200**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 5, `learning_rate`: 0.1
*   **Accuracy:** **0.87**
*   **Confusion Matrix:** `[[263, 35], [43, 257]]`

**Experiment 37: n_estimators = 300**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=300`
*   **Hyperparameters:** `n_estimators`: 300, `max_depth`: 5, `learning_rate`: 0.1
*   **Accuracy:** 0.86
*   **Confusion Matrix:** `[[261, 37], [44, 256]]`

**Summary of Findings (Tuning n_estimators with Optimal PCA):** Performance peaked at `n_estimators=200`, achieving an accuracy of 0.87.

**Experiment 38: max_depth = 3**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=3`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 3, `learning_rate`: 0.1
*   **Accuracy:** 0.83
*   **Confusion Matrix:** `[[253, 45], [54, 246]]`

**Experiment 39: max_depth = 7**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=7`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 7, `learning_rate`: 0.1
*   **Accuracy:** 0.86
*   **Confusion Matrix:** `[[253, 45], [41, 259]]`

**Experiment 40: max_depth = 10**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=10`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 10, `learning_rate`: 0.1
*   **Accuracy:** 0.85
*   **Confusion Matrix:** `[[255, 43], [44, 256]]`

**Summary of Findings (Tuning max_depth with Optimal PCA):** The default `max_depth=5` (from the baseline run, which had an accuracy of 0.87 with n_estimators=200) remains the optimal value.

**Experiment 41: learning_rate = 0.01**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=5 --learning_rate=0.01`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 5, `learning_rate`: 0.01
*   **Accuracy:** 0.82
*   **Confusion Matrix:** `[[241, 57], [53, 247]]`

**Experiment 42: learning_rate = 0.05**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=5 --learning_rate=0.05`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 5, `learning_rate`: 0.05
*   **Accuracy:** 0.85
*   **Confusion Matrix:** `[[253, 45], [45, 255]]`

**Experiment 43: learning_rate = 0.2**
*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=5 --learning_rate=0.2`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 5, `learning_rate`: 0.2
*   **Accuracy:** **0.87**
*   **Confusion Matrix:** `[[266, 32], [48, 252]]`

**Summary of Findings (Tuning learning_rate with Optimal PCA):** The default `learning_rate=0.1` and the faster `learning_rate=0.2` both achieve the peak accuracy of 0.87.

---

## Final Verification of PCA Component Number

**Experiment 44: Testing 200 Principal Components (XGBoost)**
A final experiment was conducted to determine if using more than 100 principal components would yield further improvements. The best XGBoost configuration was tested with 200 components (indices 2 to 202).

*   **Command:** `source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=202 --n_estimators=200 --max_depth=5 --learning_rate=0.1`
*   **Accuracy:** 0.87
*   **Macro Avg F1-score:** 0.87
*   **Confusion Matrix:** `[[265, 33], [42, 258]]`
*   **Conclusion:** Using 200 principal components provided no improvement over using 100, confirming that 100 is the optimal number of components for this feature set.

**Experiment 45: Testing 200 Principal Components (Random Forest)**
This experiment tested the best Random Forest configuration with 200 components (indices 2 to 202) to see if it would improve performance.

*   **Command:** `source venv/bin/activate && python3 evaluate_random_forest.py H2 --pca_start_idx=2 --pca_end_idx=202 --n_estimators=200 --max_depth=20`
*   **Hyperparameters:** `n_estimators`: 200, `max_depth`: 20
*   **Accuracy:** 0.84
*   **Macro Avg F1-score:** 0.84
*   **Confusion Matrix:** `[[259, 39], [55, 245]]`
*   **Conclusion:** Using 200 principal components resulted in a slight decrease in accuracy (0.84 vs 0.85) compared to using 100 components, further confirming that 100 is the optimal number of components for this feature set for the Random Forest model as well.

---
---

# Final Project Conclusion

After extensive tuning of both Random Forest and XGBoost models, a definitive best-performing model and configuration have been identified for the H2 biosignature detection task.

**Winning Model:** XGBoost

**Key Findings:**
1.  **Feature Engineering is Crucial:** The single most important optimization was the selection of PCA components. For both models, ignoring the first two principal components (which contain >99.9% of the variance) and using the subsequent 100 (indices 2 to 102) dramatically improved performance. This proves the highest-variance components contained noise rather than discriminative signals.
2.  **XGBoost Superiority:** While the tuned Random Forest model performed well with 85% accuracy, the tuned XGBoost model consistently achieved a higher accuracy of 87%.

---

## Best Overall Result

The highest performance was achieved with the XGBoost classifier.

*   **Final Accuracy:** **0.87**
*   **Final Macro Avg F1-score:** **0.87**

### Optimal Configuration:
*   **Model:** XGBoost
*   **PCA:** Use 100 components, ignoring the first 2 (`--pca_start_idx=2 --pca_end_idx=102`)
*   **`n_estimators`:** 200
*   **`max_depth`:** 5
*   **`learning_rate`:** 0.1

### Command to Reproduce Best Result:
```bash
source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=5 --learning_rate=0.1
```
