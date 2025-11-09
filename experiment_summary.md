# Summary of Model Tuning Experiments for H2 Biosignature Detection

## 1. Primary Finding: The Signal is Not in the Noise

The most significant and counter-intuitive discovery from this series of experiments is that the principal components (PCs) that explain the vast majority of the data's variance are actively harmful to classification performance.

- **The first two PCs**, which account for over **99.9% of the explained variance**, result in a model that performs no better than a random guess (52% accuracy).
- This indicates that the largest sources of variation in the spectral data—likely related to global parameters like stellar type or planet size—are not correlated with the presence of a biosignature. They are effectively **noise** for this classification task.

The key to building a successful model was to **ignore these high-variance components** and focus on the subsequent, lower-variance components which contain the subtle, discriminative **signal**.

---

## 2. Feature Engineering: Optimizing the PCA Feature Set

Based on the primary finding, a series of experiments were conducted to identify the optimal set of principal components.

### Key Conclusions:
1.  **Removing the "Noise" is Critical:** Explicitly removing the first two principal components consistently improved model performance. For example, a Random Forest model using PCs 0-100 achieved 83% accuracy, while the same model using PCs 2-102 achieved 84% accuracy. This demonstrates that the first two PCs actively confuse the model even when a large number of signal-bearing components are present.
2.  **More Signal is Better (to a point):** Performance scaled positively with the number of components used *after* the initial two. The optimal number was found to be **100**.
3.  **Diminishing Returns:** Extending the number of components from 100 to 200 provided no performance benefit for the best model (XGBoost) and slightly degraded performance for the Random Forest model.

**Optimal Feature Set:** The best and most consistent results were achieved using the principal components from **index 2 to 102**.

---

## 3. Model Comparison and Hyperparameter Tuning

With the optimal feature set identified, Random Forest and XGBoost models were tuned for comparison.

### Random Forest
- Using the optimal PCA set (2-102), the Random Forest model's accuracy jumped to **84%**.
- After tuning `n_estimators` (optimal: 200) and `max_depth` (optimal: 20 or None), the peak performance reached **85% accuracy**.

### XGBoost
- The XGBoost model immediately demonstrated superior performance, achieving **86% accuracy** on the optimal PCA set with its default hyperparameters.
- After tuning `n_estimators` (optimal: 200), `max_depth` (optimal: 5), and `learning_rate` (optimal: 0.1), the peak performance reached **87% accuracy**.

---

## 4. Final Recommendation

**The best-performing model is XGBoost, but only when applied to a carefully engineered feature set.**

The critical takeaway for any future work on this dataset is the feature engineering strategy: the removal of the initial high-variance principal components is more important than the choice of model or its specific hyperparameters.

### Best Overall Result:
- **Model:** XGBoost
- **Accuracy:** **0.87**
- **Macro Avg F1-score:** **0.87**

### Command to Reproduce Best Result:
This command encapsulates the optimal feature selection and hyperparameter configuration.
```bash
source venv/bin/activate && python3 evaluate_xgboost.py H2 --pca_start_idx=2 --pca_end_idx=102 --n_estimators=200 --max_depth=5 --learning_rate=0.1
```