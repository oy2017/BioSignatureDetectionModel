# 6. Results and Discussion

## 6.1. Overall Performance and Model Hierarchy
The comprehensive evaluation across five independent, mission-scaled test sets revealed a performance hierarchy among the evaluated machine learning architectures. The results, summarized in **Table II**, indicate that gradient-boosted decision trees (XGBoost) and the Multi-Layer Perceptron (MLP) provided the most accurate predictions for biosignature detection, achieving near-parity in overall performance.

**Table II. Model Performance Summary (Mean Accuracy ± Standard Deviation over 5 Sets)**
| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 88.59% (± 1.66%) | 87.93% (± 1.53%) | 89.55% (± 1.80%) | **88.73% (± 1.63%)** |
| **Random Forest** | 86.69% (± 0.70%) | 87.03% (± 0.68%) | 86.33% (± 1.55%) | **86.67% (± 0.74%)** |
| **MLP** | 88.17% (± 0.96%) | 87.78% (± 1.54%) | 88.82% (± 1.82%) | **88.28% (± 0.94%)** |
| **1D-CNN** | 81.86% (± 0.86%) | 87.04% (± 1.96%) | 75.07% (± 1.42%) | **80.59% (± 0.90%)** |

The XGBoost classifier and the optimized MLP emerged as the top-performing models, both achieving approximately 88% accuracy. To ensure statistical robustness, we calculated the standard deviation across five independent test sets for each model, as reported in Table II. To rigorously evaluate the statistical significance of these results, pairwise McNemar's tests were conducted. These confirmed that while XGBoost and MLP reached near-parity, XGBoost achieved its performance through a more robust, baseline-independent learning mechanism (discussed in Section 6.2).

The confusion matrix for the best-performing XGBoost model is presented in **Figure 1**, illustrating a high true positive rate and a well-balanced precision-recall trade-off. This visualization reveals that errors are most frequent for planets with very low atmospheric temperatures or higher surface gravity, where atmospheric scale heights are compressed, leading to weaker absorption signals. 

## 6.2. The Critical Role of PCA and Baseline Independence
A central finding of this research is the undeniable importance of Principal Component Analysis (PCA) as both a dimensionality reduction and signal-isolation mechanism. The necessity of PCA was demonstrated empirically: while models trained on PCA-transformed data achieved peak accuracies of ~88%, the optimized 1D-CNN trained on raw spectral data peaked at only 50.56% (± 1.58%) accuracy across the 5 independent test sets. Confirming it failed to learn discriminative features, the raw CNN defaulted to a “Yes-skewed” guessing strategy against the high-dimensional noise floor.

The PCA decomposition revealed a stark dichotomy in how different architectures process high-dimensional data. To understand the physical meaning of the feature space, a Pearson correlation analysis revealed a near-perfect correlation ($r = 0.961$) between PC0 and planetary radius (dictating absolute baseline transit depth), and a strong correlation between PC1 and systemic temperatures. Combined, PC0 and PC1 explain 99.26% of the total spectral variance, capturing the broad physical continuum. Crucially, these first two components exhibited near-zero correlation ($r < 0.1$) with the target chemical abundances. 

The inclusion or exclusion of these primary physical components highlighted fundamentally different levels of model robustness. For the tree-based ensembles (XGBoost and Random Forest), McNemar's tests revealed that the removal of PC0 and PC1 yielded no statistically significant difference in performance ($p > 0.4$). Because decision tree algorithms inherently perform implicit feature selection at each node split (Breiman, 2001), these models successfully ignored the massive physical variance of the continuum and learned exclusively from the high-frequency chemical absorption features (PC2–101). We characterize this scale-invariant behavior as **baseline-independent**.

Conversely, the deep learning architectures (MLP and 1D-CNN) proved strictly **baseline-dependent**. When restricted to the chemical subset (PC2–101), their accuracy suffered a statistically significant collapse ($p < 0.0001$), plummeting to ~55%. This indicates that unlike the ensemble models, the neural networks require the macro-physical context of the continuum to mathematically orient their gradient descent before they can extract high-frequency chemical signatures. 

## 6.3. Error Analysis and Physical Dependencies
To understand model failure modes, a physics-based analysis was conducted by examining how classification errors varied with physical parameters. A summary of the parameter ranges explored is provided in **Table I**.

**Figure 4** presents a triangle plot of planet and stellar parameters, with data points colored based on correct versus incorrect test set predictions. This visualization reveals that errors are most frequent for planets with very low atmospheric temperatures or higher surface gravity, where atmospheric scale heights are compressed, leading to weaker absorption signals. 

Furthermore, **Figure 5** provides a similar triangle plot focusing on atmospheric parameters (mixing ratios of CH4 and O3). As expected, the model struggles near the predefined abundance thresholds. This highlights the inherent ambiguity in marginal biosignature detections, where the signal-to-noise ratio of specific absorption features is lowest.

## 6.4. Application Trade-offs for Survey Missions
When evaluated across the five independent, mission-scaled test sets, the optimized XGBoost and Multi-Layer Perceptron (MLP) models demonstrated near-parity in overall performance. However, their operational robustness diverges significantly. 

The XGBoost architecture requires only the highly compressed 100-component chemical feature space (PC2-101) to achieve maximal performance, demonstrating native immunity to macro-physical systematics. Conversely, the MLP achieved parity only when provided the full 102-component space, relying heavily on the absolute physical continuum to orient its gradient descent. In real-world observations, the absolute spectral baseline is notoriously difficult to constrain due to radius-temperature-pressure degeneracies and systematic biases introduced by unmodeled 3D atmospheric effects (Caldas et al., 2019). 

A model dependent on this baseline (like the MLP) is inherently more fragile than a model capable of evaluating relative chemical absorption features independently of the continuum (like XGBoost). Given the strict computational and latency budgets associated with prioritizing targets for Ariel's Tier 3 benchmark observations, the tree-based XGBoost model emerges as the optimal primary triage tool, offering equivalent detection capability with a fundamentally more robust, scale-invariant feature extraction pipeline.
