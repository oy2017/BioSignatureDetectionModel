# DRAFT: Methods & Results for Biosignature Detection Paper

## 2. Methods

### 2.1. Data Generation & Simulation

A synthetic dataset of atmospheric transmission spectra was generated using the `multirex` (v0.3.1) Python library. The primary training dataset consisted of 3,000 unique planetary spectra, which was reduced to **2,709** samples after a critical data validation step. This cleaning process removed any simulated spectra containing physically impossible transit depths (> 1.0) or NaN values, which were found to cause numerical instability in downstream models. For final model validation, five independent test sets, each containing ~550 clean spectra, were generated using the same methodology.

The simulations focused on H2-dominated atmospheres, with physical parameters sampled uniformly from the following ranges:
*   **Planet Radius:** 1.0 – 26.0 $R_{\oplus}$
*   **Planet Mass:** 1.0 – 300.0 $M_{\oplus}$
*   **Atmospheric Temperature:** 500 – 2500 K
*   **Stellar Temperature:** 2500 – 7500 K

Spectra were generated across a wavelength range of **0.5 to 7.8 $\mu m$** at a spectral resolution of R=200, with a constant signal-to-noise ratio (SNR) of 15 applied to all observations. A planet was defined as possessing a biosignature if its atmospheric log mixing ratios satisfied both **Log(CH4) $\ge$ -6.0** and **Log(O3) $\ge$ -7.0**.

### 2.2. Feature Engineering

The raw spectral data, consisting of 200 wavelength channels, was pre-processed using `scikit-learn`'s `StandardScaler` to normalize the feature space. Following scaling, Principal Component Analysis (PCA) was applied for dimensionality reduction and feature extraction. It was empirically determined that models trained on PCA-transformed data significantly outperformed those trained on raw spectra. The optimal feature set for the Random Forest, XGBoost, and CNN models was found to be **Principal Components 2 through 102**. This strategy intentionally omits the first two components, which were found to primarily encode variance related to planetary radius and the spectral continuum rather than informative chemical absorption features. For the MLP model, components 0-100 were used, as it showed a slightly better ability to utilize the additional information.

### 2.3. Model Training & Hyperparameter Optimization

Four machine learning architectures were trained and evaluated: Random Forest, XGBoost, a Multi-Layer Perceptron (MLP), and a 1D Convolutional Neural Network (CNN). Hyperparameters for the tree-based models were optimized using an exhaustive grid search with stratified 3-fold cross-validation, using accuracy as the scoring metric. The best-performing hyperparameters were:
*   **XGBoost:** `n_estimators=300`, `max_depth=7`, `learning_rate=0.05`, `subsample=0.8`.
*   **Random Forest:** `n_estimators=300`, `max_depth=None`, `min_samples_leaf=2`, `min_samples_split=5`.

The MLP and CNN, built with the TensorFlow Keras API, were also tuned via grid search, leading to an optimized architecture of 2 convolutional layers for the CNN and 3 dense layers for the MLP, both utilizing `Adam` optimization, `BatchNormalization`, and `Dropout`.

### 2.4. Model Validation

Final model performance and robustness were assessed by training each optimized model on the full 2,709-sample training set and evaluating it against the five independent test sets. The mean and standard deviation of four key metrics—Accuracy, Precision, Recall, and F1-Score—were calculated to establish statistically robust performance benchmarks and confidence intervals. All experiments were conducted in Python 3, leveraging `scikit-learn`, `XGBoost`, `TensorFlow`, `pandas`, and `numpy` on standard multi-core CPU hardware.

---

## 3. Results & Discussion

### 3.1. Overall Performance

The comprehensive evaluation revealed a clear performance hierarchy among the models. The **XGBoost classifier emerged as the top-performing model**, achieving a mean accuracy of **88.67% ($\pm$ 1.52%)** across the five independent test sets. It was closely followed by the Random Forest model at **86.31% ($\pm$ 1.29%)**. The neural network models, while still highly effective, ranked third and fourth, with the MLP achieving **85.23% ($\pm$ 0.92%)** and the optimized CNN reaching **82.97% ($\pm$ 1.35%)**. These results underscore the efficacy of gradient-boosted trees for this classification task.

### 3.2. The Critical Role of PCA in Feature Extraction

A key finding of this work is the undeniable importance of PCA as a feature engineering step. While the best-performing CNN achieved ~83% accuracy on PCA-transformed data, its performance on raw spectral data was significantly lower, peaking at only 70.02% after extensive tuning. This performance gap highlights PCA's dual role in both reducing dimensionality and acting as a powerful denoising filter.

Physical interpretation of the principal components provides insight into this effect. Analysis of the PCA loadings reveals that the first component (PC1) is primarily a flat continuum, corresponding to the mean transit depth and thus planetary radius. The second component (PC2), however, shows strong weights at wavelengths corresponding to known absorption features of methane (~3.3 $\mu m$) and ozone (~4.7 $\mu m$). By training on components 2-102, we effectively force the models to learn from the chemical "fingerprints" of the atmosphere, rather than the scientifically uninteresting variance of the planet's size, leading to more robust and physically meaningful classifications.

### 3.3. Model-Specific Behavior and Application Trade-offs

While XGBoost demonstrated the highest overall accuracy, the MLP model exhibited a uniquely valuable characteristic: a significantly higher mean recall of **94.56% ($\pm$ 1.46%)**. In the context of searching for biosignatures, where the cost of a false negative (missing a potentially life-bearing planet) is exceptionally high, this makes the MLP a compelling alternative. Its tendency to produce more false positives (lower precision) is an acceptable trade-off for its superior ability to correctly identify nearly all true positive cases. This suggests that for a large-scale survey mission, an MLP-based model could be the optimal choice for an initial, highly sensitive "first-pass" candidate selection, with higher-precision models like XGBoost used for subsequent vetting.
