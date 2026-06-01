# DRAFT: Methods & Results for Biosignature Detection Paper

## 5. Methodology/Models

This study aims to design a machine learning model capable of distinguishing between biosignature and non-biosignature environments using high-resolution (HR) transmission spectra from an Ariel-like dataset. We assess a suite of machine learning models on mock data simulated via the `multirex` library, using the following methodologies.

### 5.1. Data Generation & Simulation

#### 5.1.1. Radiative Transfer Modeling
The synthetic dataset of atmospheric transmission spectra was generated using the `multirex` (v0.3.1) Python library, which leverages the TauREx 3 radiative transfer code [1]. TauREx 3 computes transmission spectra by simulating the transit of starlight through a planetary atmosphere, robustly modeling molecular absorption, Rayleigh scattering, and collision-induced absorption across atmospheric layers to produce physically realistic data. This framework allows for the rapid generation of physically rigorous spectra.

#### 5.1.2. Simulation Parameters
We generated a training set of 2,709 simulated planetary spectra (reduced from 3,000 after removing spectra with physically impossible transit depths > 1.0 or NaN values), as well as five independent testing sets, each containing ~550 clean spectra. The simulations focused on hydrogen-dominated atmospheres, following [2-4]. Each sample contained a one-dimensional transmission spectrum spanning wavelengths 0.5 to 7.8 $\mu m$ at a spectral resolution of R=200, with a constant signal-to-noise ratio (SNR) of 15 applied to all observations.

Physical parameters were simultaneously varied along a pre-defined grid sampled uniformly from the following ranges:
*   **Planet Radius:** 1.0 – 26.0 $R_{\oplus}$
*   **Planet Mass:** 1.0 – 300.0 $M_{\oplus}$
*   **Atmospheric Temperature:** 500 – 2500 K
*   **Atmospheric Base Pressure:** $10^5$ – $10^6$ Pa
*   **Atmospheric Top Pressure:** 1 – 10 Pa
*   **Stellar Temperature:** 2500 – 7500 K
*   **Stellar Radius:** 0.1 – 1.7 $R_{\odot}$
*   **Stellar Mass:** 0.1 – 1.7 $M_{\odot}$
*   **Semi-Major Axis:** 0.01 – 0.5 AU

These parameter grid choices follow the procedures of the Ariel Data Challenge [5]. A planet was defined as possessing a biosignature if its atmospheric log mixing ratios satisfied both **Log(CH4) $\ge$ -6.0** and **Log(O3) $\ge$ -7.0**, following the thresholds outlined in [6].

### 5.2. Preprocessing and Feature Engineering

The raw spectral data, consisting of 200 wavelength channels, was pre-processed using `scikit-learn`'s `StandardScaler` to normalize the feature space. Following scaling, Principal Component Analysis (PCA) was applied for dimensionality reduction and feature extraction. PCA is a linear transformation technique that projects data into a new orthogonal coordinate system via the eigendecomposition of the data's covariance matrix, allowing us to isolate distinct physical and chemical signals into uncorrelated components.

It was empirically determined that models trained on PCA-transformed data significantly outperformed those trained on raw spectra. An analysis of variance revealed that the first two principal components (PC0 and PC1) explained over 90% of the total variance but showed little quantitative correlation (Pearson correlation coefficient $r < 0.1$) with the target gas abundances. By analyzing the component loadings and reconstructing the spectra, these components were found to represent the mean transit depth and overall spectral slope, thus encoding the overall planetary radius and stellar continuum level rather than specific chemical absorption features. 

As a result, PC0 and PC1 were intentionally omitted. The final feature set for the Random Forest, XGBoost, and CNN models was constructed using the next 100 principal components (indices 2–102), which cumulatively explain approximately 9.5% of the total variance. This range was chosen to preserve chemically relevant absorption features while minimizing the influence of large-scale astrophysical systematics. Figure 1 shows the scree plot of explained variance across the principal components, justifying the choice of 100 components. Additionally, Figure 2 illustrates the PCA reconstruction and residual error across wavelengths for a representative test spectrum, demonstrating the successful retention of absorption features. For the MLP model, components 0-100 were used, as it showed a slightly better ability to utilize the additional information.

### 5.3. Machine Learning Architectures and Hyperparameter Optimization

Four machine learning architectures were trained and evaluated: Random Forest [7], XGBoost [8], a Multi-Layer Perceptron (MLP) [9], and a 1D Convolutional Neural Network (CNN) [10]. These models were selected to cover a spectrum of complexity, from robust ensemble methods capable of handling non-linear decision boundaries to deep neural networks that can automatically learn hierarchical feature representations. 

Hyperparameters for the models were systematically optimized using an exhaustive grid search with stratified 3-fold cross-validation, utilizing accuracy as the primary scoring metric. 
*   **XGBoost:** The grid search encompassed learning rates $\in \{0.01, 0.05, 0.1, 0.2\}$, maximum depths $\in \{3, 5, 7, 10\}$, number of estimators $\in \{100, 200, 300\}$, and subsample ratios $\in \{0.8, 1.0\}$. The best-performing hyperparameters were `n_estimators=300`, `max_depth=7`, `learning_rate=0.05`, and `subsample=0.8`.
*   **Random Forest:** The search space included number of estimators $\in \{100, 200, 300\}$, maximum depth $\in \{\text{None}, 10, 20, 30\}$, minimum samples for an internal node split $\in \{2, 5, 10\}$, and minimum samples per leaf $\in \{1, 2, 4\}$. The optimal parameters were `n_estimators=300`, `max_depth=None`, `min_samples_leaf=2`, and `min_samples_split=5`.

The neural network architectures, built using the TensorFlow Keras API [11], were similarly tuned:
*   **MLP:** The grid evaluated hidden layer configurations $\in \{(256, 128), (256, 128, 64), (512, 256, 128, 64)\}$, dropout rates $\in \{0.2, 0.4\}$, batch sizes $\in \{32, 64\}$, and learning rates $\in \{0.001\}$ over 30 epochs. This led to an optimized architecture of 3 dense layers (256, 128, 64 units) utilizing `Adam` optimization, batch normalization, and dropout.
*   **CNN:** The search spanned filters $\in \{32, 64\}$, kernel sizes $\in \{3, 5\}$, dropout rates $\in \{0.3, 0.5\}$, learning rates $\in \{0.001, 0.0005\}$, and batch sizes $\in \{32, 64\}$. The final network consisted of 2 convolutional layers, also utilizing batch normalization and dropout to prevent overfitting.

### 5.4. Model Validation

Model selection and hyperparameter tuning were performed using stratified 3-fold cross-validation. Final model performance was then evaluated against the five independently generated test sets. Each model was evaluated on all five sets, and we report the mean accuracy and standard deviation in Section 6. The evaluation metrics used are defined as follows, where TP, TN, FP, and FN represent true positives, true negatives, false positives, and false negatives, respectively:

*   **Accuracy:** Overall correctness, calculated as $\frac{TP + TN}{TP + TN + FP + FN}$
*   **Precision:** Proportion of positive identifications that were actually correct, calculated as $\frac{TP}{TP + FP}$
*   **Recall:** Proportion of actual positives identified correctly, calculated as $\frac{TP}{TP + FN}$
*   **F1-Score:** Harmonic mean of Precision and Recall, calculated as $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

All experiments were conducted in Python 3, leveraging `scikit-learn` (v1.7.2), `XGBoost` (v3.1.2), `TensorFlow` (v2.20.0), `pandas` (v2.3.3), and `numpy` (v2.3.5) on standard multi-core CPU hardware.

---

## 6. Results and Discussion

### 6.1. Overall Performance

The comprehensive evaluation revealed a clear performance hierarchy among the models. The **XGBoost classifier emerged as the top-performing model**, achieving a mean accuracy of **88.67% ($\pm$ 1.52%)** across the five independent test sets. It was closely followed by the Random Forest model at **86.31% ($\pm$ 1.29%)**. The neural network models, while still highly effective, ranked third and fourth, with the MLP achieving **85.23% ($\pm$ 0.92%)** and the optimized CNN reaching **82.97% ($\pm$ 1.35%)**. These results underscore the efficacy of gradient-boosted trees for this classification task. The confusion matrix for the best performing XGBoost model is presented in Figure 1, illustrating its strong true positive rate and balanced precision-recall trade-off.

### 6.2. The Critical Role of PCA in Feature Extraction

A key finding of this work is the undeniable importance of PCA as a feature engineering step. While the best-performing CNN achieved ~83% accuracy on PCA-transformed data, its performance on raw spectral data was significantly lower, peaking at only 70.02% after extensive tuning. This performance gap highlights PCA's dual role in both reducing dimensionality and acting as a powerful denoising filter.

Figure 2 displays the PCA scree plot, illustrating the explained variance captured by each principal component. Physical interpretation of the principal components provides insight into this effect. Analysis of the PCA loadings reveals that the first component (PC1) is primarily a flat continuum, corresponding to the mean transit depth and thus planetary radius. The second component (PC2), however, shows strong weights at wavelengths corresponding to known absorption features of methane (~3.3 $\mu m$) and ozone (~4.7 $\mu m$). By training on components 2-102, we effectively force the models to learn from the chemical "fingerprints" of the atmosphere. Figure 3 shows the PCA reconstruction errors across the test set (defined as the fractional difference between the PCA-reconstructed spectrum and the true spectrum), confirming the model's ability to retain vital absorption features while discarding broad-spectrum noise.

### 6.3. Error Analysis and Parameter Dependencies

To further understand the model's performance, we analyzed the classification errors as a function of physical parameters. A summary of the parameter ranges explored is provided in Table II. 

Figure 4 presents a triangle plot of planet and stellar parameters, with data points colored based on correct versus incorrect test set predictions. This visualization reveals that errors are more frequent for planets with very low atmospheric temperatures or higher surface gravity, where atmospheric scale heights are compressed, leading to weaker absorption signals. Additionally, Figure 5 provides a similar triangle plot focusing on atmospheric parameters (mixing ratios of CH4 and O3). As expected, the model struggles near the predefined habitability thresholds (Log(CH4) = -6.0 and Log(O3) = -7.0), highlighting the inherent ambiguity in marginal biosignature detections (detailed in Section 6.X).

### 6.4. Model-Specific Behavior and Application Trade-offs

While XGBoost demonstrated the highest overall accuracy, the MLP model exhibited a uniquely valuable characteristic: a significantly higher mean recall of **94.56% ($\pm$ 1.46%)**. In the context of searching for biosignatures, where the cost of a false negative (missing a potentially life-bearing planet) is exceptionally high, this makes the MLP a compelling alternative. Its tendency to produce more false positives (lower precision) is an acceptable trade-off for its superior ability to correctly identify nearly all true positive cases. This suggests that for a large-scale survey mission, an MLP-based model could be the optimal choice for an initial, highly sensitive "first-pass" candidate selection, with higher-precision models like XGBoost used for subsequent vetting.
