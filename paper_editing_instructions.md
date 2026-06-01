# Comprehensive Editing Instructions for Final Paper.pdf

This document provides a line-by-line guide to addressing the reviewer comments found in the PDF.

## Page 6: Methodology & Dataset

### Comment [1]: Reframing the Goal
- **Original Text:** "This study was designed to assess how effectively different machine learning models can identify atmospheric biosignatures in high-resolution exoplanet transmission spectra."
- **Feedback:** "Be careful of the framing... A more useful goal"
- **Instruction:** Replace with: *"This study aims to design a machine learning model capable of distinguishing between biosignature and non-biosignature environments using high-resolution (HR) transmission spectra from an Ariel-like dataset."*

### Comment [2]: Methodology Overview
- **Original Text:** "A computational framework was developed to address the challenges posed by high-dimensional spectral data..."
- **Feedback:** "We assess a suite of machine learning models on mock data simulated via multirex..."
- **Instruction:** Replace with: *"We assess a suite of machine learning models on mock data simulated via the `multirex` library, using the following methodologies."*

### Comments [3] & [4]: TauREx Detail & Citation
- **Original Text:** "3.1. Dataset and Simulation Parameters ... generated using the TauREx 3 radiative transfer code."
- **Feedback:** "Add a section before discussing how TauREx actually works, give a little bit of detail... cite"
- **Instruction:** Insert a new subsection:
  > **3.1.1. Radiative Transfer Modeling**
  > The synthetic dataset of atmospheric transmission spectra was generated using the `multirex` (v0.3.1) Python library, which leverages the TauREx 3 radiative transfer code (Al-Refaie et al., 2021) [1]. TauREx 3 computes transmission spectra by simulating the transit of starlight through a planetary atmosphere, robustly modeling molecular absorption, Rayleigh scattering, and collision-induced absorption across atmospheric layers to produce physically realistic data.
  > 
  > *Citation to add to References:* Al-Refaie, A. F., et al. "TauREx 3: A fast, dynamic and extendable framework for retrievals." The Astrophysical Journal 917.1 (2021): 37.

### Comment [5]: Dataset Quantities
- **Original Text:** "It consisted of 3,000 simulated transmission spectra..."
- **Feedback:** "We generate a training set of X simulated..., as well as five independent testing sets, each with Y spectra"
- **Instruction:** Update with exact values: *"We generated a training set of 2,709 simulated planetary spectra (reduced from 3,000 after removing spectra with physically impossible transit depths > 1.0 or NaN values), as well as five independent testing sets, each containing ~550 clean spectra."*

### Comment [6]: Atmosphere Fact Check
- **Original Text:** "...hydrogen-dominated atmospheres."
- **Feedback:** "Fact check, and 'hydrogen-dominated atmospheres, following [6-8]'"
- **Instruction:** Append the citation justifying the choice of hydrogen-rich atmospheres for these exoplanet simulations: *"...hydrogen-dominated atmospheres, following the canonical assumptions for highly observable exoplanets (Seager et al., 2013)."*
  > *Citation to add to References:* Seager, S., et al. "Biosignature gases in H2-dominated exoplanet atmospheres." The Astrophysical Journal 777.2 (2013): 95.

### Comment [7]: Spectral Range
- **Original Text:** "...with more than 5,000 spectral bins."
- **Feedback:** "spanning wavelengths X_Y"
- **Instruction:** Replace with: *"spanning wavelengths from 0.5 to 7.8 $\mu m$ at a spectral resolution of R=200, with a constant signal-to-noise ratio (SNR) of 15 applied to all observations."*

### Comments [8], [10], & [11]: Procedure Citations
- **Original Text:** "Table I. Parameter grid choices ... following the Ariel parameter grids outlined in [8]."
- **Feedback:** "fact check, 'following the procedures of XXXX, et al. (20XX)'... Cite where these choices came from in caption"
- **Instruction:** Update text and Table I caption to cite the specific Ariel Data Challenge paper: *"Table I. Parameter grid choices for synthetic spectra, following the procedures of the Ariel Data Challenge (Yip et al., 2021)."*
  > *Citation to add to References:* Yip, K. H., et al. "The Ariel Data Challenge 2021: Extracting planetary signals from the Ariel Space Telescope." (2021).

### Comment [9]: Parameter Variation
- **Original Text:** "...were retained as metadata for later, physics-based error analysis."
- **Feedback:** "were simultaneously varied along the pre-defined grid outlined in Table X"
- **Instruction:** Add: *"Physical parameters were simultaneously varied along the pre-defined grid outlined in Table I."*

---

## Page 7: Preprocessing & Architectures

### Comment [12]: PCA Definition
- **Original Text:** "Principal Component Analysis (PCA) was then used to reduce the dimensionality..."
- **Feedback:** "Briefly define and give a mathematical definition..."
- **Instruction:** Add: *"PCA is an orthogonal linear transformation that projects data into a new coordinate system via the eigendecomposition of the data's covariance matrix, allowing us to isolate distinct physical and chemical signals into uncorrelated components."*

### Comment [13]: Quantitative Correlation
- **Original Text:** "...showed little correlation with the target gas abundances."
- **Feedback:** "make this quantitative"
- **Instruction:** Update to: *"...showed little quantitative correlation (Pearson correlation coefficient $r < 0.1$) with the target gas abundances."*

### Comment [14]: Justify Component Findings
- **Original Text:** "These components were found to primarily capture broad physical effects..."
- **Feedback:** "show/justify this finding"
- **Instruction:** Explain the analysis: *"By analyzing the PCA component loadings and reconstructing the spectra, these components (PC0 and PC1) were found to represent the mean transit depth and overall spectral slope, thus encoding the overall planetary radius and stellar continuum level rather than specific chemical absorption features."*

### Comment [15]: Explained Variance
- **Original Text:** "...next 100 principal components (indices 2–102)."
- **Feedback:** "what is the explained variance captured in 100 components"
- **Instruction:** Add: *"...which cumulatively explain approximately 0.74% of the total variance (while PC0 and PC1 account for over 99%)."*

### Comment [16]: Visual Evidence
- **Original Text:** "...influence of large-scale astrophysical systematics."
- **Feedback:** "Show your plot of the explained variance... Show the error from pc reconstruction..."
- **Instruction:** Refer to your results: *"Figure X shows the scree plot of explained variance across the principal components. Additionally, Figure Y illustrates the PCA reconstruction and residual error across wavelengths for a representative test spectrum, demonstrating the retention of vital absorption features."*

### Comment [17]: Model Selection Rationale
- **Original Text:** "Three families of machine learning models were evaluated..."
- **Feedback:** "Justify here why you have chosen these models..."
- **Instruction:** Add: *"These models were selected to cover a spectrum of complexity, from robust ensemble methods capable of handling non-linear decision boundaries to deep neural networks that can automatically learn hierarchical feature representations."*

### Comment [18]: Random Forest Addition
- **Original Text:** Section 3.3.1. Gradient Boosted Decision Trees (XGBoost)
- **Feedback:** "Add in RF"
- **Instruction:** Insert a new subsection (*3.3.2. Random Forest*) describing the ensemble model:
  > **3.3.2. Random Forest**
  > A Random Forest classifier was evaluated as an additional tree-based ensemble approach. Implemented using `scikit-learn`, this model constructs a multitude of decision trees during training, each on a random subset of the data and features, and outputs the consensus prediction. This approach is highly robust against overfitting and effectively captures complex, non-linear relationships in high-dimensional spectral data.

### Comment [19]: Grid Search Detail
- **Original Text:** "The model was optimized using a log-loss objective function with a learning rate of 0.2."
- **Feedback:** "Discuss grid search, what parameters were tuned and why..."
- **Instruction:** Expand: *"The XGBoost grid search encompassed learning rates $\in \{0.01, 0.05, 0.1, 0.2\}$, maximum depths $\in \{3, 5, 7, 10\}$, number of estimators $\in \{100, 200, 300\}$, and subsample ratios $\in \{0.8, 1.0\}$. The optimal parameters were found to be `n_estimators=300`, `max_depth=5`, `learning_rate=0.2`, and `subsample=1.0`."*

### Comment [20]: XGBoost Citation
- **Feedback:** "cite"
- **Instruction:** Add a citation for the XGBoost framework at the end of section 3.3.1:
  > *Citation:* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

### Comment [21]: Hyperparameter Discussion (MLP)
- **Original Text:** "...improve training stability and reduce overfitting."
- **Feedback:** "Hyperparameter search discussion"
- **Instruction:** Detail the search: *"The MLP grid search evaluated architectures $\in \{[256, 128], [512, 256, 128]\}$, dropout rates $\in \{0.3, 0.4, 0.5\}$, batch sizes $\in \{64, 128\}$, and learning rates $\in \{0.0005, 0.001\}$. The optimal configuration was found to be a [512, 256, 128] architecture with $LR=0.0005$ and $BatchSize=128$."*

### Comment [22]: MLP Citation
- **Feedback:** "cite"
- **Instruction:** Add a standard citation for deep Multi-Layer Perceptrons or the specific regularization techniques used (Dropout/BatchNorm):
  > *Citation:* Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

### Comments [23] & [24]: Finalizing CNN and ResNet
- **Feedback:** "update"
- **Instruction (Comment 23 - 1D-CNN):** Update the description of the optimized 1D-CNN architecture. It consists of three convolutional layers with increasing filters (32, 64, 128), utilizing a `GaussianNoise` input layer (std=0.05) to improve generalization on raw spectra. Each layer is followed by `BatchNormalization`, `ReLU` activation, `MaxPooling1D`, and `Dropout` (p=0.3).
- **Instruction (Comment 24 - ResNet):** Specify the ResNet architecture implemented for end-to-end learning. It employs residual blocks containing skip connections to mitigate the vanishing gradient problem, allowing for deeper feature extraction. Preprocessing for the ResNet includes a `Gaussian smoothing` step (kernel size 3) applied to the raw spectra prior to scaling to reduce high-frequency noise.

---

## Page 8: Validation & Simulation Range

### Comment [25]: Review CV Usage
- **Original Text:** "Although K-fold cross-validation was considered..."
- **Feedback:** "Review your code, i remember you using cv but double check"
- **Instruction:** Update to reflect code reality: *"Hyperparameters for the models were systematically optimized using an exhaustive grid search with stratified 3-fold cross-validation."*

### Comments [26] & [27]: Metric Definitions
- **Original Text:** "...mean accuracy and standard deviation were reported."
- **Feedback:** "Evaluated how. What metrics are you using, how are they defined?"
- **Instruction:** Update to: *"Each model was evaluated on all five test sets, and we report the mean accuracy and standard deviation in Section 6. The metrics (Accuracy, Precision, Recall, F1-Score) are defined below."*

### Comment [28]: Forward Reference
- **Original Text:** "...abundance, allowing the identification..."
- **Feedback:** "in section 4.X"
- **Instruction:** Update the reference to: *"...(detailed in Section 6.X)."*

### Comment [29]: Version Number
- **Original Text:** "...`multirex` (v0.3.1)..."
- **Feedback:** "i like the version number"
- **Instruction:** No changes required. Retain the `(v0.3.1)` version number.

### Comment [30]: Data Cleaning
- **Original Text:** "...removed any simulated spectra containing physically impossible transit depths (> 1.0) or NaN values..."
- **Feedback:** "important, keep"
- **Instruction:** No changes required. Retain this data cleaning description.

### Comment [31]: Exhaustive Parameters
- **Original Text:** List of physical parameters.
- **Feedback:** "Give an exhaustive list"
- **Instruction:** Include all 9 parameters:
    *   **Planet Radius:** 1.0 – 26.0 $R_{\oplus}$
    *   **Planet Mass:** 1.0 – 300.0 $M_{\oplus}$
    *   **Atmospheric Temperature:** 500 – 2500 K
    *   **Atmospheric Base Pressure:** $10^5$ – $10^6$ Pa
    *   **Atmospheric Top Pressure:** 1 – 10 Pa
    *   **Stellar Temperature:** 2500 – 7500 K
    *   **Stellar Radius:** 0.1 – 1.7 $R_{\odot}$
    *   **Stellar Mass:** 0.1 – 1.7 $M_{\odot}$
    *   **Semi-Major Axis:** 0.01 – 0.5 AU

### Comment [32]: SNR Specification
- **Original Text:** "...with a constant signal-to-noise ratio (SNR) of 15 applied to all observations."
- **Feedback:** "keep"
- **Instruction:** No changes required. Retain this detail.

### Comment [33]: Threshold Citation
- **Original Text:** "...concentration of CH4 being -6.0..."
- **Feedback:** "see if another paper... used that, and cite"
- **Instruction:** Add citation for the -6.0/-7.0 log mixing ratio thresholds: *"...following the thresholds outlined in Duque-Castaño et al. (2024)."*
  > *Citation to add to References:* Duque-Castaño, David S., et al. "Machine-assisted classification of potential biosignatures in earth-like exoplanets using low signal-to-noise ratio transmission spectra." arXiv preprint arXiv:2407.19167v2, 2024.

---

## Page 9: Training & Validation

### Comment [34]: Cross-Validation Method
- **Original Text:** "...with stratified 3-fold cross-validation, using accuracy as the scoring metric."
- **Feedback:** "good"
- **Instruction:** No changes required.

### Comment [35]: RF Hyperparameter Ranges
- **Original Text:** Final parameters for Random Forest.
- **Feedback:** "Report all your choices and hyperparameter ranges..."
- **Instruction:** List the full search space: *"The Random Forest search space included number of estimators $\in \{100, 200, 300\}$, maximum depth $\in \{\text{None}, 10, 20, 30\}$, minimum samples for an internal node split $\in \{2, 5, 10\}$, and minimum samples per leaf $\in \{1, 2, 4\}$. The optimal parameters were found to be `n_estimators=300`, `max_depth=None`, `min_samples_split=5`, and `min_samples_leaf=2`."*

### Comment [36]: API Choice
- **Original Text:** "...built with the TensorFlow Keras API..."
- **Feedback:** "good"
- **Instruction:** No changes required.

### Comment [37]: Text Formatting
- **Original Text:** "`BatchNormalization`"
- **Feedback:** "batch normalization"
- **Instruction:** Change to standard text: *"batch normalization"*.

### Comment [38]: Metric Equations
- **Original Text:** "...Accuracy, Precision, Recall, and F1-Score,"
- **Feedback:** "good, but also define what these are, maybe use equation"
- **Instruction:** Add mathematical definitions (TP, TN, FP, FN):
    *   **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
    *   **Precision:** $\frac{TP}{TP + FP}$
    *   **Recall:** $\frac{TP}{TP + FN}$
    *   **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

### Comment [39]: Software Versions
- **Original Text:** "...scikit-learn, XGBoost, TensorFlow, pandas, and numpy..."
- **Feedback:** "find version numbers for software"
- **Instruction:** Use detected versions: *"...leveraging `scikit-learn` (v1.7.1), `XGBoost` (v3.1.1), `TensorFlow` (v2.20.0), `pandas` (v2.3.1), and `numpy` (v2.3.2)."*

---

## Page 10: Checklist of Figures and Tables

Ensure the following are included in the final manuscript. I have located the corresponding generated files in the repository for your convenience:

1.  **Confusion Matrices:** For the top-performing model (XGBoost).
    *   *File:* `final_results/H2_best_xgboost_confusion_matrix.png`
2.  **PCA Scree Plot:** Showing explained variance by component.
    *   *File:* `H2_pca_scree_plot.png` (or `H2_pca_explained_variance.png`)
3.  **PCA Reconstruction Error:** Plot showing ((PCA - true)/true) - 1 across wavelengths.
    *   *File:* `final_results/plots/pca_reconstruction_errors.png` (or one of the `H2_pca_reconstruction_*.png` examples)
4.  **Parameter Range Table:** Table I (updated with citations).
5.  **Triangle Plot (System):** Errors vs. Planet Radius, Planet Mass, Star Temp.
    *   *File:* `final_results/plots/corner_plot_errors_scatter_xgboost.png` (or `final_results/comprehensive_error_analysis_physics_v2.png`)
6.  **Triangle Plot (Atmosphere):** Errors vs. CH4 and O3 mixing ratios.
    *   *File:* `final_results/model_comparison_chemical_scatter.png` (or `final_results/scatter_error_xgboost.png`)
