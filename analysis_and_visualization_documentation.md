# Analysis & Visualization Documentation
**Date:** January 24, 2026
**Project:** BioSignature Detection Model

## 1. PCA Analysis & Signal Isolation

### A. The "Skip PC0-PC1" Strategy
A breakthrough in this project was identifying that the most "prominent" variance in the data is actually noise for our classification task.
*   **Discovery Script:** `analyze_pca_correlation.py`
*   **Finding:** Principal Components 0 and 1 account for the vast majority of spectral variance but have near-zero correlation with the biosignature label. They capture dominant physical systematics (Star Temp, Planet Radius).
*   **Strategy:** By starting our feature index at PC2, we "neutralize" these massive systematics and allow models to focus on the subtle chemical absorption features.

### B. Reconstruction Quality
**Script:** `measure_reconstruction_stats.py`
We measured how much "signal" is lost when reducing dimensions.
*   **Finding:** 100 components (indices 2-102) provide an optimal balance, maintaining low Mean Squared Error (MSE) while maximizing the correlation between features and the target label.

## 2. Comparative Physics Analysis
We developed a suite of tools to understand *where* models fail in physical space.
*   **Chemical Distribution:** `plot_chemical_scatter_comparison.py` visualizes model errors against Log(CH4) and Log(O3) concentrations.
*   **Physical Parameters:** `plot_comprehensive_error_analysis.py` creates a "physics dashboard" showing error rates against Star Temp, Planet Mass, and Planet Radius.

### C. The Carbon Monoxide (CO) Check
**Graph:** `error_vs_co_abundance.png`
We specifically tested if CO, a common atmospheric gas with strong absorption features, acts as a "confounder" for our biosignature detection.
*   **Finding:** The error rate remains flat (~10-15% for XGBoost) regardless of CO concentration (from Log -9 to -4).
*   **Conclusion:** The models successfully distinguish biosignature gases (CH4/O3) from CO. High CO abundance does **not** trigger false positives or mask the signal.

## 3. Stability & Variance Testing
To ensure results weren't due to random luck, we implemented rigorous stability testing.
*   **Data Generation:** `generate_5_sets.py` fragments the original test set into 5 independent, shuffled versions.
*   **Measurement:** `measure_model_variance.py` runs a model against all 5 sets to calculate mean accuracy and standard deviation.
*   **Result:** XGBoost and MLP showed high stability (+/- 1.0%), while CNNs showed higher variance (+/- 4.0%), further justifying the choice of XGBoost.

## 4. Visualization Script Index
*   `visualize_pca_reconstruction.py`: Plots original spectrum vs. PCA-reconstructed spectrum.
*   `generate_co_error_plot.py`: Specifically analyzes errors relative to Carbon Monoxide abundance.
*   `generate_separate_scatter_plots.py`: Creates individual diagnostic plots for each model type.

## 5. Conclusion
Our analysis pipeline proves that the biosignature signal is **high-frequency and subtle**, requiring the removal of dominant low-frequency systematics (PC0-1) and a wide feature set (100+ components) to achieve ~87% accuracy.
