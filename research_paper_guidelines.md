# Research Paper Section Guidelines

This document outlines the expected level of technical detail and content for the **Methods** and **Results + Discussion** sections of your research paper, drawing examples from the `Yip_2021_AJ_162_195.pdf` paper.

## I. Methods Section: Aim for Reproducibility and Justification

The Methods section should be highly technical and detailed, providing enough information for another researcher to replicate your experiment exactly. Every significant step and parameter choice should be explicitly stated.

### 1. Data Generation
*   **Technical Detail:** Very specific about the *source* of the data, the *number* of samples, and the *exact ranges* and *sampling methods* for all physical parameters.
*   **Example from Paper (Section 2.2. Data Generation):
    "For the purposes of this study, we generated synthetic planetary atmospheres from planets contained in the Ariel Target list (Edwards et al. 2019a). A total of 11,940 transmission spectra were produced... Table 1 summarizes the sampling range, sampling method, and their respective scales."
*   **Your project's equivalent:** You would specify `multirex` (v0.3.1), 3000 training / 600 test spectra, H2-dominated, wavelength 0.5-7.8 $\mu m$ (R=200, SNR=15), and the exact ranges for `planet_radius` (1.0-26.0 $R_{\oplus}$), `planet_mass` (1.0-300.0 $M_{\oplus}$), `atm_temperature` (500-2500 K), `star_temperature` (2500-7500 K), and log-abundances for specific gases (H2O, CH4, CO, CO2, NH3). Also, your specific biosignature definition.

### 2. Data Preprocessing
*   **Technical Detail:** Describe all transformations, scaling, dimensionality reduction, and data splitting with precise parameters.
*   **Example from Paper (Section 2.3. Data Preprocessing):
    "The synthetic spectra and their corresponding AMPs are standardized (normalized so that each feature, i.e., wavelength bin and each AMP has zero mean and unit variance) to facilitate the training of the DNN models... The standardized data set is then split uniformly at random into three subsets: the original training set (70%), the validation set (10%), and the test set (20%)."
*   **Your project's equivalent:** Mention `StandardScaler`, `PCA` (specifying components 2-102), the rationale for skipping PC0/PC1, and your data cleaning step (filtering values > 1.0).

### 3. Model Training
*   **Technical Detail:** Provide full architectural details (number of layers, units, activation functions, dropout rates), optimizer, loss function, learning rate, epochs, batch size, and cross-validation strategy.
*   **Example from Paper (Section 2.4. Model Training & Appendix A):
    "We trained a DNN to perform a multi-output regression task... The model is trained in a supervised manner by minimizing the Mean Squared Error (MSE)... In all cases, the models were trained for 100 epochs with an initial learning rate of 0.01 and a learning rate decay of 10–4 using the Adam optimizer."
*   **Your project's equivalent:** For XGBoost/Random Forest, specify `GridSearchCV` with `StratifiedKFold` (`n_splits=3`), `scoring='accuracy'`, and the best parameters found (e.g., `n_estimators=300`, `max_depth=7`, `learning_rate=0.05`, `subsample=0.8` for XGBoost). For deep learning models, mention `Sequential` API, `Dense`/`Conv1D` layers, `BatchNormalization`, `Dropout`, `Activation('relu')`/`'sigmoid'`, `Adam` optimizer, `binary_crossentropy` loss, `epochs=100`, `batch_size=X`, and `EarlyStopping` (`patience=Y`).

### 4. Software and Hardware
*   **Technical Detail:** List all major libraries and the computational environment.
*   **Example from Paper (Appendix A):** "All the networks were developed using the open source Keras (Version 2.3.1) Python module... with Tensorflow (Version 2.4.1) as backend."
*   **Your project's equivalent:** Python 3, `multirex`, `scikit-learn`, `XGBoost`, `TensorFlow`/`Keras`, `pandas`, `numpy`, and mention CPU-based computation.

## II. Results + Discussion Section: Interpret and Contextualize

This section presents your findings, interprets their physical meaning, compares them to prior work, and discusses limitations and future directions.

### 1. Presentation of Results
*   **Technical Detail:** Use quantitative metrics and refer directly to figures and tables.
*   **Example from Paper (Table 4):
    "Table 4 shows the average performance of each architecture across five runs... All three architectures yielded models with comparable predictive performances."
*   **Your project's equivalent:** Present the final accuracy with standard deviations (e.g., "XGBoost achieved a mean accuracy of 88.67% $\pm$ 1.52%..."). Refer to your bar charts, confusion matrices, and physics error plots.

### 2. Interpretation and Physical Insight
*   **Technical Detail:** Explain *why* the models perform as they do, linking to physical phenomena. Use concepts like PCA loadings, absorption features, and physical parameter correlations.
*   **Example from Paper (Section 3.2 & 4.2):
    "In our application on the simulated Ariel-like data set, the model's predictions on the gases exhibit a similar trend: the prediction starts off with small bias and variance at high abundances, and both the bias and the variance gradually become higher as the abundance drops... A molecule's absorption feature is most prominent at high abundances, and this helps to tightly constrain the model's predictions."
*   **Your project's equivalent:** Discuss why XGBoost performs best (e.g., its ensemble nature and decision boundaries). Explain *why* PCA is critical, referring to your PCA loading plots (e.g., "PC2 highlights specific absorption features of Methane at 3.3$\mu m$ and Ozone at 4.7$\mu m$"). Discuss the trade-offs (e.g., MLP's high recall vs. lower precision for critical biosignature detection).

### 3. Comparison and Context
*   **Technical Detail:** Compare your results to benchmarks (like other models in your study) and broader literature.
*   **Example from Paper (Section 5):
    "We demonstrate that several DNN architectures (MLPs, CNNs, and LSTMs) are capable of producing models that achieve good predictive performance in this task... we found that they all behaved similarly for this data set, and that they are capable of reliably determining molecular abundances down to as low as 10^-5.8."
*   **Your project's equivalent:** Compare your 91.53% XGBoost to the previous ~87% benchmarks and the CNN's 84.92%. Discuss the implications of your findings for exoplanet atmospheric characterization.

In essence, aim for **clarity, conciseness, and specificity**. Every claim should be supported by data or a clear physical/computational rationale.
