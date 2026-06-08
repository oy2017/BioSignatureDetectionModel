# BioSignature Detection Model

This repository contains a machine learning pipeline for detecting atmospheric biosignatures in high-dimensional exoplanet transmission spectra, simulated at an Ariel-like resolving power of 200.

## 🌟 Key Findings
*   **Optimal Architecture:** XGBoost achieved **88.59% accuracy**, matching the performance of deep neural networks while maintaining superior robustness.
*   **Baseline Independence:** XGBoost can detect biosignatures strictly from relative chemical morphology (PCA components 2-101), making it immune to the absolute continuum errors common in real telescope data.
*   **Neural Network Dependence:** MLP and CNN models proved to be **baseline-dependent**, requiring macro-physical components (PC0 and PC1) to orient their feature extraction.

## 🚀 Core Scripts
*   `run_master_5set_evaluation.py`: The definitive script to train all four models and evaluate them across 5 independent test sets using the mathematically rigorous pipeline.
*   `run_master_gridsearch.py`: The exhaustive hyperparameter tuning script used for the final paper.
*   `plot_final_results.py`: Generates the primary comparison visualizations.
*   `test_pc0_significance.py`: Performs McNemar's statistical tests to prove the baseline independence/dependence findings.

## 📖 Documentation
For a full technical history of the project, including the discovery and fix of the **PCA Scaling Bug** and the physical interpretation of the principal components, see:
*   [**Technical Experiment Summary**](experiment_summary.md)
*   [**Deep Learning Tuning Journal**](cnn_tuning_journal.md)
*   [**Analysis & Visualization Notes**](analysis_and_visualization_documentation.md)

## 📊 Results Summary (Table II)
| Model | Accuracy | F1-Score |
| :--- | :---: | :---: |
| **XGBoost** | 88.59% (± 1.66%) | **88.73% (± 1.63%)** |
| **MLP** | 88.17% (± 0.96%) | **88.28% (± 0.94%)** |
| **Random Forest** | 86.69% (± 0.70%) | **86.67% (± 0.74%)** |
| **1D-CNN** | 81.86% (± 0.86%) | **80.59% (± 0.90%)** |
