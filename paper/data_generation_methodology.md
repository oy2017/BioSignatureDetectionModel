# Methodology for Generating a Balanced Exoplanet Biosignature Training Dataset

## Abstract

This document outlines the methodology employed for generating a synthetic exoplanet atmospheric dataset, specifically designed to train machine learning models for the detection of biosignatures within the context of the ESA Ariel mission. A stratified generation approach was utilized to create a perfectly balanced dataset, ensuring an equal representation of biosignature-positive and biosignature-negative planetary systems. This strategy addresses the critical challenge of data imbalance inherent in biosignature detection, thereby enhancing the robustness and efficacy of subsequent machine learning analyses.

## 1. Introduction

The search for atmospheric biosignatures in exoplanets represents a cornerstone of astrobiological research. The forthcoming Ariel mission aims to characterize the atmospheres of a large population of exoplanets, providing a wealth of data for such investigations ("Ariel Factsheet"). To effectively train machine learning models capable of identifying these subtle indicators of life, a high-quality, representative, and balanced training dataset is indispensable. This document details the methodology developed to generate such a dataset using the `MultiREx` simulation framework.

## 2. Rationale for a Balanced Dataset

The natural rarity of confirmed biosignatures in exoplanetary atmospheres poses a significant challenge for machine learning. If a machine learning model were trained on a dataset reflecting this inherent imbalance, it could achieve high overall accuracy by simply predicting the majority class ("no biosignature"), thereby failing to learn the nuanced features characteristic of actual biosignatures.

To mitigate this, a **perfectly balanced 50/50 split** between biosignature-positive and biosignature-negative examples was deemed crucial. This ensures that the machine learning model receives an equal opportunity to learn from both classes, compelling it to identify the distinguishing spectral features of biosignatures rather than relying on class prevalence.

## 3. Stratified Data Generation Methodology

To achieve the desired 50/50 split and ensure the scientific integrity of the dataset, a **stratified data generation methodology** was implemented. This approach involved intentionally creating distinct categories of planetary systems based on their atmospheric composition.

### 3.1. Biosignature Definition

A biosignature was defined by the simultaneous presence of methane (CH4) and ozone (O3) in the exoplanet's atmosphere, exceeding specific abundance thresholds. This definition is grounded in the concept of chemical disequilibrium, where the co-existence of an oxidizing gas (O3, a proxy for O2) and a reducing gas (CH4) strongly implies continuous replenishment by a biological source (Duque-Castaño et al.). The chosen thresholds were:
*   **Methane (CH4):** Log₁₀ abundance > -6 (equivalent to > 10⁻⁶ or 1 ppm)
*   **Ozone (O3):** Log₁₀ abundance > -7 (equivalent to > 10⁻⁷ or 0.1 ppm)

These thresholds were selected based on detectability within Ariel-like mission parameters and biological relevance (Duque-Castaño et al.).

### 3.2. Stratified Group Generation

The dataset was generated in distinct batches, each designed to produce a specific class of atmospheric composition:

*   **Biosignature Group (50% of dataset):** This group was generated with atmospheric compositions where both CH4 and O3 abundances were sampled from ranges guaranteed to exceed their respective biosignature thresholds. This ensures that all planets in this group are correctly labeled as biosignature-positive.

*   **Non-Biosignature Group (50% of dataset):** To provide a diverse set of negative examples, this group was further subdivided into three categories, each designed to ensure that the biosignature criteria were *not* met:
    *   **High CH4, Low O3:** Planets with significant methane but insufficient ozone.
    *   **Low CH4, High O3:** Planets with significant ozone but insufficient methane.
    *   **Low CH4, Low O3:** Planets with low abundances of both methane and ozone.
    This stratification within the non-biosignature class prevents the model from learning overly simplistic negative indicators and encourages it to identify the specific combination required for a biosignature.

### 3.3. Parameter Sampling

For all generated planets, other system parameters (e.g., stellar temperature, stellar radius, planet mass, planet radius, semi-major axis) were sampled from ranges representative of the diverse exoplanet population targeted by the Ariel mission.

## 4. Conclusion

The implementation of a stratified data generation methodology has successfully yielded a high-quality training dataset that is perfectly balanced between biosignature-positive and biosignature-negative examples. This approach ensures that the resulting machine learning models will be robustly trained to identify the specific chemical signatures indicative of life, thereby contributing significantly to the scientific objectives of exoplanet characterization missions such as Ariel.

## Works Cited

"Ariel Factsheet." *ESA*, European Space Agency, www.esa.int/Science_Exploration/Space_Science/Ariel/Ariel_factsheet. Accessed 17 Oct. 2025.

"Atmospheric Biosignatures." *Astrobiology at NASA*, NASA, www.astrobiology.nasa.gov/research/astrobiology-at-nasa/astrobiology-strategy/appendix-c-glossary/. Accessed 17 Oct. 2025.

Duque-Castaño, David S., et al. "Machine-assisted classification of potential biosignatures in earth-like exoplanets using low signal-to-noise ratio transmission spectra." *arXiv preprint arXiv:2407.19167v2*, 2024.
