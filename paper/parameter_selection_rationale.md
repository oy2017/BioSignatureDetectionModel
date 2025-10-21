# Rationale for System Parameter Selection in Synthetic Spectral Datasets for the Ariel Mission

## 1. Introduction

The generation of synthetic exoplanet transmission spectra is a critical step in the development and testing of machine learning models for biosignature detection. The physical plausibility of these synthetic datasets is paramount to ensuring that a model's performance is representative of what can be expected from observational data. This document outlines the rationale for the selection of system parameters used to generate two distinct datasets of synthetic spectra using the **MultiREx simulation framework**, corresponding to the European Space Agency's (ESA) Ariel mission baseline: one for planets with nitrogen-dominated (`N2`) atmospheres and another for hydrogen-dominated (`H2`) atmospheres.

## 2. Ariel Mission Scientific Context and Atmospheric Diversity

The Ariel space mission, scheduled for launch in 2029, is designed to conduct a large-scale survey of approximately 1,000 exoplanet atmospheres. A key objective is to perform a chemical census of planets in our galactic neighborhood to understand their formation and evolution. The mission's target list is intentionally diverse, encompassing a wide range of planet types, from hot gas giants to smaller, rocky super-Earths.

This diversity implies a wide range of expected atmospheric compositions:
- **Hydrogen-Dominated Atmospheres:** Gas giants and mini-Neptunes, due to their high gravitational potential, are expected to retain their primordial atmospheres, which are primarily composed of hydrogen (`H2`) and helium.
- **Nitrogen-Dominated Atmospheres:** Smaller, terrestrial planets (super-Earths) that have undergone significant atmospheric evolution, potentially through processes like volcanic outgassing after the loss of their primordial envelope, may possess secondary atmospheres rich in heavier molecules such as nitrogen (`N2`), water (`H2O`), or carbon dioxide (`CO2`).

Given this context, the generation of separate datasets for both `H2` and `N2` fill gases is a scientifically motivated step to train models capable of handling the diverse targets Ariel will observe.

## 3. Parameterization Strategy and Stability-Driven Adjustments

The primary goal was to create two parallel datasets that were identical in all respects except for the atmospheric fill gas. However, initial simulations revealed that this approach was not physically viable across the entire desired parameter space.

### 3.1. N2-Dominated System Parameters

For the `N2`-dominated atmospheres, a broad parameter space was defined to capture the full diversity of potential Ariel targets, including smaller rocky worlds. The selected ranges were:

- **Planet Radius:** 1.0 - 15.0 Earth Radii
- **Planet Mass:** 1.0 - 500.0 Earth Masses
- **Star Temperature:** 2500 - 7500 K
- **Atmosphere Temperature:** 500 - 2500 K

This configuration proved to be stable, and the MultiREx simulation successfully generated the entire dataset without any physical model failures.

### 3.2. H2-Dominated System Parameters and Physical Plausibility

Applying the same broad parameter ranges to H2-dominated atmospheres within the MultiREx simulation framework led to a high frequency of model failures. These failures, manifesting as `NaN` (Not a Number) values in the output spectra, serve as a key indicator that the input parameters describe a planetary system that is not physically sound.

The root cause was traced to the combination of a low planet mass and a large radius which, when coupled with the low mean molecular weight of an H2-rich atmosphere, results in exceptionally low-density planets ("puffy" planets) with extremely large atmospheric scale heights. Such configurations are often physically unstable, and the underlying radiative transfer models within MultiREx correctly failed to converge for these non-physical scenarios.

To ensure the physical plausibility of the entire generated dataset, the parameter space for the H2-dominated systems was constrained to exclude these unstable regimes. This was achieved by adjusting the planet mass and radius ranges to be more representative of gas giants and mini-Neptunes. The final, stable parameters were:

- **Planet Radius:** 5.0 - 15.0 Earth Radii
- **Planet Mass:** 20.0 - 300.0 Earth Masses

These adjustments were critical for ensuring that every generated spectrum in the dataset corresponds to a physically realistic planetary system.

## 4. Conclusion

The final parameter sets for the `N2` and `H2` simulations reflect a necessary, physically-motivated divergence. The generation of stable and realistic synthetic spectra via the MultiREx framework requires that the selected planetary parameters are compatible with the chosen atmospheric composition. The large number of simulation failures producing `NaN` spectra under the initial broad parameters for H2-rich worlds correctly identified a regime of physically implausible planets. By constraining the parameter space for H2-dominated atmospheres, we ensured the successful generation of a dataset where every spectrum corresponds to a physically stable system. This approach ensures that both datasets, while distinct, represent valid and stable scenarios for the training and evaluation of atmospheric retrieval and classification models.
