# Data Generation Parameter Ranges

This table summarizes the parameter ranges used for generating the synthetic exoplanet atmospheric spectra with the `multirex` library.

| Parameter                | Range                    | Unit          | Notes                                     |
| :----------------------- | :----------------------- | :------------ | :---------------------------------------- |
| **Physical System**      |                          |               |                                           |
| Planet Radius            | 1.0 - 26.0               | $R_{\oplus}$ (Earth Radii) |                                           |
| Planet Mass              | 1.0 - 300.0              | $M_{\oplus}$ (Earth Masses) |                                           |
| Star Temperature         | 2500 - 7500              | K (Kelvin)    |                                           |
| Atmosphere Temperature   | 500 - 2500               | K (Kelvin)    | Isothermal atmosphere                     |
| **Spectral Data**        |                          |               |                                           |
| Wavelength Range         | 0.5 - 7.8                | $\mu m$ (microns) | Resolution R=200, 200 channels            |
| SNR                      | 15                       | -             | Constant signal-to-noise ratio            |
| **Chemical Abundances (Log Mixing Ratios)** |          |               |                                           |
| Biosignature Definition  | Log($CH_4$) $\ge$ -6 AND Log($O_3$) $\ge$ -7 | -             | Defines a 'biosignature' label              |
| H2O (Water)              | -10 to -1                | Log Mixing Ratio | Sampled uniformly in log space            |
| CO (Carbon Monoxide)     | -9 to -3                 | Log Mixing Ratio | Sampled uniformly in log space            |
| CO2 (Carbon Dioxide)     | -9 to -3                 | Log Mixing Ratio | Sampled uniformly in log space            |
| NH3 (Ammonia)            | -9 to -3                 | Log Mixing Ratio | Sampled uniformly in log space            |
| CH4 (Methane)            | -10 to -3                | Log Mixing Ratio | Sampled uniformly in log space (lower bound for non-bio) |
| O3 (Ozone)               | -10 to -1                | Log Mixing Ratio | Sampled uniformly in log space (lower bound for non-bio) |
