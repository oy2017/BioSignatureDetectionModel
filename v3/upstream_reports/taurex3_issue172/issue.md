## Summary

`ExoTransmitOpacity._load_exo_transmit` multiplies the pressure line of an Exo-Transmit opacity table by `1e5`, i.e. it reads the pressures as bar and converts them to Pa:

https://github.com/ucl-exoplanets/taurex3/blob/a422762d006d625dca1ae2621612c252ffdd3380/src/taurex/opacity/exotransmit.py#L107

```python
self._pressure_grid = np.array([float(col) for col in lines[1].split()]) * 1e5
```

As far as I can tell, the Exo-Transmit tables are already in Pa, so every cross section is looked up at a pressure 1e5 times lower than the layer pressure. The same line is in the older `TauREx3_public` code.

## Why the tables appear to be in Pa

- Exo-Transmit's own reader turns the tabulated cross section into an opacity with `opac.P[j] / (KBOLTZMANN * opac.T[k])`, with `KBOLTZMANN = 1.380658E-23` J/K ([readopactable.c](https://github.com/elizakempton/Exo_Transmit/blob/master/readopactable.c), [constant.h](https://github.com/elizakempton/Exo_Transmit/blob/master/constant.h)). That is a number density only if P is in Pa.
- The same code compares `opac.P` directly with the pressures of the T-P profile files (`rt_transmission.c`, `Locate(NPRESSURE, opac.P, atmos.P[j], &b)`). The shipped profiles run from 0.1 to 1.01e5, i.e. up to 1 bar in Pa, and the user manual gives the cloud-top pressure "in units of Pa" and states that the code works in SI units.
- The table grid is 1e-4 ... 1e8. Read as bar, the top of the grid would be 1e8 bar.

## Reproduction (TauREx 3.3.2, table from the Exo-Transmit repository)

```python
import numpy as np
from taurex.opacity.exotransmit import ExoTransmitOpacity
op = ExoTransmitOpacity("opacH2O.dat")   # https://github.com/elizakempton/Exo_Transmit/blob/master/Opac/opacH2O.dat
with open("opacH2O.dat") as f:
    f.readline(); file_p = np.array(f.readline().split(), float)
print(file_p[[0, 1, -1]])                           # [1.e-04 1.e-03 1.e+08]
print(np.asarray(op.pressureGrid)[[0, 1, -1]])      # [1.e+01 1.e+02 1.e+13]
```

## Size of the effect

- Band-mean H2O cross section at 1000 K, stock loader divided by the loader with the grid divided by 1e5: 1.00 at 1e2 Pa; 0.88-1.17 at 1e4 Pa; 0.82-1.35 at 1e6 Pa (1.3-1.5, 1.6-1.7 and 3.9-4.1 um bands). Deep layers are the most affected; below the table's lowest converted pressure (10 Pa) everything is clamped to the 1e-4 Pa row.
- Transmission spectra of 1,804 planets (isothermal, 500-2500 K, H2O/CH4/CO/CO2/NH3, Exo-Transmit tables via MultiREx) compared with Exo-Transmit's own code run on the same atmospheres: median shape correlation 0.986 (10th percentile 0.867) with the stock loader, 0.9986 (0.997) with the pressure grid divided by 1e5; the corrected spectra are closer to Exo-Transmit for 95.6 % of planets.

## Suggested fix

Drop the `* 1e5` (or divide by 1e5 after reading), unless there is an Exo-Transmit-format table set in bar that this loader is meant for. If there is, a unit keyword on the loader would keep both working.

Downstream note: [MultiREx](https://github.com/D4san/MultiREx-public) ships Exo-Transmit tables and uses this loader by default, so its spectra are affected too.

Thanks for maintaining TauREx.
