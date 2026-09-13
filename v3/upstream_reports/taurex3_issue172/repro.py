# Minimal reproduction with stock TauREx 3.3.2 (no MultiREx): the Exo-Transmit table's pressure line is scaled by 1e5
import numpy as np
from taurex.opacity.exotransmit import ExoTransmitOpacity
op = ExoTransmitOpacity("opacH2O.dat")              # table from github.com/elizakempton/Exo_Transmit/Opac
with open("opacH2O.dat") as f:
    f.readline(); file_p = np.array(f.readline().split(), float)
print("pressure line in the file      :", file_p[[0, 1, -1]])
print("TauREx pressureGrid [Pa]       :", np.asarray(op.pressureGrid)[[0, 1, -1]])
print("ratio                          :", np.asarray(op.pressureGrid)[0] / file_p[0])
