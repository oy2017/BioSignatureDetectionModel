import numpy as np
from taurex.opacity.exotransmit import ExoTransmitOpacity
stock = ExoTransmitOpacity("opacH2O.dat")
fixed = ExoTransmitOpacity("opacH2O.dat"); fixed._pressure_grid = fixed._pressure_grid / 1e5
fixed._min_pressure, fixed._max_pressure = fixed._pressure_grid.min(), fixed._pressure_grid.max()
wl = 1e4 / np.asarray(stock.wavenumberGrid)
for P in (1e2, 1e4, 1e6):
    a, b = stock.opacity(1000.0, P), fixed.opacity(1000.0, P)
    for lo, hi in ((1.3, 1.5), (1.6, 1.7), (3.9, 4.1)):
        m = (wl > lo) & (wl < hi)
        print(f"T=1000 K, P={P:.0e} Pa, {lo}-{hi} um: band-mean cross section stock/fixed = {a[m].mean()/b[m].mean():.3f}")
