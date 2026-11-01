"""Identifiability with a free radius against the fixed radius on the same 200 planets. Writes results/identifiability_free_radius.txt."""
import os, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = pd.read_csv('results/identifiability_calibration_freeradius.csv'); B = pd.read_csv('results/identifiability_calibration.csv'); J = F.merge(B, on='index', suffixes=('_f', '_b')); L = []
for m in ['CH4', 'H2O', 'CO2', 'NH3']:
    f, b = J[f'{m}_p_true_f'], J[f'{m}_p_true_b']; u = lambda x: ((x >= .05) & (x <= .95)).mean()
    L.append(f'{m} n={len(J)} undecided at 0.95: fixed {u(b):.3f} free {u(f):.3f}; p_true above 0.5: fixed {(b > .5).mean():.3f} free {(f > .5).mean():.3f}; p_true below 0.05: fixed {(b < .05).mean():.3f} free {(f < .05).mean():.3f}')
t = '\n'.join(L); open('results/identifiability_free_radius.txt', 'w').write(t + '\n'); print(t)
