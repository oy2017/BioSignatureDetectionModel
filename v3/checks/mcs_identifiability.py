"""Identifiability under the second population (Mission Candidate Sample planets, FastChem equilibrium chemistry, clear) against POP-I on the same planets. Writes results/identifiability_mcs.txt."""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = pd.read_csv('results/identifiability_calibration_mcs.csv'); B = pd.read_csv('results/identifiability_calibration.csv'); J = A.merge(B, on='index', suffixes=('_m', '_b')); assert (J.name_m == J.name_b).all(); r = np.random.default_rng(408); L = [f'n={len(J)} same planets']
u = lambda x: ((x >= .05) & (x <= .95)).mean(axis=-1)
for m in ['CH4', 'H2O', 'CO2', 'NH3']:
    a, b, t, tb = J[f'{m}_p_true_m'].to_numpy(), J[f'{m}_p_true_b'].to_numpy(), J[f'{m}_true_m'].to_numpy(), J[f'{m}_true_b'].to_numpy(); ix = r.integers(0, len(J), (5000, len(J)))
    ci = np.percentile(u(a[ix]), [2.5, 97.5]); far = np.abs(t + 4) >= .5; farb = np.abs(tb + 4) >= .5
    L.append(f'{m} undecided: second population {u(a):.3f} ({ci[0]:.3f},{ci[1]:.3f}), POP-I same planets {u(b):.3f}; away from threshold {u(a[far]):.3f} and {u(b[farb]):.3f}; within half a decade {1 - far.mean():.2f} and {1 - farb.mean():.2f}; present {(t > -4).mean():.2f} and {(tb > -4).mean():.2f}; p_true above 0.5 {(a > .5).mean():.3f}, below 0.05 {(a < .05).mean():.3f}')
t = '\n'.join(L); open('results/identifiability_mcs.txt', 'w').write(t + '\n'); print(t)
