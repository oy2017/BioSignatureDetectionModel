"""The re-implemented classifier's errors split by the retrieval, under the second population and, as a check, POP-I on its 200 planets (expectations X). Writes results/identifiability_mcs_split.txt."""
import os, sys, numpy as np, pandas as pd, joblib
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, '.')
from bin_spectra import bin_native
import alfnoor_faithful as AF, alfnoor_trust as AT, alfnoor_jwst as AJ
from common import DATA, SEED
MOLS = ['CH4', 'H2O', 'CO2', 'NH3']; LEARNERS = ['KNN', 'MLP', 'RFC', 'SVC']; wl = np.load(os.path.join(DATA, 'native_wl.npy')); edges = AF.layout_edges('tier3_r20'); L_ = joblib.load('data/alfnoor_jwst/learners_all.joblib'); OUT = []
def run(tag, P, Y, R):
    ok = np.all(np.isfinite(Y), axis=1); P = P[ok].reset_index(drop=False).rename(columns={'index': '_index'}); Y = Y[ok]; keep = P['_index'].isin(R.index).to_numpy(); P, Y = P[keep], Y[keep]
    sig = AF.sigma(P, edges, 'radiometric'); X = bin_native(Y, wl, edges) + np.random.default_rng(SEED + 409).normal(0, 1, (len(P), len(edges) - 1)) * sig; Z = AJ.znorm(X, L_['cols']); a = R.loc[P['_index']]; tu = fu = tn = 0
    for m in MOLS:
        v = sum((AT.prob(L_['models'][('1e-4', m, k)], Z) >= .5).astype(float) for k in LEARNERS); y = a[f'{m}_true'].to_numpy() > -4; present = v > 2; split = v == 2; credit = np.where(split, .5, (present == y).astype(float)); err = 1 - credit; p = a[f'{m}_p_true'].to_numpy(); und = (p >= .05) & (p <= .95)
        OUT.append(f'{tag:<8}{m:<4} n={len(y)} present {y.mean():.2f} credit {credit.mean():.3f} calls present {present.mean():.2f}; errors {err.sum():.1f}, on undecidable {err[und].sum():.1f} ({err[und].sum() / max(err.sum(), 1e-9):.2f}), on decidable {err[~und].sum():.1f}; undecidable labels {und.mean():.3f}; balanced credit {(.5 * (credit[y].mean() + credit[~y].mean())) if 0 < y.mean() < 1 else float("nan"):.3f}')
        tu += err[und].sum(); fu += err[~und].sum(); tn += len(y)
    OUT.append(f'{tag:<8}all  labels {tn}: accuracy {1 - (tu + fu) / tn:.3f}; errors on undecidable {tu:.1f}, on decidable {fu:.1f}, share {tu / (tu + fu):.2f}')
M = pd.read_parquet(os.path.join(DATA, 'mcs', 'mcs_params.parquet')); M['T_p'] = M['atm temperature']; M['cloud_pressure'] = 1e6
run('second', M, np.load(os.path.join(DATA, 'mcs', 'mcs_native.npy')).astype(np.float64), pd.read_csv('results/identifiability_calibration_mcs.csv').set_index('index'))
src = AF.DIRS['faithful']; B = pd.read_csv('results/identifiability_calibration.csv').set_index('index'); B = B.loc[B.index.intersection(pd.read_csv('results/identifiability_calibration_mcs.csv')['index'])]
P1 = pd.read_parquet(os.path.join(src, 'pop1_params.parquet')).merge(M[['name', 'atm temperature']].drop_duplicates('name').rename(columns={'atm temperature': 'T_p'}), on='name', how='left')
run('POP-I', P1, np.load(os.path.join(src, 'pop1_native.npy')).astype(np.float64), B)
t = '\n'.join(OUT); open('results/identifiability_mcs_split.txt', 'w').write(t + '\n'); print(t)
