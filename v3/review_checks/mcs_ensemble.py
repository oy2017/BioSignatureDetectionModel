"""The matched ensemble at Ariel coverage on the second population (expectations X4 to X6). Writes results/identifiability_mcs_ensemble.txt."""
import os, sys, warnings, numpy as np, pandas as pd, joblib
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, '.')
from bin_spectra import bin_native
from common import DATA, SEED
import alfnoor_faithful as AF, alfnoor_trust as AT, alfnoor_jwst as AJ, matched_ariel as MA, round3_ariel as RA, round3_learner as R
MOLS = R.MOLS; LE = ['KNN', 'MLP', 'RFC', 'SVC']; wl = np.load(os.path.join(DATA, 'native_wl.npy')); e = AF.layout_edges('tier3_r20'); L_ = joblib.load('data/alfnoor_jwst/learners_all.joblib')
X, P = RA.population(e); s = AF.sigma(P, e, 'radiometric'); Y = (P[[f'atm {m}' for m in MOLS]].to_numpy() > -4).astype(int); hold = np.random.default_rng(SEED + 77).uniform(size=len(X)) < 0.15; nets = []
for k in range(R.MEMBERS):
    rng = np.random.default_rng(SEED + 8100 + k); Z = np.vstack([RA.draw(X[~hold], s[~hold], rng) for _ in range(R.COPIES)]); nets.append(MLPClassifier((256, 128), batch_size=256, early_stopping=True, n_iter_no_change=6, max_iter=60, random_state=SEED + 8100 + k).fit(Z, np.tile(Y[~hold], (R.COPIES, 1)))); print(k, flush=True)
prob = lambda Z: np.mean([n.predict_proba(Z) for n in nets], axis=0); t = R.thresholds(prob(RA.draw(X[hold], s[hold], np.random.default_rng(SEED + 8199))), Y[hold])
M = pd.read_parquet(os.path.join(DATA, 'mcs', 'mcs_params.parquet')); M['T_p'] = M['atm temperature']; M['cloud_pressure'] = 1e6; N = np.load(os.path.join(DATA, 'mcs', 'mcs_native.npy')).astype(np.float64); ok = np.isfinite(N).all(1); M = M[ok].reset_index(drop=False).rename(columns={'index': '_index'}); N = N[ok]
sig = AF.sigma(M, e, 'radiometric'); Xn = bin_native(N, wl, e) + np.random.default_rng(SEED + 409).normal(0, 1, (len(M), len(e) - 1)) * sig
pe = prob(MA.feats(Xn, sig)); Z = AJ.znorm(Xn, L_['cols']); R0 = pd.read_csv('results/identifiability_calibration_mcs.csv').set_index('index'); rr = M['_index'].isin(R0.index).to_numpy(); O = [f'{len(M)} planets, {rr.sum()} retrieved; thresholds {dict(zip(MOLS, np.round(t, 2)))}']; tot = np.zeros((2, 2))
for i, m in enumerate(MOLS):
    y = M[f'atm {m}'].to_numpy() > -4; v = sum((AT.prob(L_['models'][('1e-4', m, k)], Z) >= .5).astype(float) for k in LE); cc = np.where(v == 2, .5, ((v > 2) == y).astype(float)); ce = ((pe[:, i] >= t[i]) == y).astype(float); bal = lambda c: .5 * (c[y].mean() + c[~y].mean()) if 0 < y.mean() < 1 else float('nan'); fp = lambda c: 1 - c[~y].mean()
    q = R0.loc[M['_index'][rr]]; p = q[f'{m}_p_true'].to_numpy(); und = (p >= .05) & (p <= .95); ee, ec = (1 - ce)[rr], (1 - cc)[rr]; tot += [[ec[und].sum(), ec[~und].sum()], [ee[und].sum(), ee[~und].sum()]]
    O.append(f'{m} n={len(y)} present {y.mean():.2f}: balanced credit ensemble {bal(ce):.3f} classifier {bal(cc):.3f}; accuracy {ce.mean():.3f} {cc.mean():.3f}; false-present {fp(ce):.3f} {fp(cc):.3f}; retrieved errors undecidable {ee[und].sum():.1f} {ec[und].sum():.1f}, decidable {ee[~und].sum():.1f} {ec[~und].sum():.1f}')
O.append(f'retrieved errors: classifier undecidable {tot[0, 0]:.1f} decidable {tot[0, 1]:.1f}; ensemble undecidable {tot[1, 0]:.1f} decidable {tot[1, 1]:.1f}')
# ten noise draws, and a bootstrap over planets on each draw, added after the first run because two draws of the classifier differed by nine errors
D = []; cls = {m: [AT_m for AT_m in [L_['models'][('1e-4', m, k)] for k in LE]] for m in MOLS}; ids = M['_index'].to_numpy(); rows = []
for d in range(10):
    Xd = bin_native(N, wl, e) + np.random.default_rng(SEED + 500 + d).normal(0, 1, (len(M), len(e) - 1)) * sig; pe = prob(MA.feats(Xd, sig)); Z = AJ.znorm(Xd, L_['cols'])
    for i, m in enumerate(MOLS):
        y = M[f'atm {m}'].to_numpy() > -4; v = sum((AT.prob(c, Z) >= .5).astype(float) for c in cls[m]); cc = np.where(v == 2, .5, ((v > 2) == y).astype(float)); ce = ((pe[:, i] >= t[i]) == y).astype(float)
        p = R0[f'{m}_p_true'].reindex(ids).to_numpy(); und = (p >= .05) & (p <= .95); rows.append(pd.DataFrame(dict(draw=d, mol=m, id=ids, y=y, cc=cc, ce=ce, ret=rr, und=und)))
A = pd.concat(rows); A.to_csv('results/identifiability_mcs_ensemble_rows.csv.gz', index=False)
O.append('ten noise draws, mean +- sd, ensemble against classifier')
bal = lambda g, c: .5 * (g[g.y][c].mean() + g[~g.y][c].mean())
for m, g in A.groupby('mol', sort=False):
    a = g.groupby('draw').apply(lambda x: pd.Series(dict(be=bal(x, 'ce'), bc=bal(x, 'cc'), ae=x.ce.mean(), ac=x.cc.mean(), fe=1 - x[~x.y].ce.mean(), fc=1 - x[~x.y].cc.mean())))
    O.append(f"{m} balanced credit {a.be.mean():.3f} +- {a.be.std():.3f} {a.bc.mean():.3f} +- {a.bc.std():.3f}; accuracy {a.ae.mean():.3f} {a.ac.mean():.3f}; false-present {a.fe.mean():.3f} {a.fc.mean():.3f}")
q = A[A.ret]; g = q.groupby(['draw', 'und']).agg(ee=('ce', lambda x: (1 - x).sum()), ec=('cc', lambda x: (1 - x).sum())).reset_index()
for u, n in ((False, 'decidable'), (True, 'undecidable')):
    x = g[g.und == u]; O.append(f'retrieved 200, errors on {n} labels: ensemble {x.ee.mean():.1f} +- {x.ee.std():.1f}, classifier {x.ec.mean():.1f} +- {x.ec.std():.1f}')
rng = np.random.default_rng(410); P0 = q[(q.draw == 0)]; idu = P0.id.unique(); bs = []; G = {i: x for i, x in P0.groupby('id')}
for _ in range(2000):
    B = pd.concat([G[i] for i in rng.choice(idu, len(idu))]); d_ = B[~B.und]; bs.append((1 - d_.cc).sum() - (1 - d_.ce).sum())
O.append(f'classifier minus ensemble, errors on decidable labels, draw 0: {(1 - P0[~P0.und].cc).sum() - (1 - P0[~P0.und].ce).sum():.1f} ({np.percentile(bs, 2.5):.1f},{np.percentile(bs, 97.5):.1f})')
t_ = '\n'.join(O); open('results/identifiability_mcs_ensemble.txt', 'w').write(t_ + '\n'); print(t_)
