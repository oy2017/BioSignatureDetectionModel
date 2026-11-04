"""Section W of REVISION_EXPECTATIONS.md: decline rules with the 70 % kept within each gas and each call. Post hoc on seen grids. Writes results/round3/round3_margin_stratified.txt."""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOLS = ['CH4', 'H2O', 'CO2', 'NH3']; RU = ['margin', 'ensemble', 'knn', 'mahalanobis', 'pca_recon']
BASE = lambda s: s.replace('-alt', '')
E = pd.read_csv('results/round3/round3_rules_rows.csv.gz'); L = []
def flat(g, r): return pd.concat([x[x[r] <= x[r].quantile(.7)] for _, x in g.groupby('molecule')])
def strat(g, r): return pd.concat([x[x[r] <= x[r].quantile(.7)] for _, x in g.groupby(['molecule', 'call'])])
def bal(x): return .5 * (x[x.y].c.mean() + x[~x.y].c.mean())
def mb(x): return np.mean([bal(q) for _, q in x.groupby('molecule') if q.y.nunique() == 2])
for c in ('rainout', 'local', 'crossfield'):
    Y = pd.read_csv(f'results/goyal_test/truth_{c}.csv').melt(id_vars=['planet', 'row'], value_vars=[f'v_{m}' for m in MOLS], var_name='molecule', value_name='v'); Y['molecule'] = Y.molecule.str[2:]
    C = pd.read_csv(f'results/round3/rule_rows_{c}.csv.gz').merge(Y, on=['planet', 'row', 'molecule']); e = E[E.cond == c].merge(Y, on=['planet', 'row', 'molecule'])
    for q in (C, e):
        q['y'] = q.v > -4; q['call'] = q.present >= .5; q['c'] = np.where(q.y, q.present, 1 - q.present); q['u'] = q.planet.map(BASE)
    for p in ('JWST', 'Tier-1'):
        a, b = e[e.precision == p], C[C.precision == p]
        fa = mb(flat(a, 'margin')) - mb(a); sa = mb(strat(a, 'margin')) - mb(a); ib = {r: mb(strat(b, r)) - mb(b) for r in RU}; ba = max(ib, key=ib.get)
        ga = {u: x for u, x in a.groupby('u')}; gb = {u: x for u, x in b.groupby('u')}; us = sorted(ga); rng = np.random.default_rng(406); o = []
        for _ in range(200):
            pk = rng.choice(us, len(us)); xa = pd.concat([ga[u] for u in pk]); xb = pd.concat([gb[u] for u in pk])
            o.append(mb(strat(xa, 'margin')) - mb(xa) - (mb(strat(xb, ba)) - mb(xb)))
        k = strat(a, 'margin'); per = ' '.join(f"{m} {bal(a[a.molecule == m]):.3f}->{bal(k[k.molecule == m]):.3f}" for m in MOLS)
        L.append(f"{c:<11}{p:<7} ensemble none {mb(a):.3f} flat {fa:+.3f} stratified {sa:+.3f} kept {mb(a) + sa:.3f}; classifier none {mb(b):.3f} best stratified {ba} {ib[ba]:+.3f} kept {mb(b) + ib[ba]:.3f}; difference {sa - ib[ba]:+.3f} ({np.percentile(o, 2.5):+.3f},{np.percentile(o, 97.5):+.3f}) | {per}")
t = '\n'.join(L); open('results/round3/round3_margin_stratified.txt', 'w').write(t + '\n'); print(t)
