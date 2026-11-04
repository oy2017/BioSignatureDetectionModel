"""The ensemble against the routing method and the published recipe on real labels outside the panel the routing method was built on, and the labels a confirmation would need. Writes results/round3/round3_real_new.txt."""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = lambda s: s.replace('-alt', '').replace('-hu', '').replace('-hm', '')
C = ['batch', 'planet', 'precision', 'molecule', 'design', 'label', 'credit']
P = pd.read_csv('results/jwst_panel/panel_tier1.csv').assign(batch='panel'); F = pd.read_csv('results/jwst_panel_fourth/fourth_batch.csv').assign(batch='fourth'); I = pd.read_csv('results/independent_test/independent_test.csv').assign(batch='independent', precision='JWST'); N = pd.read_csv('results/round3/real_C+B-mlp.csv')
A = pd.concat([x[x.label.notna()][C] for x in (P, F, I)] + [N[C]]); A['u'] = A.planet.map(BASE); rng = np.random.default_rng(407); L = []
def diff(q, d):
    v = q[q.design.isin(['C+B-mlp', d])].pivot_table(index=['u', 'planet', 'molecule'], columns='design', values='credit').dropna(); x = (v['C+B-mlp'] - v[d]).groupby('u').sum().to_numpy(); b = x[rng.integers(0, len(x), (20000, len(x)))].sum(1)
    return len(v), v['C+B-mlp'].sum(), v[d].sum(), x.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(x), b.std()
for p, bs in (('JWST', (('outside the panel', ('fourth', 'independent')), ('all', ('panel', 'fourth', 'independent')))), ('Tier-1', (('outside the panel', ('fourth',)), ('all', ('panel', 'fourth'))))):
    for tag, b in bs:
        q = A[(A.precision == p) & A.batch.isin(b)]
        for d in ('the method', 'published'):
            n, a, c, g, lo, hi, np_, sd = diff(q, d)
            need = n * (1.96 * sd / g) ** 2 if g > 0 else float('nan')
            L.append(f'{p:<7}{tag:<18}n={n:<4}planets={np_:<3}{d:<11}ensemble {a:5.1f} this {c:5.1f} {g:+5.1f} ({lo:+.1f},{hi:+.1f}); labels needed for an interval just above zero if this gain per label is real: {need:.0f}')
t = '\n'.join(L); open('results/round3/round3_real_new.txt', 'w').write(t + '\n'); print(t)
