"""Section W: the stratified margin rule on the real JWST and Tier-1 labels of C+B-mlp, against no decline and the flat rule. Writes results/round3/round3_margin_stratified_real.txt."""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = pd.read_csv('results/round3/real_C+B-mlp.csv'); A = A[A.label.notna()].copy(); A['y'] = A.label == 'present'; A['call'] = A.prob >= A.threshold; A['m'] = -np.abs(A.prob - .5); A['u'] = A.planet.str.replace('-alt', '').str.replace('-hu', '').str.replace('-hm', ''); L = []
def bal(x): return .5 * (x[x.y].credit.mean() + x[~x.y].credit.mean())
def flat(g): return pd.concat([x[x.m <= x.m.quantile(.7)] for _, x in g.groupby('molecule')]) if False else pd.concat([x[x.m >= x.m.quantile(.3)] for _, x in g.groupby('molecule')])
def strat(g): return pd.concat([x[x.m >= x.m.quantile(.3)] for _, x in g.groupby(['molecule', 'call'])])
for p, g in A.groupby('precision'):
    r = np.random.default_rng(406); us = sorted(set(g.u)); G = {u: x for u, x in g.groupby('u')}; o = []
    for _ in range(2000):
        q = pd.concat([G[u] for u in r.choice(us, len(us))])
        if q.y.nunique() == 2: o.append(bal(strat(q)) - bal(q))
    f, s = flat(g), strat(g)
    L.append(f"{p:<7} n={len(g)} none balanced {bal(g):.3f} right {g.credit.sum():.1f}; flat n={len(f)} {bal(f):.3f}; stratified n={len(s)} {bal(s):.3f} right {s.credit.sum():.1f}; stratified minus none {bal(s) - bal(g):+.3f} ({np.percentile(o, 2.5):+.3f},{np.percentile(o, 97.5):+.3f})")
t = '\n'.join(L); open('results/round3/round3_margin_stratified_real.txt', 'w').write(t + '\n'); print(t)
