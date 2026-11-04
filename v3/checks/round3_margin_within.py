"""Decline rules with the 70 % kept within each gas, so that the mix of gases cannot supply the gain (REVISION_EXPECTATIONS.md, section V). Output: results/round3/round3_margin_within.txt"""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOLS=['CH4','H2O','CO2','NH3']; RU=['margin','ensemble','knn','mahalanobis','pca_recon']; BASE=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
E=pd.read_csv('results/round3/round3_rules_rows.csv.gz'); E['absent']=1-E.present; L=[]
def keep(g,r): return pd.concat([x[x[r]<=x[r].quantile(.7)] for _,x in g.groupby('molecule')])
def bal(x): return .5*(x[x.y].c.mean()+x[~x.y].c.mean())
def meanbal(x): return np.mean([bal(q) for _,q in x.groupby('molecule') if q.y.nunique()==2])
for c in ('rainout','local','crossfield'):
    Y=pd.read_csv(f'results/goyal_test/truth_{c}.csv').melt(id_vars=['planet','row'],value_vars=[f'v_{m}' for m in MOLS],var_name='molecule',value_name='v'); Y['molecule']=Y.molecule.str[2:]
    C=pd.read_csv(f'results/round3/rule_rows_{c}.csv.gz').merge(Y,on=['planet','row','molecule']); e=E[E.cond==c].merge(Y,on=['planet','row','molecule'])
    for q in (C,e): q['y']=q.v>-4; q['c']=np.where(q.y,q.present,q.absent)+.5*(1-q.present-q.absent); q['u']=q.planet.map(BASE)
    for p in ('JWST','Tier-1'):
        a,b=e[e.precision==p],C[C.precision==p]; s=f'{c:<11}{p:<7}'
        ia={r:meanbal(keep(a,r))-meanbal(a) for r in RU}; ib={r:meanbal(keep(b,r))-meanbal(b) for r in RU}; ba=max(ia,key=ia.get); bb=max(ib,key=ib.get)
        ga={u:x for u,x in a.groupby('u')}; gb={u:x for u,x in b.groupby('u')}; us=sorted(ga); rng=np.random.default_rng(405); o=[]
        for _ in range(200):
            pk=rng.choice(us,len(us)); xa=pd.concat([ga[u] for u in pk]); xb=pd.concat([gb[u] for u in pk]); o.append(meanbal(keep(xa,ba))-meanbal(xa)-(meanbal(keep(xb,bb))-meanbal(xb)))
        L.append(s+f'mean over gases: ensemble none {meanbal(a):.3f}; adds '+' '.join(f'{r} {ia[r]:+.3f}' for r in RU)+f' | classifier none {meanbal(b):.3f}; adds '+' '.join(f'{r} {ib[r]:+.3f}' for r in RU)+f' | best {ba} minus best {bb} {ia[ba]-ib[bb]:+.3f} ({np.percentile(o,2.5):+.3f},{np.percentile(o,97.5):+.3f})')
t='\n'.join(L); open('results/round3/round3_margin_within.txt','w').write(t+'\n'); print(t)
