"""Real labels and precision of the present calls for C+B-mlp (Section 3.7, Table 20). Output: results/round3/round3_real.txt"""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
C=['batch','planet','precision','molecule','design','label','credit']
P=pd.read_csv('results/jwst_panel/panel_tier1.csv').assign(batch='panel');F=pd.read_csv('results/jwst_panel_fourth/fourth_batch.csv').assign(batch='fourth');I=pd.read_csv('results/independent_test/independent_test.csv').assign(batch='independent',precision='JWST');N=pd.read_csv('results/round3/real_C+B-mlp.csv')
A=pd.concat([x[x.label.notna()][C] for x in (P,F,I)]+[N[C]]); A['u']=A.planet.map(BASE); rng=np.random.default_rng(405); L=['== real labels: ensemble against each design; difference with a 95 % bootstrap over planets']
for p,bs in (('JWST',('panel','fourth','independent','all')),('Tier-1',('panel','fourth','all'))):
    for b in bs:
        q=A[(A.precision==p)&((A.batch==b)|(b=='all'))]
        for d in ('published','AIRS-only','haze+deck-retrained','the method'):
            v=q[q.design.isin(['C+B-mlp',d])].pivot_table(index=['u','planet','molecule'],columns='design',values='credit').dropna(); x=(v['C+B-mlp']-v[d]).groupby('u').sum().to_numpy(); bb=x[rng.integers(0,len(x),(20000,len(x)))].sum(1)
            L.append(f'  {p:<7}{b:<12}n={len(v):<4}{d:<22}ensemble {v["C+B-mlp"].sum():5.1f}  this {v[d].sum():5.1f}  {x.sum():+5.1f} ({np.percentile(bb,2.5):+.1f},{np.percentile(bb,97.5):+.1f})')
L.append('== grids: share of present calls that are truly present (precision), at the grid prevalence and at a stated prevalence of 0.1')
for c in ('rainout','local','crossfield'):
    Y=pd.read_csv(f'results/goyal_test/truth_{c}.csv').melt(id_vars=['planet','row'],value_vars=[f'v_{m}' for m in ('CH4','H2O','CO2','NH3')],var_name='molecule',value_name='v'); Y['molecule']=Y.molecule.str[2:]
    V=pd.concat([pd.read_csv(f'results/goyal_test/votes_frozen_{c}.csv.gz').query('design=="the method"'),pd.read_csv(f'results/round3/votes_C+B-mlp_{c}.csv.gz').drop(columns='prob')]).merge(Y,on=['planet','row','molecule']); V['y']=V.v>-4
    for (d,p),g in V.groupby(['design','precision']):
        tp,fp=g[g.y].present.mean(),g[~g.y].present.mean(); L.append(f'  {c:<11}{p:<7}{d:<12}prevalence {g.y.mean():.2f}  precision {g[g.present==1].y.mean():.3f}  at prevalence 0.1 {0.1*tp/(0.1*tp+0.9*fp):.3f}')
t='\n'.join(L); open('results/round3/round3_real.txt','w').write(t+'\n'); print(t)
