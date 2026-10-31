"""Stability across seeds and the band-removed control for C+B-mlp (REVISION_EXPECTATIONS.md, section N). Output: results/round3/round3_checks.txt"""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
def truth(c):
    Y=pd.read_csv(f'results/goyal_test/truth_{c}.csv'); Y=Y.melt(id_vars=['planet','row'],value_vars=[f'v_{m}' for m in ('CH4','H2O','CO2','NH3')],var_name='molecule',value_name='v'); Y['molecule']=Y.molecule.str[2:]; return Y
def score(V,Y):
    A=V.merge(Y,on=['planet','row','molecule']); A['y']=A.v>-4; A['credit']=np.where(A.y,A.present,A.absent)+.5*(1-A.present-A.absent); return A
def bal(g): return .5*(g[g.y].credit.mean()+g[~g.y].credit.mean())
L=['== stability: balanced credit, seeds 0, 1, 2 (false-present)']
for c in ('rainout','local','crossfield'):
    Y=truth(c)
    for p in ('JWST','Tier-1'):
        r=[]
        for s in ('','seed1/','seed2/'):
            g=score(pd.read_csv(f'results/round3/{s}votes_C+B-mlp_{c}.csv.gz').query('precision==@p'),Y); r.append(f'{bal(g):.3f} ({g[~g.y].present.mean():.3f})')
        L.append(f'  {c:<11}{p:<7}'+'  '.join(r))
L.append('== stability: real labels, score (present, absent)')
for p,bs in (('JWST',('panel','fourth','independent')),('Tier-1',('panel','fourth'))):
    for b in bs+('all',):
        r=[]
        for s in ('','seed1/','seed2/'):
            d=pd.read_csv(f'results/round3/{s}real_C+B-mlp.csv'); d=d[(d.precision==p)&((d.batch==b)|(b=='all'))]; r.append(f'{d.credit.sum():5.1f} ({d[d.label=="present"].credit.sum():.1f}, {d[d.label=="absent"].credit.sum():.1f})')
        L.append(f'  {p:<7}{b:<12}n={len(d):<4}'+'  '.join(r))
L.append('== bands removed, JWST precision: balanced credit (present, false-present); full design; frozen method')
for c in ('rainout','local','crossfield'):
    Y=truth(c); B=score(pd.read_csv(f'results/round3/band_control_votes_{c}.csv.gz'),Y); F=score(pd.read_csv(f'results/round3/votes_C+B-mlp_{c}.csv.gz').query('precision=="JWST"'),Y); O=score(pd.read_csv(f'results/goyal_test/votes_frozen_{c}.csv.gz').query('precision=="JWST" and design=="the method"'),Y); L.append(f' {c}')
    for m in ('all','CH4','H2O','CO2','NH3'):
        k=(lambda q:q if m=='all' else q[q.molecule==m]); b,f,o=k(B),k(F),k(O); L.append(f'  {m:<4} removed {bal(b):.3f} ({b[b.y].credit.mean():.3f}, {b[~b.y].present.mean():.3f})  full {bal(f):.3f} ({f[f.y].credit.mean():.3f}, {f[~f.y].present.mean():.3f})  method {bal(o):.3f}  full minus removed {bal(f)-bal(b):+.3f}')
R=pd.read_csv('results/round3/band_control_real.csv'); D=pd.read_csv('results/round3/real_C+B-mlp.csv').query('precision=="JWST"')
M=D.merge(R,on=['batch','planet','molecule','label'],suffixes=('','_rm')); L.append(f'== real labels, JWST precision, bands removed: {len(M)} labels')
for lab in ('present','absent'):
    g=M[M.label==lab]; L.append(f'  {lab}: full {g.credit.sum():.1f}, removed {g.credit_rm.sum():.1f} of {len(g)}; verdict lost {int(((g.credit==1)&(g.credit_rm==0)).sum())}, gained {int(((g.credit==0)&(g.credit_rm==1)).sum())}')
for m,g in M.groupby('molecule'): L.append(f'  {m}: present full {g[g.label=="present"].credit.sum():.1f}, removed {g[g.label=="present"].credit_rm.sum():.1f} of {(g.label=="present").sum()}; absent full {g[g.label=="absent"].credit.sum():.1f}, removed {g[g.label=="absent"].credit_rm.sum():.1f} of {(g.label=="absent").sum()}')
t='\n'.join(L); open('results/round3/round3_checks.txt','w').write(t+'\n'); print(t)
