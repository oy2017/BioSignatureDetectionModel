"""Round two on the rainout grid, on the masks before the audit's corrections (REVISION_EXPECTATIONS.md, sections L and M). Output: results/goyal_test/broad_rainout.txt"""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
V=pd.concat([pd.read_csv(f'results/goyal_test/votes_{k}_rainout.csv.gz') for k in ('frozen','matched','broad')]); Y=pd.read_csv('results/goyal_test/truth_rainout.csv')
Yl=Y.melt(id_vars=['planet','row'],value_vars=[f'v_{m}' for m in ('CH4','H2O','CO2','NH3')],var_name='molecule',value_name='v'); Yl['molecule']=Yl.molecule.str[2:]
k=['batch','planet','precision','molecule','row']; h=V[V.design=='haze+deck-retrained'].set_index(k); b=V[V.design=='B+bal'].set_index(k)
s=(b.present-b.absent)+(h.present-h.absent).reindex(b.index); E=b.copy(); E['design']='B+bal+ens'; E['present']=(s>0).astype(float); E['absent']=(s<0).astype(float); V=pd.concat([V,E.reset_index()])
A=V.merge(Yl,on=['planet','row','molecule']); A['y']=A.v>-4; A['credit']=np.where(A.y,A.present,A.absent)+.5*(1-A.present-A.absent); L=[]
for p in ('JWST','Tier-1'):
    for q,t in ((A[A.precision==p],'all'),(A[(A.precision==p)&(A.molecule=='CO2')],'CO2')):
        L.append(f'== {p}, {t}')
        for d,g in q.groupby('design'): L.append(f'  {d:<22} balanced {.5*(g[g.y].credit.mean()+g[~g.y].credit.mean()):.3f}  present {g[g.y].credit.mean():.3f}  absent {g[~g.y].credit.mean():.3f}  false-present {g[~g.y].present.mean():.3f}')
t='\n'.join(L); open('results/goyal_test/broad_rainout.txt','w').write(t+'\n'); print(t)
