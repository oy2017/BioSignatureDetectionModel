"""Does C+B-mlp's probability mark unreliable verdicts (REVISION_EXPECTATIONS.md, section P)? Output: results/round3/round3_confidence.txt"""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score as auc
L=[]; T=pd.read_csv('results/round3/thresholds_C+B-mlp.csv')
Tl=T.melt(id_vars=['planet','precision'],var_name='molecule',value_name='t')
def report(A,tag):
    A=A.dropna(subset=['t']); A['right']=A.present==A.y.astype(float); A['m']=np.abs(A.prob-A.t); k=A.m>=A.m.quantile(.3); r=auc(A.y,A.prob); e=auc(~A.right,-A.m)
    L.append(f'{tag:<24} n={len(A):<6} AUROC label {r:.3f}  AUROC errors by margin {e:.3f}  accuracy all {100*A.right.mean():.1f}, most confident 70 % {100*A[k].right.mean():.1f}')
for c in ('rainout','local','crossfield'):
    Y=pd.read_csv(f'results/goyal_test/truth_{c}.csv'); Y=Y.melt(id_vars=['planet','row'],value_vars=[f'v_{m}' for m in ('CH4','H2O','CO2','NH3')],var_name='molecule',value_name='v'); Y['molecule']=Y.molecule.str[2:]
    V=pd.read_csv(f'results/round3/votes_C+B-mlp_{c}.csv.gz').merge(Y,on=['planet','row','molecule']).merge(Tl,on=['planet','precision','molecule']); V['y']=V.v>-4
    for p in ('JWST','Tier-1'):
        report(V[V.precision==p].copy(),f'{c}, {p}')
        for m in ('CH4','H2O','CO2','NH3'): report(V[(V.precision==p)&(V.molecule==m)].copy(),f'   {m}')
R=pd.read_csv('results/round3/real_C+B-mlp.csv'); R['y']=R.label=='present'; R['t']=R.threshold
for p in ('JWST',):
    q=R[R.precision==p].copy(); q['present']=(q.prob>=q.t).astype(float); report(q,f'real, {p}')
t='\n'.join(L); open('results/round3/round3_confidence.txt','w').write(t+'\n'); print(t)
