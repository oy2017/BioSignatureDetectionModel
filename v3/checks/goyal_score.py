"""Scores on the Goyal grid (REVISION_EXPECTATIONS.md, section J). Output: results/goyal_test/goyal_<cond>.txt"""
import os, sys, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); tag=sys.argv[1] if len(sys.argv)>1 else 'rainout'
BASE=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
import glob; V=pd.concat([pd.read_csv(f) for f in [f'results/goyal_test/votes_{k}_{tag}.csv.gz' for k in ('frozen','matched')]+sorted(glob.glob(f'results/round3/votes_*_{tag}.csv.gz'))]).drop(columns='prob',errors='ignore'); Y=pd.read_csv(f'results/goyal_test/truth_{tag}.csv')
Yl=Y.melt(id_vars=['planet','row','T','g','met','co','haze','cloud'],value_vars=[f'v_{m}' for m in ('CH4','H2O','CO2','NH3')],var_name='molecule',value_name='v'); Yl['molecule']=Yl.molecule.str[2:]
A=V.merge(Yl,on=['planet','row','molecule']); A['y']=A.v>-4; A['credit']=np.where(A.y,A.present,A.absent)+.5*(1-A.present-A.absent); A['u']=A.planet.map(BASE)
DES=['published','AIRS-only','shape flag','haze+deck-retrained','the method','matched+amp']+sorted(set(V.design)-{'published','AIRS-only','shape flag','haze+deck-retrained','the method','matched+amp'}); rng=np.random.default_rng(405)
def per(q):
    g=q.groupby(['u','design','y']).credit.mean().unstack('y'); return g
def bal(q,idx=None):
    # balanced credit with planets as units
    s=q.groupby(['u','design','y']).credit.agg(['sum','count']).reset_index(); return s
L=[]
def table(q,title):
    L.append(f'== {title}: {q[q.design=="the method"].shape[0]} labels, {q[q.design=="the method"].y.mean()*100:.0f} % present')
    s=q.groupby(['design','u','y']).credit.agg(['sum','count']); us=sorted(q.u.unique()); arr={}
    for d in DES:
        p=s.loc[d].unstack('y').reindex(us).fillna(0); arr[d]=(p['sum'][True].to_numpy(),p['count'][True].to_numpy(),p['sum'][False].to_numpy(),p['count'][False].to_numpy())
    ix=rng.integers(0,len(us),(4000,len(us)))
    def B(a,i=None):
        i=np.arange(len(us)) if i is None else i; return .5*(a[0][i].sum(-1)/a[1][i].sum(-1)+a[2][i].sum(-1)/a[3][i].sum(-1))
    for d in DES:
        a=arr[d]; rec=a[0].sum()/a[1].sum(); spec=a[2].sum()/a[3].sum(); dif=B(a,ix)-B(arr['the method'],ix); dm=B(arr['matched+amp'],ix)-B(a,ix); g=q[q.design==d]; fp=g[~g.y].present.mean()
        L.append(f'  {d:<22} balanced {B(a):.3f}  present {rec:.3f}  absent {spec:.3f}  false-present {fp:.3f}  minus method {B(a)-B(arr["the method"]):+.3f} ({np.percentile(dif,2.5):+.3f},{np.percentile(dif,97.5):+.3f})  matched+amp minus this {B(arr["matched+amp"])-B(a):+.3f} ({np.percentile(dm,2.5):+.3f},{np.percentile(dm,97.5):+.3f})')
for prec in ('JWST','Tier-1'):
    q=A[A.precision==prec]; table(q,f'{prec}, all'); table(q[np.abs(q.v+4)>.5],f'{prec}, labels beyond 0.5 dex of the threshold')
    for m in ('CH4','H2O','CO2','NH3'): table(q[q.molecule==m],f'{prec}, {m}')
    table(q[q.batch=='panel'],f'{prec}, panel masks'); table(q[q.batch=='fourth'],f'{prec}, fourth-batch masks')
t='\n'.join(L); open(f'results/goyal_test/goyal_{tag}.txt','w').write(t+'\n'); print(t)
