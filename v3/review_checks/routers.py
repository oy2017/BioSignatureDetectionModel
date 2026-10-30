"""Check from the referee pass of 2026-10-30. Output saved as results/review_checks/routers.txt."""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
V3='/Users/fanyang/BioSignatureDetectionModel/v3'; os.chdir(V3); sys.path.insert(0,V3)
import numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
import alfnoor_diagnostic as D, router_real as RR, jwst_panel as JP
from muting_router import RETRAINED, verdict
t0=time.time()
cases,Pte,e,sig=D.test_cases(); sc=D.Scorer(e); S0,P0,_=sc.scores(cases['clean'])
print('loaded',time.time()-t0,flush=True)
q=lambda s: np.quantile(s[np.isfinite(s)],1-D.DECLINE)
thr={r:q(S0[(r,None)]) for r in ('knn','mahalanobis','pca_recon','optical')}
thr_cell={r:{(m,k):q(S0[(r,(m,k))]) for m in D.MOLS for k in D.LEARNERS} for r in ('margin','ensemble','disagree')}
print({k:round(v,2) for k,v in thr.items()})
F=pd.read_csv('results/jwst_panel/amplitude_flag.csv').set_index('planet')
up=F.ratio.where(np.isfinite(F.ratio))>F.hi_clear_max; low=F.ratio<F.lo_own_p01
rows=[]; dist={}
for pl,d in RR.PANEL.items():
    labs=d['labels']; cov=pd.read_csv(d['coverage']); x=cov.depth.to_numpy().copy(); filled=cov.filled.to_numpy(bool); x[filled]=np.nanmean(x[~filled])
    S,_,_=sc.scores(x[None,:]); V=pd.read_csv(d['verdicts']); V=V[np.isclose(V.threshold.astype(float),1e-4)]
    Vh=pd.read_csv(os.path.join(RETRAINED['haze'],f'{pl}_verdicts.csv')); Vh=Vh[np.isclose(Vh.threshold.astype(float),1e-4)]
    pres={(v,m,k):int(V[(V.variant==v)&(V.molecule==m)&(V.learner==k)].present.iloc[0]) for v in ('all','airs_only') for m in D.MOLS for k in D.LEARNERS}
    dist[pl]={r:float(S[(r,None)][0]) for r in thr}
    for m in D.MOLS:
        lab=labs.get(m)
        hv=verdict(Vh,'all',m)
        for router in ('none','knn','mahalanobis','pca_recon','optical','margin','ensemble','disagree','shape','knn+optical','knn|disagree','airs'):
            n=0;nf=0
            for k in D.LEARNERS:
                fl={'none':False,'airs':True}.get(router)
                if fl is None:
                    if router in thr: fl=bool(S[(router,None)][0]>thr[router])
                    elif router in thr_cell: fl=bool(S[(router,(m,k))][0]>thr_cell[router][(m,k)])
                    elif router=='shape': fl=bool(S[('optical',None)][0]>thr['optical']) or bool(S[('disagree',(m,k))][0]>thr_cell['disagree'][(m,k)])
                    elif router=='knn+optical': fl=bool(S[('optical',None)][0]>thr['optical']) or bool(S[('knn',None)][0]>thr['knn'])
                    elif router=='knn|disagree': fl=bool(S[('knn',None)][0]>thr['knn']) or bool(S[('disagree',(m,k))][0]>thr_cell['disagree'][(m,k)])
                nf+=fl; n+=pres[('airs_only' if fl else 'all',m,k)]
            v=JP.P if n>=3 else (JP.A if n<=1 else 'split')
            # with the amplitude flag in front
            va=v
            if bool(up.get(pl,False)): va=hv
            if bool(low.get(pl,False)): va='split'
            rows.append(dict(planet=pl,molecule=m,router=router,verdict=v,verdict_amp=va,label=lab))
df=pd.DataFrame(rows); df.to_csv('results/review_checks/routers_verdicts.csv',index=False)
lab=df[df.label.notna()].copy()
def credit(v,l): return 0.5 if v=='split' else float(v==l)
lab['c']=[credit(v,l) for v,l in zip(lab.verdict,lab.label)]; lab['ca']=[credit(v,l) for v,l in zip(lab.verdict_amp,lab.label)]
meth=[p for p,d in RR.PANEL.items() if d['labels'].get('CH4')==JP.P]
out=[]
for r,g in lab.groupby('router'):
    ch=g[(g.molecule=='CH4')&g.planet.isin(meth)]
    cha=ch.verdict_amp.value_counts()
    out.append(dict(router=r,score=g.c.sum(),right=(g.c==1).sum(),wrong=(g.c==0).sum(),split=(g.c==.5).sum(),CH4p=(ch.verdict=='present').sum(),score_with_amp=g.ca.sum(),wrong_amp=(g.ca==0).sum(),CH4p_amp=cha.get('present',0),present_credit=g[g.label=='present'].c.sum(),absent_credit=g[g.label=='absent'].c.sum()))
print(pd.DataFrame(out).sort_values('score',ascending=False).to_string())
print(pd.DataFrame(dist).T.round(2).to_string())
