"""Check from the referee pass of 2026-10-30. Output saved as results/review_checks/flags_vs_classes.txt."""
import os, sys, warnings
warnings.filterwarnings('ignore')
V3='/Users/fanyang/BioSignatureDetectionModel/v3'; os.chdir(V3); sys.path.insert(0,V3)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import alfnoor_diagnostic as D, alfnoor_trust as AT, muting_router as MR, alfnoor_jwst as AJ, joblib
from common import DATA
cases,Pte,e,sig=D.test_cases(); sc=D.Scorer(e)
C=pd.read_csv('results/identifiability_calibration.csv').drop_duplicates('name').set_index('name').reindex(Pte.name.to_numpy())
S0,P0,_=sc.scores(cases['clean'])
print('== AUROC of each flag for recoverable (p_true>0.95) vs irrecoverable errors, clean; mean over learners and molecules')
rows=[]
for case in ('clean','haze3e7'):
    S,P,_=sc.scores(cases[case]) if case!='clean' else (S0,P0,None)
    for rule in ('optical','disagree','margin','ensemble','knn','mahalanobis'):
        vals=[]
        for m in D.MOLS:
            y=(Pte[f'atm {m}'].to_numpy()>-4); pt=C[f'{m}_p_true'].to_numpy(float)
            for k in D.LEARNERS:
                s=sc.get(S,rule,m,k); w=(P['all'][(m,k)]>=0.5)!=y; ok=np.isfinite(pt)&np.isfinite(s)
                rec=(pt>0.95)
                # score higher = flag fires; AUROC for 'flag ranks recoverable errors above irrecoverable'
                a=AT.auroc(rec[w&ok],s[w&ok])
                vals.append(a)
        rows.append(dict(case=case,rule=rule,auroc_recoverable_vs_irrecoverable=np.nanmean(vals)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# Share of errors each flag marks, by class, on haze with clean-posterior classes
print('== share flagged among correct labels vs errors, haze3e7 (optical)')
S,P,_=sc.scores(cases['haze3e7']); o=S[('optical',None)]; thr=np.quantile(S0[('optical',None)],0.9)
print('optical fires on %.1f %% of hazed planets'%(100*(o>thr).mean()))
# Majority-of-four, half-credit for split, in simulation: like-for-like with the panel scoring
_,PQ=MR.scale_height_proxy(); T_HI,T_LO=PQ['clear_max'],PQ['pop_p01']; print('T',T_HI,T_LO)
mod=Pte['modulation_5H'].to_numpy(float)*1e6; noise2=np.mean((sig*1e6)**2,axis=1)
ret=joblib.load(os.path.join(DATA,'alfnoor_jwst','learners_all_haze.joblib'))
def ratio(X):
    v=np.var(X*1e6,axis=1)-noise2; r=np.full(len(X),np.inf); pos=v>0; r[pos]=mod[pos]/np.sqrt(v[pos]); return r
opt=lambda X: np.abs(sc._optical(AT.norm(X))-sc.opt_mu)/sc.opt_sd
def dis(P,m,k):
    p=P['all'][(m,k)]; return np.maximum(np.abs(p-P['no_optical'][(m,k)]),np.abs(p-P['airs_only'][(m,k)]))
thr_opt=np.quantile(opt(cases['clean']),0.9); thr_dis={(m,k):np.quantile(dis(P0,m,k),0.9) for m in D.MOLS for k in D.LEARNERS}
out=[]
for case in ('clean','haze3e7','white_x2','white_x3'):
    X=cases[case]; P=P0 if case=='clean' else sc.probs(X); o=opt(X); PR={(m,k):AT.prob(ret['models'][('1e-4',m,k)],AJ.znorm(X,ret['cols'])) for m in D.MOLS for k in D.LEARNERS}
    r=ratio(X); up=np.isfinite(r)&(r>T_HI); low=r<T_LO
    res={'published':[], 'AIRS':[], 'haze':[], 'method':[]}
    for m in D.MOLS:
        y=(Pte[f'atm {m}'].to_numpy()>-4)
        V={'published':[], 'AIRS':[], 'haze':[], 'method':[]}
        for k in D.LEARNERS:
            p=P['all'][(m,k)]; pa=P['airs_only'][(m,k)]; pr=PR[(m,k)]; shape=(o>thr_opt)|(dis(P,m,k)>thr_dis[(m,k)]); ps=np.where(shape,pa,p); pp=np.where(up,pr,ps)
            V['published'].append(p>=.5); V['AIRS'].append(pa>=.5); V['haze'].append(pr>=.5); V['method'].append(pp>=.5)
        for name,v in V.items():
            n=np.sum(v,axis=0)
            verdict=np.where(n>=3,1.0,np.where(n<=1,0.0,0.5))
            cred=np.where(verdict==0.5,0.5,(verdict==y.astype(float)).astype(float))
            if name=='method': cred=np.where(low,0.5,cred)
            res[name].append(cred.mean())
    out.append(dict(case=case,**{k:100*np.mean(v) for k,v in res.items()}))
print('== simulation scored as the panel is: majority of four, split or withheld one half (%), mean over molecules')
print(pd.DataFrame(out).round(2).to_string(index=False))
# Spearman of simulated accuracy vs panel score across seven designs
des=['reimpl','noopt','airs','shape','haze','hazedeck','method']
pj=[45.5,50.0,52.5,51.5,50.5,52.0,54.0]; pt=[41.7,43.1,43.4,43.3,43.1,43.5,44.4]
sim={'clean':[81.2,77.4,73.8,76.6,80.1,80.2,76.4],'haze':[63.5,71.8,73.3,73.3,73.2,73.6,72.2],'confined':[67.1,74.9,73.6,72.7,73.9,75.4,72.9],'deck':[71.4,72.3,72.1,71.8,73.3,74.6,71.6],'spots':[70.4,74.2,74.0,73.9,75.7,76.1,73.2],'noise2':[71.9,69.6,68.3,70.2,71.7,71.7,69.8]}
for k,v in sim.items():
    print(k,'Spearman vs JWST panel %.2f, vs Tier-1 panel %.2f'%(spearmanr(v,pj)[0],spearmanr(v,pt)[0]))
