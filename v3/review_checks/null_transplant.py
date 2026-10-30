"""Check from the referee pass of 2026-10-30. Output saved as results/review_checks/null_transplant.txt."""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
V3='/Users/fanyang/BioSignatureDetectionModel/v3'; os.chdir(V3); sys.path.insert(0,V3)
import numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
import alfnoor_diagnostic as D, jwst_muting_census as C, router_real as RR, jwst_to_ariel as J
edges=C.edges; cen=0.5*(edges[1:]+edges[:-1]); sc=D.Scorer(edges); M=C.M; LE=D.LEARNERS
print(M.columns.tolist()[:40])
base=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
def twin(pl,ab,mass=None):
    r=M[M.name==base(pl)].iloc[0].to_dict(); r.update({"atm top_pressure":1e-4,"atm CO":-15.0,"atm O3":-15.0,**{f"atm {k}":v for k,v in ab.items()}})
    if mass is not None: r['p_mass']=mass
    return C.render(r), r
def load(pl):
    d=RR.PANEL[pl]; cov=pd.read_csv(d['coverage']); x=cov.depth.to_numpy().copy(); filled=cov.filled.to_numpy(bool); meas=~filled; x[filled]=np.nanmean(x[meas]); return x,meas,cov.err.to_numpy(),filled
def votes(X,m,var='all'):
    P=sc.probs(X); return [int(sum(P[var][(m,k)][i]>=0.5 for k in LE)) for i in range(len(X))]
print('NULL TRANSPLANT: CH4 votes (of 4) with bins below 1.95 um replaced by twin: real | twin with CH4 as published | twin with CH4=-7 (methane-free) ')
res=[]
for pl in ['TOI-270d','TOI-270d-hm','K2-18b','K2-18b-alt','K2-18b-hu','TOI-1231b','LTT3780c']:
    ab=C.TWINS[pl]; x,meas,err,filled=load(pl)
    t,_=twin(pl,ab); ab0=dict(ab); ab0['CH4']=-7.0; t0,_=twin(pl,ab0)
    air=meas&(cen>1.95)
    al=lambda T: T-np.mean(T[air])+np.mean(x[air])
    v=[x.copy(),x.copy(),x.copy()]
    t1=al(t); t2=al(t0); blk=cen<1.95
    v[1][blk]=t1[blk]; v[2][blk]=t2[blk]
    # also a flat block: bins below 1.95 replaced by the AIRS mean (no feature at all)
    fl=x.copy(); fl[blk]=np.mean(x[air])
    out=votes(np.vstack(v+[fl]),'CH4')
    res.append((pl,*out)); print(pl,out,flush=True)
print('repaired (>=3): published',sum(r[2]>=3 for r in res),' methane-free',sum(r[3]>=3 for r in res),' flat',sum(r[4]>=3 for r in res))
print()
print('HIP 67522 b')
pl='HIP67522b'; ab=C.TWINS[pl]; x,meas,err,filled=load(pl); r=M[M.name==pl].iloc[0]
print({k:r[k] for k in M.columns if k in ('p_mass','p_radius','p_mass_err','modulation_5H','atm temperature','s radius','tier1_transits','mass_flag')})
t,rr=twin(pl,ab)
mu=np.mean(t[meas]); s_corr=np.sqrt(max(np.std(x[meas])**2-np.nanmean(err[meas]**2),0)); s_t=np.std(t[meas]); f=s_corr/s_t; print('scale factor',round(f,2),'real corrected ppm',round(s_corr*1e6),'twin ppm',round(s_t*1e6))
def thr(b):
    y=b.copy(); y[filled]=np.mean(y[meas]); return y
base_t=thr(t); scaled=thr(mu+(t-mu)*f)
rng=np.random.default_rng(405); e=np.where(meas,np.nan_to_num(err),0.0)
for name,y in (('real',x),('twin catalogue mass, noise-free',base_t),('twin x scale, noise-free',scaled)):
    P=sc.probs(y[None,:]); print(name,{v:{mm:int(sum(P[v][(mm,k)][0]>=0.5 for k in LE)) for mm in D.MOLS} for v in ('all','airs_only')})
for name,y in (('twin catalogue mass, 200 draws real errors',base_t),('twin x scale, 200 draws real errors',scaled)):
    X=y[None,:]+rng.normal(0,1,(200,len(y)))*e; P=sc.probs(X)
    for v in ('all','airs_only'):
        n=sum((P[v][('CO2',k)]>=0.5).astype(int) for k in LE); print(name,v,'CO2 majority freq %.2f split %.2f'%((n>=3).mean(),(n==2).mean()))
# true mass twin (mass in MCS units?)
print('catalogue p_mass value',r.get('p_mass'))
