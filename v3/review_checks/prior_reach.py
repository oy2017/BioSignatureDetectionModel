"""Check from the referee pass of 2026-10-30. Output saved as results/review_checks/prior_reach.txt."""
import os, sys, warnings
warnings.filterwarnings('ignore')
V3='/Users/fanyang/BioSignatureDetectionModel/v3'; os.chdir(V3); sys.path.insert(0,V3)
import numpy as np, pandas as pd
import jwst_muting_census as C, router_real as RR
edges=C.edges; cen=0.5*(edges[1:]+edges[:-1]); M=C.M
base=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
g=['HAT-P-26b','WASP-107b','WASP-39b','GJ3470b','WASP-166b','WASP-17b','WASP-80b','HD209458b','WASP-94Ab','HAT-P-12b']
rows=[]
for pl in g:
    d=RR.PANEL[pl]; cov=pd.read_csv(d['coverage']); x=cov.depth.to_numpy(); meas=~cov.filled.to_numpy(bool); err=cov.err.to_numpy()
    s=np.std(x[meas])*1e6; n=np.sqrt(np.nanmean(err[meas]**2))*1e6; real=np.sqrt(max(s**2-n**2,0))
    r=M[M.name==pl].iloc[0].to_dict(); r.update({"atm top_pressure":1e-4,"atm CO":-15.0,"atm O3":-15.0})
    out={}
    for tag,ab in (('pub',C.TWINS[pl]),('floor-7',{k:-7.0 for k in ('CH4','H2O','CO2','NH3')}),('h2o-5',{'CH4':-7.0,'H2O':-5.0,'CO2':-7.0,'NH3':-7.0}),('h2o-4',{'CH4':-7.0,'H2O':-4.0,'CO2':-7.0,'NH3':-7.0})):
        rr=dict(r); rr.update({f'atm {k}':v for k,v in ab.items()})
        for t,fac in (('T',1.0),('cold',0.7)):
            r3=dict(rr); r3['atm temperature']=rr['atm temperature']*fac
            for cp,cpn in ((None,'clear'),(5e2,'500Pa')):
                y=C.render(r3,cp)
                out[f'{tag}_{t}_{cpn}']=np.nan if y is None else np.std(y[meas])*1e6
    rows.append(dict(planet=pl,real=real,twin=out['pub_T_clear'],pub_min=min(out['pub_cold_500Pa'],out['pub_T_500Pa']),h2o4_min=min(out['h2o-4_cold_500Pa'],out['h2o-4_T_500Pa']),h2o5_min=min(out['h2o-5_cold_500Pa'],out['h2o-5_T_500Pa']),floor_min=min(out['floor-7_cold_500Pa'],out['floor-7_T_500Pa'])))
    print(rows[-1],flush=True)
df=pd.DataFrame(rows).round(0); print(df.to_string(index=False))
for c in ('pub_min','h2o4_min','h2o5_min','floor_min'):
    print(c,'population can go at or below the real dispersion on',int((df[c]<=df.real).sum()),'of',len(df))
