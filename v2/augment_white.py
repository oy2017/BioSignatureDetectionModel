"""Second stochastic axis: does augmenting with WHITE noise recover more than
augmenting with correlated noise? If the deterministic/stochastic dichotomy is
real, both stochastic axes should recover far less than the deterministic ones."""
import json, os, sys
import joblib, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from common import DATA, MODELS, RESULTS, SEED, SNR, TESTS, Features, load_split, metrics
from pipeline import make_xgb
from augment import corr_noise
cfg="ariel"; CLEAN=0.9033
best=json.load(open(os.path.join(RESULTS,f"{cfg}_best.json")))["best"]
fr=joblib.load(os.path.join(MODELS,f"{cfg}_{best}.joblib"))
ff,fm,params=fr["features"],fr["model"],fr["params"]; kind=best.split("_")[0]
Xtr,ytr,Ptr=load_split("train",cfg)
from common import centres
cen=centres(cfg)
import numpy as _np, os as _os
Xtr_nf=_np.load(_os.path.join(DATA,f"train_{cfg}.npy")).astype(float)
Xc=np.vstack([load_split(t,cfg)[0] for t in TESTS]); yc=np.concatenate([load_split(t,cfg)[1] for t in TESTS])
Xc_nf=np.vstack([np.load(os.path.join(DATA,f"{t}_{cfg}.npy")) for t in TESTS]).astype(float)
Pc=pd.concat([load_split(t,cfg,noisy=False)[2] for t in TESTS],ignore_index=True)
L=["White-noise augmentation (second stochastic axis)","",
   f"{'case':<12}{'frozen':>9}{'augmented':>11}{'gain':>8}{'% of gap':>10}"]
rng=np.random.default_rng(SEED+7)
lev=rng.choice([15,12,10,8],size=len(ytr))
Xa=Xtr.copy()
for s in (12,10,8):
    m=lev==s; Xa[m]=corr_noise(Xtr[m],rng,s,kind="white",Xnf=Xtr_nf[m],params=Ptr[m].reset_index(drop=True),cen=cen)
f=Features(kind).fit(Xa); m_aug=make_xgb(params).fit(f.transform(Xa),ytr)
rng2=np.random.default_rng(SEED+8); rows=[]
for s in (12,10,8,5):
    Xs=corr_noise(Xc,rng2,s,kind="white",Xnf=Xc_nf,params=Pc,cen=cen)
    a_fr=metrics(yc,fm.predict_proba(ff.transform(Xs))[:,1])["accuracy"]
    a_au=metrics(yc,m_aug.predict_proba(f.transform(Xs))[:,1])["accuracy"]
    pct=(a_au-a_fr)/max(CLEAN-a_fr,1e-9)*100
    L.append(f"{'SNR '+str(s):<12}{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}{pct:9.0f}%")
    rows.append(dict(axis="white noise",case=f"snr{s}",frozen=a_fr,augmented=a_au,pct_of_gap=pct))
a_fr=metrics(yc,fm.predict_proba(ff.transform(Xc))[:,1])["accuracy"]
a_au=metrics(yc,m_aug.predict_proba(f.transform(Xc))[:,1])["accuracy"]
L.append(f"{'clean':<12}{a_fr*100:8.2f}%{a_au*100:10.2f}%{(a_au-a_fr)*100:+7.2f}")
open(os.path.join(RESULTS,"ariel_augment_white.txt"),"w").write("\n".join(L)+"\n")
pd.DataFrame(rows).to_csv(os.path.join(RESULTS,"ariel_augment_white.csv"),index=False)
print("\n".join(L))
