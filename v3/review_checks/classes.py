"""Check from the referee pass of 2026-10-30. Output saved as results/review_checks/classes.txt."""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
V3='/Users/fanyang/BioSignatureDetectionModel/v3'; os.chdir(V3); sys.path.insert(0,V3)
import numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
import alfnoor_diagnostic as D
cases,Pte,e,sig=D.test_cases(); sc=D.Scorer(e)
P=sc.probs(cases['clean'])
C=pd.read_csv('results/identifiability_calibration.csv').drop_duplicates('name').set_index('name')
names=[n for n in Pte.name if n in C.index]; idx={n:i for i,n in enumerate(Pte.name)}; I=np.array([idx[n] for n in names]); C=C.loc[names]
print(len(names))
tot={}
for m in D.MOLS:
    y=(Pte[f'atm {m}'].to_numpy()[I]>-4).astype(int)
    pt=C[f'{m}_p_true'].to_numpy(float); pp=C[f'{m}_p_present'].to_numpy(float)
    ret_right=(pp>=0.5)==(y==1)
    for k in D.LEARNERS:
        w=(P['all'][(m,k)][I]>=0.5)!=(y==1); fn=w&(y==1); fp=w&(y==0)
        tot.setdefault('n_err',[]).append(w.sum()); tot.setdefault('fn',[]).append(fn.sum()); tot.setdefault('fp',[]).append(fp.sum())
        for lev,name in ((0.95,'dec95'),(0.84,'dec84'),(0.5,'dec50')):
            tot.setdefault(name,[]).append((pt[w]>lev).sum())
        tot.setdefault('ret_right',[]).append(ret_right[w].sum())
        tot.setdefault('ret_right_fn',[]).append(ret_right[fn].sum()); tot.setdefault('ret_right_fp',[]).append(ret_right[fp].sum())
        und=pt<=0.95
        tot.setdefault('err_und',[]).append((w&und).sum())
        tot.setdefault('ret_right_und',[]).append((ret_right&w&und).sum())
        tot.setdefault('ret_right_und_fn',[]).append((ret_right&fn&und).sum()); tot.setdefault('fn_und',[]).append((fn&und).sum())
        tot.setdefault('ret_right_und_fp',[]).append((ret_right&fp&und).sum()); tot.setdefault('fp_und',[]).append((fp&und).sum())
T={k:np.sum(v) for k,v in tot.items()}
print(T)
print('errors',T['n_err'],'FN share %.1f FP share %.1f'%(100*T['fn']/T['n_err'],100*T['fp']/T['n_err']))
for n in ('dec95','dec84','dec50'): print(n,'share of errors with p_true above level: %.1f'%(100*T[n]/T['n_err']))
print('retrieval right on all errors %.1f; FN %.1f; FP %.1f'%(100*T['ret_right']/T['n_err'],100*T['ret_right_fn']/T['fn'],100*T['ret_right_fp']/T['fp']))
print('on errors on undecidable labels: n %d; retrieval right %.1f; FN (n %d) %.1f; FP (n %d) %.1f'%(T['err_und'],100*T['ret_right_und']/T['err_und'],T['fn_und'],100*T['ret_right_und_fn']/T['fn_und'],T['fp_und'],100*T['ret_right_und_fp']/T['fp_und']))
# Per molecule share at 0.84
for m in D.MOLS:
    y=(Pte[f'atm {m}'].to_numpy()[I]>-4).astype(int); pt=C[f'{m}_p_true'].to_numpy(float)
    r=[];r2=[]
    for k in D.LEARNERS:
        w=(P['all'][(m,k)][I]>=0.5)!=(y==1); r.append((pt[w]>0.95).mean()); r2.append((pt[w]>0.84).mean())
    print(m,'recoverable at 0.95: %.1f; at 0.84: %.1f'%(100*np.mean(r),100*np.mean(r2)))
# Noisy subsample: FN / FP split and who is right
N=pd.read_csv('results/identifiability_calibration_noisy.csv').drop_duplicates('name'); N=N[N['error'].isna()] if 'error' in N else N; N=N.set_index('name')
ns=[n for n in N.index if n in idx]; J=np.array([idx[n] for n in ns]); N=N.loc[ns]
t={}
for m in D.MOLS:
    y=(Pte[f'atm {m}'].to_numpy()[J]>-4).astype(int); pp=N[f'{m}_p_present'].to_numpy(float); rr=(pp>=0.5)==(y==1)
    for k in D.LEARNERS:
        w=(P['all'][(m,k)][J]>=0.5)!=(y==1); fn=w&(y==1); fp=w&(y==0)
        t.setdefault('w',[]).append(w.sum()); t.setdefault('fn',[]).append(fn.sum()); t.setdefault('fp',[]).append(fp.sum()); t.setdefault('r',[]).append(rr[w].sum()); t.setdefault('rfn',[]).append(rr[fn].sum()); t.setdefault('rfp',[]).append(rr[fp].sum())
t={k:np.sum(v) for k,v in t.items()}; print('noisy realisation (n=%d planets):'%len(ns),t,'right all %.1f FN %.1f FP %.1f'%(100*t['r']/t['w'],100*t['rfn']/t['fn'],100*t['rfp']/t['fp']))
# Classifier majority on the same undecidable labels vs retrieval, on noisy realisation
