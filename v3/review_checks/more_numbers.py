"""Further numbers for the referee pass of 2026-10-30. Output saved as results/review_checks/more_numbers.txt."""
import os, sys, re, warnings
warnings.filterwarnings('ignore')
V3='/Users/fanyang/BioSignatureDetectionModel/v3'; os.chdir(V3); sys.path.insert(0,V3)
import numpy as np, pandas as pd
pd.set_option('display.width',220); pd.set_option('display.max_rows',100)
C=pd.read_csv('results/identifiability_calibration.csv').drop_duplicates('name')
print('1. Retrieval on undecidable labels (p_true <= 0.95)')
for m in ['CH4','H2O','CO2','NH3']:
    y=C[f'{m}_label']==1; pt=C[f'{m}_p_true']; pp=C[f'{m}_p_present']; u=pt<=.95
    print(m,'right (p_true>.5) %.1f; majority class %.1f; within 0.1 of the 0.29 prior %.1f; present among right %.0f'%(100*(pt[u]>.5).mean(),100*max(y[u].mean(),1-y[u].mean()),100*((pp[u]-.29).abs()<.1).mean(),100*y[u&(pt>.5)].mean()))
T=pd.read_csv('results/jwst_panel/panel_mechanism_transplants.csv')
print('\n2. Transplants on absent labels: learners present (majority >=3, split ==2)')
a=T[T.label=='absent']; g=a.pivot_table(index=['planet','molecule'],columns='variant',values='learners_present')
for c in g.columns: print(' ',c,'majority',int((g[c]>=3).sum()),'split',int((g[c]==2).sum()),'of',len(g))
print('\n3. Present labels with a twin above the training ceiling of -2')
import jwst_muting_census as CM, router_real as RR
n=0;h=0;L=[]
for pl,ab in CM.TWINS.items():
    if pl not in RR.PANEL: continue
    for m,v in RR.PANEL[pl]['labels'].items():
        if v=='present':
            n+=1
            if ab[m]>-2: h+=1;L.append((pl,m,ab[m]))
print(h,'of',n,L)
print('\n4. The present labels the re-implemented classifier does not find')
V=pd.read_csv('results/jwst_panel/amplitude_flag_router.csv').set_index(['planet','molecule']); rule='mask, clear maximum (the rule) -> haze, two-sided'
S=pd.read_csv('results/jwst_panel/panel_mechanism_scaled.csv'); tw=S[(S.test=='twin')&(S.precision=='JWST')&(S.variant=='all')].set_index(['planet','molecule']).present_freq
tb=[]
for (pl,m),r in V[V.label=='present'].iterrows():
    if r.published=='present': continue
    t=tw.get((pl,m),np.nan)
    cause='no twin' if not np.isfinite(t) else ('coverage' if t<.5 else ('below 1.95 um, AIRS-only finds it' if r['AIRS-only everywhere']=='present' else 'not assigned'))
    tb.append(dict(planet=pl,molecule=m,verdict=r.published,twin=t,airs=r['AIRS-only everywhere'],method=r[rule],cause=cause))
TB=pd.DataFrame(tb); print(TB.round(2).to_string(index=False)); print(TB.cause.value_counts().to_dict()); TB.to_csv('results/review_checks/missed_present.csv',index=False)
print('\n5. e3 budget, fifth hazed, f=20 %'); 
E=pd.read_csv('results/e3_budget_freehaze.csv')
for h in (.1,.2):
    for tg in ('CH4','H2O','joint'):
        q=E[(E.q==h)&(E.target==tg)]; f=lambda s,ff=None: q[(q.strategy==s)&((q.f==ff) if ff is not None else True)].lost.mean()
        print(h,tg,'classifier %.0f; clear everywhere %.0f; optical->clear %.0f; optical->free %.0f; random->free %.0f'%(f('classifier alone'),f('clear-model retrieval of every planet (existing method)'),f('+ optical, clear-model retrieval',.2),f('+ optical',.2),f('+ random',.2)))
print('\n6.'); os.system("grep -n 'recoverable\\|decidable by\\|errors' results/noisy_vs_noisefree_haze3e7_freehaze.txt | cut -c1-200 | head -8")
print('\n7. Method against fixes label by label: see the transcript of the check; leave-one-planet-out')
Tt=pd.read_csv('results/jwst_panel/panel_tier1.csv'); BASE=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
q=Tt[(Tt.precision=='JWST')&Tt.label.notna()].pivot_table(index=['planet','molecule'],columns='design',values='credit'); plan=np.array([BASE(p) for p in q.index.get_level_values(0)])
for ref in ('AIRS-only','haze+deck-retrained','shape flag'):
    d=q['the method']-q[ref]; per=d.groupby(plan).sum(); loo=d.sum()-per; print(ref,'total',d.sum(),'differ',int((d!=0).sum()),'better',int((d>0).sum()),'worse',int((d<0).sum()),'leave-one-planet-out min',loo.min())
