"""How much can the fourth batch say, and what does the method do on recall and false positives? Output saved as results/review_checks/how_conclusive.txt."""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
P=pd.read_csv('results/jwst_panel/panel_tier1.csv'); F=pd.read_csv('results/jwst_panel_fourth/fourth_batch.csv')
DES=['published','AIRS-only','haze+deck-retrained','shape flag','the method']
rng=np.random.default_rng(405)
def tab(df,prec):
    return df[(df.precision==prec)&df.label.notna()]
print('1. Power: draw 12 planets from the 23 of the panel (the method built on them, so this favours it); gain over the re-implemented classifier per draw')
for prec in ('JWST','Tier-1'):
    q=tab(P,prec); pv=q.pivot_table(index=['planet','molecule'],columns='design',values='credit'); plan=np.array([BASE(a) for a in pv.index.get_level_values(0)]); d=(pv['the method']-pv['published']).groupby(plan).sum(); n=pd.Series(plan).value_counts().reindex(d.index)
    ix=rng.integers(0,len(d),(100000,12)); g=d.to_numpy()[ix].sum(1); m=n.to_numpy()[ix].sum(1); g24=g/m*24
    print(f' {prec}: per 24 labels mean {g24.mean():+.1f}, 2.5-97.5 % {np.percentile(g24,2.5):+.1f} to {np.percentile(g24,97.5):+.1f}; share of draws at or below the observed new-batch gain ({0.0 if prec=="JWST" else -0.2}): {100*(g24<=(0.0 if prec=="JWST" else -0.2)).mean():.1f} %')
print('\n2. Present and absent labels separately: method minus re-implemented classifier, bootstrap over planets')
for name,df in (('panel',P),('new planets',F)):
    for prec in ('JWST','Tier-1'):
        q=tab(df,prec)
        for lab in ('present','absent'):
            pv=q[q.label==lab].pivot_table(index=['planet','molecule'],columns='design',values='credit'); plan=np.array([BASE(a) for a in pv.index.get_level_values(0)]); d=(pv['the method']-pv['published']).groupby(plan).sum().to_numpy(); ix=rng.integers(0,len(d),(20000,len(d))); b=d[ix].sum(1)
            print(f' {name:<12}{prec:<7}{lab:<8} n={len(pv):>3} re-implemented {pv["published"].sum():5.1f} method {pv["the method"].sum():5.1f} diff {d.sum():+5.1f} ({np.percentile(b,2.5):+.1f},{np.percentile(b,97.5):+.1f}) on {len(d)} planets')
print('\n3. Asymmetric cost: credit with a missed present label weighted w against a false present (per the reviewer comment B11); method minus each design')
for name,df in (('panel',P),('new planets',F)):
    q=tab(df,'JWST')
    for w in (1,2,3,5):
        c=q.assign(c=np.where(q.label=='present',w*q.credit,q.credit)).groupby('design').c.sum(); print(f' {name:<12} w={w}: '+', '.join(f'{d} {c[d]:.1f}' for d in DES))
print('\n4. Ranking: AUROC of present against absent labels, score = share of draws present + half split, Tier-1; per molecule pooled; 95 % bootstrap of method minus re-implemented over planets')
def auc(y,s):
    a=s[y==1]; b=s[y==0]
    return np.nan if not len(a) or not len(b) else (np.mean(a[:,None]>b[None,:])+.5*np.mean(a[:,None]==b[None,:]))
for name,df in (('panel',P),('new planets',F)):
    for prec in ('Tier-1',):
        q=tab(df,prec); q=q.assign(s=q.present+.5*q.split,y=(q.label=='present').astype(int),u=q.planet.map(BASE)); units=q.u.unique(); out={}
        for d in DES:
            g=q[q.design==d]; out[d]=auc(g.y.to_numpy(),g.s.to_numpy())
        diffs=[]
        for _ in range(2000):
            pick=rng.choice(units,len(units)); a=pd.concat([q[q.u==u] for u in pick]); m=a[a.design=='the method']; r=a[a.design=='published']; diffs.append(auc(m.y.to_numpy(),m.s.to_numpy())-auc(r.y.to_numpy(),r.s.to_numpy()))
        print(f' {name}: '+', '.join(f'{k} {v:.2f}' for k,v in out.items())+f'; method minus re-implemented {np.nanpercentile(diffs,2.5):+.2f} to {np.nanpercentile(diffs,97.5):+.2f}')
print('\n5. Pooled, all 90 labels, 35 planets, descriptive only (66 are development labels)')
for prec in ('JWST','Tier-1'):
    a=tab(P,prec); b=tab(F,prec); q=pd.concat([a[a.design.isin(DES)],b[b.design.isin(DES)]]); pv=q.pivot_table(index=['planet','molecule'],columns='design',values='credit'); plan=np.array([BASE(x) for x in pv.index.get_level_values(0)]); d=(pv['the method']-pv['published']).groupby(plan).sum().to_numpy(); n=pd.Series(plan).value_counts().to_numpy(); ix=rng.integers(0,len(d),(20000,len(d))); bb=d[ix].sum(1)
    print(f' {prec}: labels {len(pv)}; '+', '.join(f'{k} {pv[k].sum():.1f}' for k in DES)+f'; method minus re-implemented {d.sum():+.1f} ({np.percentile(bb,2.5):+.1f},{np.percentile(bb,97.5):+.1f})')
