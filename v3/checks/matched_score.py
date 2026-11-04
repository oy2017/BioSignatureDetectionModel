"""Scores of the per-reduction learners beside the frozen designs. Output: results/matched_learner/matched_learner.txt"""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE=lambda s:s.replace('-alt','').replace('-hu','').replace('-hm','')
M=pd.read_csv('results/matched_learner/matched_learner.csv'); P=pd.read_csv('results/jwst_panel/panel_tier1.csv').assign(batch='panel'); F=pd.read_csv('results/jwst_panel_fourth/fourth_batch.csv').assign(batch='fourth')
A=pd.concat([P,F,M]); A=A[A.label.notna()]; DES=['published','AIRS-only','haze+deck-retrained','the method','matched','matched+amp']; A=A[A.design.isin(DES)]; A['u']=A.planet.map(BASE)
rng=np.random.default_rng(405)
def auc(y,s):
    a=s[y==1]; b=s[y==0]; return (np.mean(a[:,None]>b[None,:])+.5*np.mean(a[:,None]==b[None,:])) if len(a) and len(b) else np.nan
L=[]
for prec in ('JWST','Tier-1'):
    for b in ('panel','fourth','all'):
        q=A[(A.precision==prec)&((A.batch==b)|(b=='all'))]; L.append(f'== {prec} {b}: {q[q.design=="published"].shape[0]} labels')
        for d in DES:
            g=q[q.design==d]; pr=g[g.label=='present'].credit; ab=g[g.label=='absent'].credit; bal=.5*(pr.mean()+ab.mean()); s=g.present+.5*g.split
            pv=q[q.design.isin([d,'the method'])].pivot_table(index=['u','planet','molecule'],columns='design',values='credit')
            if d!='the method':
                x=(pv[d]-pv['the method']).groupby('u').sum().to_numpy(); bb=x[rng.integers(0,len(x),(20000,len(x)))].sum(1); t=f'vs method {x.sum():+5.1f} ({np.percentile(bb,2.5):+.1f},{np.percentile(bb,97.5):+.1f})'
            else: t=''
            L.append(f'  {d:<22} score {g.credit.sum():5.1f}  present {pr.sum():5.1f}/{len(pr)}  absent {ab.sum():5.1f}/{len(ab)}  balanced {bal:.3f}  AUROC {auc((g.label=="present").to_numpy().astype(int),s.to_numpy()):.2f}  {t}')
j=A[(A.precision=='JWST')&A.design.isin(['the method','matched','matched+amp'])].assign(v=lambda d:np.where(d.present==1,'P',np.where(d.absent==1,'A','S'))).pivot_table(index=['batch','planet','molecule','label'],columns='design',values='v',aggfunc='first')
L+=['',j.to_string()]
t='\n'.join(L); open('results/matched_learner/matched_learner.txt','w').write(t+'\n'); print(t)
