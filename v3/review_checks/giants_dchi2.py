"""Check from the referee pass of 2026-10-30. Output saved as results/review_checks/giants_dchi2.txt."""
import os, sys
V3='/Users/fanyang/BioSignatureDetectionModel/v3'; os.chdir(V3)
import numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
I=pd.read_csv('results/identifiability_none_tier1.csv'); print(I.columns.tolist()[:12], len(I))
thr={'CH4':0.45,'H2O':0.98,'CO2':0.90,'NH3':0.56}
M={'HAT-P-26b':5.32,'WASP-107b':4.45,'WASP-39b':3.73,'GJ3470b':3.45,'WASP-166b':3.07,'WASP-17b':3.99,'WASP-80b':2.89,'HD209458b':2.35,'WASP-94Ab':3.30,'HAT-P-12b':5.34}
I['thr']=I.molecule.map(thr); I['dec']=I.dchi2>I.thr
g=I[I.name.isin(M)].copy(); print(sorted(g.name.unique()), len(g))
g['M']=g.name.map(M); g['dec_M']=(g.dchi2/g.M**2)>g.thr; g['dec_x3']=(g.dchi2/9)>g.thr
print('population undecidable %',(100*(1-I.groupby('molecule').dec.mean())).round(1).to_dict())
print('population x3 undecidable %',(100*(1-((I.dchi2/9)>I.thr).groupby(I.molecule).mean())).round(1).to_dict())
print('ten giants undecidable (their random POP-I abundances) %',(100*(1-g.groupby('molecule').dec.mean())).round(1).to_dict(), 'all %.1f'%(100*(1-g.dec.mean())))
print('ten giants, dchi2/M^2 undecidable %',(100*(1-g.groupby('molecule').dec_M.mean())).round(1).to_dict(),'all %.1f'%(100*(1-g.dec_M.mean())))
print(g.groupby('name')[['dec','dec_M']].mean().round(2).T.to_string())
# noise of the ten giants relative to the population: AIRS sigma median
P=pd.read_parquet('data/alfnoor_faithful/pop1_params.parquet')
pct=lambda n,col: float((P[col]<P[P.name==n][col].iloc[0]).mean()*100)
# Compare each giant's median dchi2 percentile
pc=I.groupby('molecule').dchi2.rank(pct=True); I['pc']=pc
print('median percentile of dchi2 for ten giants', round(100*I[I.name.isin(M)].pc.median(),1))
print(I[I.name.isin(M)].groupby('name').pc.median().round(2).to_dict())
# population median dchi2 vs giants
print('median dchi2: population',I.dchi2.median().round(2),'giants',g.dchi2.median().round(2))
# Absent / present label dist.
