"""Does the simulated haze's effect on the spectrum depend on the model's top pressure?

TauREx's LeeMieContribution passes a density profile of ones and documents lee_mie_mix_ratio as particles/m3, and with
lee_mie_bottomP/topP unset (forward_model.build_model's default, as used by every haze run so far) the haze fills the
column from the base to the 1e-4 Pa top at one number density. Three POP-I planets (10th, 50th, 90th percentile of the
five-scale-height modulation) are rendered at 3e7 m^-3 with the model top moved and with the haze confined below 1 and
100 Pa; printed is the hazed minus clear depth averaged over the optical points, Ariel's NIRSpec bins and the AIRS bins.

Usage: OMP_NUM_THREADS=1 python haze_top_test.py > results/haze_top_test.txt
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '.')
import forward_model as FM, shift_aerosol as SA, alfnoor_faithful as AF
from bin_spectra import bin_native
wl=np.load('data/native_wl.npy'); edges=AF.layout_edges('tier3_r20'); cen=.5*(edges[1:]+edges[:-1])
P=pd.read_parquet('data/alfnoor_faithful/pop1_params.parquet')
ms=P['modulation_5H'].to_numpy(); order=np.argsort(ms); picks=[order[int(.1*len(P))],order[int(.5*len(P))],order[int(.9*len(P))]]
opt,nir,air=cen<1.1,(cen>1.1)&(cen<1.95),cen>1.95
def spec(r,top=1e-4,haze=None):
    r=dict(r); r['atm top_pressure']=top
    y=FM.spectrum(r,wl,SA.GASES,SA.FILL_GAS,cloud_pressure=cp,haze=haze)
    return bin_native(np.asarray(y,float)[None,:],wl,edges)[0]
print('HAZE radius',SA.HAZE_RADIUS,'q',SA.HAZE_Q,' | delta depth hazed-clear (ppm): optical / NIRSpec / AIRS mean')
for i in picks:
    r=P.iloc[int(i)].to_dict(); cp=r.pop('cloud_pressure',None); r={k:v for k,v in r.items() if k not in ('atm HCN','atm C2H2')}
    print(f"{r['name']}: T {r['atm temperature']:.0f} K, Rp {r['p_radius']:.1f}, Mp {r['p_mass']:.0f}, 5H mod {r['modulation_5H']*1e6:.0f} ppm, own deck {cp:.3g} Pa")
    base=lambda d:f"{1e6*d[opt].mean():8.0f} {1e6*d[nir].mean():7.0f} {1e6*d[air].mean():6.0f}"
    for label,top,hz in [('whole column, model top 1e-4 Pa (as run)',1e-4,{}),('whole column, model top 1e-6 Pa',1e-6,{}),('whole column, model top 1e-2 Pa',1e-2,{}),('confined below 1 Pa, top 1e-4',1e-4,{'topP':1.0}),('confined below 100 Pa, top 1e-4',1e-4,{'topP':100.0})]:
        h=dict(radius=SA.HAZE_RADIUS,q=SA.HAZE_Q,mix_ratio=3e7,**hz)
        try: d=spec(r,top,h)-spec(r,top,None); print(f"   {label:44s} {base(d)}")
        except Exception as e: print('   ',label,'failed',repr(e)[:120])
