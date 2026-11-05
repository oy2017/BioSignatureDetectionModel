"""AF2: the matched ensemble against the routing method and the other designs on the real labels under alternative label sets.

Sets: (a) committed 64 + 24 (+17); (b) the 65-label rule-consistent panel set of label_sensitivity.py plus the later batches;
(c) its 59-label judgement-free set plus the later batches; (d) one uniform rule across all three batches: a label is scored only if
every printed free-chemistry variant (label_sources.csv notes for the panel, fourth_batch_second_reader.csv for the fourth batch,
independent_data_survey.md for the 17 further labels) lies on one side of 10^-4; variants are pooled per planet across reductions.
A 1-sigma interval is one variant; a quoted limit (any level) is a variant where it is the reported constraint; a second level
printed for the same posterior is not a separate variant. (d') the same, with limits that lie above 10^-4 treated as uninformative
rather than crossing. The ensemble has stored verdicts only on the committed labels, so paired differences cover labels both designs
have; labels that enter under a rule are scored for the baselines only. Writes results/round3/real_label_rules.txt."""
import os, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = lambda s: s.replace('-alt', '').replace('-hu', '').replace('-hm', '').replace('-JWST', '').replace('-HST', '')
DES = ['the method', 'published', 'haze+deck-retrained', 'AIRS-only']; ENS = 'C+B-mlp'; TH = -4.0

# ---- data, as in checks/round3_real_new.py, but keeping the unscored panel cells with their present/split/absent columns
P = pd.read_csv('results/jwst_panel/panel_tier1.csv').assign(batch='panel'); F = pd.read_csv('results/jwst_panel_fourth/fourth_batch.csv').assign(batch='fourth')
I = pd.read_csv('results/independent_test/independent_test.csv').assign(batch='independent', precision='JWST'); N = pd.read_csv('results/round3/real_C+B-mlp.csv')
for X in (P, F): X['credit_if_absent'] = X.absent + 0.5 * X.split; X['credit_if_present'] = X.present + 0.5 * X.split
C = ['batch', 'planet', 'precision', 'molecule', 'design', 'label', 'credit']
A = pd.concat([P[C + ['credit_if_absent', 'credit_if_present']], F[C + ['credit_if_absent', 'credit_if_present']], I[C], N[C]], ignore_index=True); A['u'] = A.planet.map(BASE)
committed = {(r.batch, r.planet, r.molecule): r.label for r in A[(A.design == 'published') & (A.precision == 'JWST')].itertuples()}

# ---- sets (b) and (c): copied from label_sensitivity.py
OUT_REAL = [('TOI-270d', 'NH3')]; IN_REAL = {('HD209458b', 'CO2'): 'absent', ('WASP-52b', 'NH3'): 'absent'}
OUT_JUDGE = [('TOI-270d-hm', 'NH3'), ('HAT-P-18b', 'CO2'), ('L98-59d', 'H2O'), ('L98-59d', 'CH4'), ('L98-59d', 'NH3'), ('WASP-17b', 'H2O'), ('WASP-39b', 'H2O')]; IN_JUDGE = {('WASP-52b', 'CO2'): 'absent'}
panel_committed = {k[1:]: v for k, v in committed.items() if k[0] == 'panel' and isinstance(v, str)}
corrected = {k: v for k, v in panel_committed.items() if k not in OUT_REAL}; corrected.update(IN_REAL)
strict = {k: v for k, v in corrected.items() if k not in OUT_JUDGE}; strict.update(IN_JUDGE)
assert len(panel_committed) == 64 and len(corrected) == 65 and len(strict) == 59

# ---- uniform rule: printed free-chemistry variants per planet unit and molecule (log10 VMR). 'm' median+plus-minus; '<' upper limit; '>' lower limit; '[' range; 'p' point (no interval printed).
V = {}
def add(unit, mol, *vs): V.setdefault((unit, mol), []).extend(vs)
# panel (results/jwst_panel/label_sources.csv, value columns and notes)
add('WASP-39b', 'H2O', '-2.08+0.09-0.09', '-2.64+0.08-0.08', '-2.47+0.52-0.46', 'p-1.629', '-5.28+0.37-0.36', '-6.35+0.34-0.33', '-4.15+0.49-0.38', '-3.10+0.20-0.19', '-3.53+0.44-0.31', '-1.45+0.19-0.20', '-1.13+0.08-0.12', '-4.85+0.38-0.35', '-3.29+0.59-0.56', '-3.27+0.26-0.24')
add('TOI-270d', 'CH4', '-1.72+0.28-0.29', '-2.11+0.37-0.40', '-2.61+0.36-0.38', '-1.64+0.38-0.36', '-2.44+0.34-0.46', '-2.42+0.40-0.53', 'p-2.70', 'p-2.97')
add('TOI-270d', 'CO2', '-1.29+0.17-0.19', '-1.90+0.36-0.46', '-1.67+0.40-0.60', '-1.96+0.49-0.79', '-1.93+0.49-0.77', '-3.95+0.72-0.90')
add('TOI-270d', 'NH3', '-3.83+0.51-0.77', '-3.21+0.38-0.55', '-5.9+1.6-3.9', '<-4.27', '<-5.75', '<-5.87')
add('K2-18b', 'CH4', '-1.74+0.59-0.69', '-2.04+0.61-0.72', '-1.89+0.63-0.70', '[-2.14,-0.53]', '-1.06+0.24-0.37', '-1.15+0.31-0.46', '-1.09+0.21-0.26')
add('K2-18b', 'H2O', '<-3.06', '<-3.21', '<-3.49', '<-3.59', '<-2.44', '<-5.25', '<-6.89', '<-3.52', '<-3.57')
add('K2-18b', 'NH3', '<-4.51', '<-4.46', '<-4.93', '<-5.28', '<-4.60', '<-3.66', '<-4.70', '<-5.59', '<-5.62', '<-4.93', '<-5.17')
add('TOI-1231b', 'CH4', '-0.85+0.18-0.19', '-1.19+0.25-0.27', '-1.06+0.24-0.24', '-3.78+0.40-0.33')
add('TOI-1231b', 'CO2', '-1.9+0.4-0.71', '-1.84+0.40-0.62', '-1.45+0.26-0.39', '-2.19+0.51-0.93', '-6.05+0.70-0.66')
add('TOI-421b', 'CH4', '-8.39+1.91-2.18', '-9.15+1.66-1.89'); add('TOI-421b', 'CO2', '-9.14+2.02-1.71', '-9.29+1.85-1.74')
add('HAT-P-18b', 'CH4', '-5.08+0.35-0.34', '<-6.29', '<-6.12', '<-6.30'); add('HAT-P-18b', 'CO2', '-4.19+0.4-0.4', '-5.72+0.61-0.72', '-4.86+0.45-0.45', '-4.39+0.38-0.26', '-5.16+0.40-0.36'); add('HAT-P-18b', 'NH3', '<-6.01', '<-6.13', '<-6.21')
add('WASP-166b', 'CH4', '-7.93+1.34-1.37'); add('WASP-166b', 'H2O', '-1.3+0.21-0.23', '-1.42+0.20-0.24'); add('WASP-166b', 'CO2', '-2.23+0.31-0.37', '-2.13+0.27-0.39')
add('WASP-107b', 'CH4', '-5.8+0.2-0.2', '-6.1+0.5-0.4', '-6.03+0.22-0.20', '-6.14+0.42-1.70', '-8.52+2.09-2.09', '-8.83+1.25-1.23')
add('WASP-107b', 'H2O', '-2.1+0.2-0.3', '-2.6+0.7-0.6', '-1.85+0.22-0.23', '-1.72+0.26-0.25', '-2.19+0.42-0.26', '-3.81+0.38-0.29')
add('WASP-107b', 'NH3', '-5.0+0.2-0.2', '-5.1+0.5-0.5', '-6.38+0.78-1.54', '-8.12+1.99-2.36', '-5.47+0.34-0.66', '-6.04+0.30-0.25')
add('HD189733b', 'CH4', '<-7.9', '-9.88+1.35-1.31'); add('HD189733b', 'CO2', '-6.24+0.67-0.42', '-5.32+0.37-0.66'); add('HD189733b', 'NH3', '<-5.3', '-6.35+0.80-3.11')
add('HAT-P-26b', 'CH4', '<-5.49', '<-4.99', '<-5.03', '<-5.18', '<-4.59'); add('HAT-P-26b', 'H2O', '-2.07+0.39-0.59', '-1.30+0.12-0.12', '-1.41+0.17-0.36', '-2.06+0.55-0.68')
add('HAT-P-26b', 'CO2', '-2.91+0.56-0.71', '-2.38+0.29-0.26', '-2.80+0.38-0.59', '-3.31+0.73-0.73'); add('HAT-P-26b', 'NH3', '<-4.20', '<-4.05', '<-4.43', '<-4.03', '<-4.09', '<-3.85', '<-4.04', '<-4.17')
add('GJ3470b', 'H2O', '-1.08+0.43-0.52'); add('GJ3470b', 'CO2', '-2.47+0.61-0.43'); add('GJ3470b', 'NH3', '-7.3+2.78-3.02')
add('GJ9827d', 'H2O', '>-1.86', '>-0.50')
add('WASP-52b', 'CH4', '<-7.39', '<-7.30', '<-6.41'); add('WASP-52b', 'CO2', '-5.09+1.09-1.14', '-5.71+0.84-0.93', '-7.11+1.11-0.50', '-5.67+1.18-1.23', '-5.70+0.65-1.24', '<-6.16'); add('WASP-52b', 'NH3', '<-5.55', '<-4.58', '<-5.94', '<-6.03', '<-6.07')
add('L98-59d', 'CH4', '<-6.0', '<-1.8', '<-4.4', '<-0.9'); add('L98-59d', 'H2O', '<-5.0', '<-0.3', '<-2.7', '<-0.1'); add('L98-59d', 'NH3', '<-6.0', '<-1.6', '<-4.8', '<-0.4')
add('GJ1214b', 'CH4', '-2.25+0.72-1.08'); add('GJ1214b', 'CO2', '[-0.337,-0.004]')  # 91 +8/-45 % bulk fraction
add('LTT3780c', 'CH4', '-2.34+0.71-0.8', '-2.44+0.71-0.76', '-1.57+0.36-0.48'); add('LTT3780c', 'NH3', '<-4.44', '<-4.13', '<-4.15', '<-4.80', '<-4.22', '<-4.36')
add('WASP-80b', 'H2O', '-1.8+0.6-1.0')  # the 3-sigma lower limit -4.70 is the same posterior
add('HD209458b', 'CH4', '<-5.6', '-9.51+1.55-1.49', '<-7.30', '<-6.35'); add('HD209458b', 'CO2', '-5.31+0.37-1.12', '-6.08+0.85-1.10', '-7.52+0.30-0.26', '-7.16+1.02-0.76'); add('HD209458b', 'NH3', '<-4.2', '-8.22+2.47-2.29', '<-5.79', '<-5.16')
add('WASP-96b', 'CH4', '<-5.91', '<-5.81', '<-6.18', '-5.75+0.27-0.26'); add('WASP-96b', 'H2O', '-2.61+0.27-0.3', '-2.82+0.24-0.26', '-2.67+0.29-0.28', '-2.76+0.30-0.28')
add('WASP-96b', 'CO2', '-4.65+0.34-0.39', '-4.91+0.30-0.30', '-4.48+0.32-0.30', '-4.46+0.34-0.31', '-4.15+0.70-0.77', '-4.34+0.35-0.38'); add('WASP-96b', 'NH3', '<-5.47', '<-5.87', '<-5.49', '<-3.91')
add('WASP-17b', 'CH4', '-9.03+2.0-1.93', '-9.06+2.04-1.91', '-10.42+2.24-2.13', '-8.52+1.69-2.23'); add('WASP-17b', 'H2O', '-2.96+0.31-0.24', '-3.00+0.17-0.16', '-2.67+0.42-0.37', '-4.53+0.37-0.36', '-4.75+0.33-0.29')
add('WASP-17b', 'CO2', '-5.8+0.62-1.31', '-6.23+0.75-2.74', '-5.40+0.56-0.67', '-5.54+0.53-0.72')
add('WASP-94Ab', 'CH4', '<-6.29', '<-5.83'); add('WASP-94Ab', 'H2O', '-1.6+0.43-0.4', '-1.59+0.35-0.64'); add('WASP-94Ab', 'NH3', '<-5.10', '<-5.07')
add('HIP67522b', 'CO2', '-5.56+0.81-0.95')
add('HAT-P-12b', 'CH4', '<-6.27', '<-6.33'); add('HAT-P-12b', 'H2O', '-1.93+0.3-0.32', '-1.82+0.38-0.44'); add('HAT-P-12b', 'NH3', '<-7.58', '<-6.57', '<-4.16')
NOFREE = {('WASP-39b', 'CH4'): 'no free-chemistry value printed (3-sigma limit from equilibrium-model grids; Constantinou prints no number)'}
# committed-unscored panel cells that stay unscored under any reading (reason from label_sources.csv); listed so the audit is complete
STAY = 'straddles or limit above -4 or molecule not reported or literature disagrees across -4'
# fourth batch (results/jwst_panel/fourth_batch_second_reader.csv, value_as_printed)
add('TrES-4b', 'CH4', '<-7.07', '<-7.01', '<-7.01', '<-7.47', '<-7.54', '<-7.20', '<-7.53'); add('TrES-4b', 'H2O', '-2.98+0.68-0.73', '-3.29+0.57-0.59'); add('TrES-4b', 'CO2', '-6.86+0.62-0.65', '-6.81+0.56-0.56')
add('HATS-6b', 'CH4', '-5.38+0.18-0.19', '-5.22+0.29-0.29', '-5.27+0.22-0.23', '-5.62+0.20-0.21', '-5.39+0.19-0.19'); add('HATS-6b', 'H2O', '-4.88+0.25-0.24', '-4.69+0.37-0.37', '-4.69+0.38-0.36', '-4.55+0.28-0.28')
add('HATS-6b', 'CO2', '-7.59+0.41-0.44', '-6.85+0.49-0.49'); add('HATS-6b', 'NH3', '-6.03+0.18-0.19', '-5.72+0.22-0.22')
add('TOI-5205b', 'H2O', '-10.0+1.4-1.3', '-9.04+1.57-1.88', '-8.16+1.44-2.46'); add('TOI-5205b', 'CO2', '-8.6+0.7-1.6', '-6.4+1.2-1.2'); add('TOI-5205b', 'NH3', '-7.06+0.44-1.98')
add('TOI-6894b', 'CH4', '-1.67+0.16-0.16', '-1.64+0.16-0.16'); add('TOI-6894b', 'H2O', '-1.40+0.12-0.15', '-1.36+0.11-0.13'); add('TOI-6894b', 'NH3', '-3.19+0.24-0.26', '-3.14+0.24-0.26')
add('TOI-199b', 'CH4', '-2.46+0.85-1.15', '-2.19+0.71-1.28', '-2.47+0.81-1.17', '-2.33+0.80-1.45', '-1.35+0.56-0.99')
add('LP791-18c', 'CH4', '-0.86+0.53-1.70', '-0.35+0.15-0.27', '-0.23+0.23-0.63')
add('NGTS-2b', 'CH4', '-9.83+1.58-1.50', '-7.04+1.80-1.80'); add('NGTS-2b', 'NH3', '-9.23+1.92-1.86', '-7.71+2.45-2.45')
add('HAT-P-30b', 'CH4', '<-5.9', '<-6.1', '<-4.1', '<-5.9', '<-5.7', '<-5.7', '<-6.0', '<-6.1', '<-6.2', '-8.05+1.17-2.22'); add('HAT-P-30b', 'NH3', '-9.35+1.72-1.72', '-9.42+1.78-1.68')
add('HAT-P-1b', 'CH4', '-9.55+1.65-1.59', '-9.56+1.73-1.58'); add('HAT-P-1b', 'H2O', '-1.70+0.43-0.71', '-1.68+0.24-0.30')
add('TOI-1130b', 'H2O', '>-2.58')  # mass fraction -0.7 +0.3/-1.1 converted at mu = 3 (second reader)
add('Kepler-51d', 'CH4', '[-7.68,-6.86]')  # 66 +72/-45 ppb
add('WASP-121b', 'CO2', '-8.64+1.82-1.99', '-8.20+2.11-2.35', '-8.80+1.64-1.89', '-9.02+1.92-1.54')
# 17 further labels (independent_data_survey.md Tables A1 and B3; results/independent_test/SOURCES.md repeats the V1298 Tau c and HAT-P-14 b values)
add('V1298Tauc', 'CH4', '-8.38+2.36-2.20', '-8.45+2.22-2.24'); add('V1298Tauc', 'H2O', '-1.83+0.68-0.77', '-1.59+0.60-0.65', '-1.56+0.32-0.52'); add('V1298Tauc', 'CO2', '-7.35+2.81-2.89', '-7.12+2.96-2.90'); add('V1298Tauc', 'NH3', '-8.30+2.48-2.28', '-8.43+2.26-2.20')
add('HAT-P-14b', 'CH4', '-7.46+1.96-1.65', '-7.14+2.20-1.87'); add('HAT-P-14b', 'CO2', '-6.58+2.18-2.20', '-7.00+2.27-1.95')  # log mass fraction; VMR lies 0.8-1.3 dex lower, so the test is conservative
add('HAT-P-32b', 'H2O', '-1.18+0.12-0.32', '-2.84+0.92-0.92', '-2.39+1.09-1.57'); add('HAT-P-41b', 'H2O', '-2.55+0.92-1.36', '-2.77+1.09-1.09', '-0.90+0.28-1.20')
add('HD106315c', 'H2O', '-1.92+0.58-1.01', '-1.40+0.40-0.74'); add('WASP-62b', 'H2O', '-1.40+0.23-0.51', '-2.03+0.52-1.27'); add('WASP-79b', 'H2O', '-2.05+0.40-0.60', '-2.43+0.57-0.76')
add('GJ436b', 'H2O', '-6.89+2.83-2.86', '-6.74+2.70-2.70'); add('HAT-P-2b', 'H2O', '-7.23+2.80-2.83'); add('HIP41378f', 'H2O', '-8.80+1.14-1.77'); add('K2-24b', 'H2O', '-7.83+2.92-2.64'); add('WASP-103b', 'H2O', '-7.71+2.86-2.59')

def interval(s):
    if s[0] == '<': return (-np.inf, float(s[1:]), 'ul')
    if s[0] == '>': return (float(s[1:]), np.inf, 'll')
    if s[0] == '[': a, b = s[1:-1].split(','); return (float(a), float(b), 'm')
    if s[0] == 'p': return (float(s[1:]), float(s[1:]), 'm')
    i = s.index('+'); j = s.index('-', i); m, pl, mi = float(s[:i]), float(s[i + 1:j]), float(s[j + 1:]); return (m - mi, m + pl, 'm')

def rule(vs, tolerant, lab):
    iv = [(v, interval(v)) for v in vs]
    if tolerant: iv = [(v, x) for v, x in iv if not ((x[2] == 'ul' and x[1] > TH) or (x[2] == 'll' and x[0] < TH))]
    if not iv: return None, 'no informative variant'
    if all(x[1] <= TH + 1e-9 for v, x in iv): return 'absent', ('boundary' if any(abs(x[1] - TH) < 0.005 for v, x in iv) else '')
    if all(x[0] >= TH - 1e-9 for v, x in iv): return 'present', ('boundary' if any(abs(x[0] - TH) < 0.005 for v, x in iv) else '')
    off = [v for v, x in iv if not (x[1] <= TH + 1e-9 if lab == 'absent' else x[0] >= TH - 1e-9 if lab == 'present' else not (x[0] < TH < x[1]))]
    return None, 'variants not on the committed side: ' + ', '.join(off[:4])

uniform, log = {}, {'literal': [], 'tolerant': []}
for tol, tag in ((False, 'literal'), (True, 'tolerant')):
    labs = {}
    for (b, pl, m), lab in committed.items():
        u = BASE(pl); key = (u, m)
        if (pl, m) in NOFREE: labs[(b, pl, m)] = lab; log[tag].append(f'    kept as committed, rule not applicable: {b} {pl} {m} ({lab}): {NOFREE[(pl, m)]}'); continue
        if key not in V:
            if isinstance(lab, str): labs[(b, pl, m)] = lab; log[tag].append(f'    kept as committed, no variant list encoded: {b} {pl} {m} ({lab})')
            continue
        r, why = rule(V[key], tol, lab)
        if tol and not isinstance(lab, str) and rule(V[key], False, lab)[0] is None: continue  # (d') only relaxes the leaving of committed labels; it lets no further cell enter
        if isinstance(lab, str) and r == lab: labs[(b, pl, m)] = lab; log[tag].append(f'    boundary, kept: {b} {pl} {m} ({lab})') if why == 'boundary' else None
        elif isinstance(lab, str) and r is None: log[tag].append(f'    leaves: {b} {pl} {m} (was {lab}); {why}')
        elif isinstance(lab, str): log[tag].append(f'    FLIPS: {b} {pl} {m} committed {lab}, rule {r}'); labs[(b, pl, m)] = r
        elif r is not None: labs[(b, pl, m)] = r; log[tag].append(f'    enters: {b} {pl} {m} as {r}' + (' (1-sigma top exactly -4.00 on one variant)' if why == 'boundary' else ''))
    uniform[tag] = labs

# ---- scoring
rng = np.random.default_rng(407)
def credit_table(labels, prec):
    """rows (batch, planet, molecule, label, u) x design credits; entering labels get credit_if_* for the baselines and NaN for the ensemble."""
    rows = []
    for (b, pl, m), lab in labels.items():
        q = A[(A.batch == b) & (A.planet == pl) & (A.molecule == m) & (A.precision == prec)].set_index('design')
        if len(q) == 0: continue
        d = dict(batch=b, planet=pl, molecule=m, label=lab, u=BASE(pl))
        for des in [ENS] + DES:
            if des not in q.index: d[des] = np.nan; continue
            r = q.loc[des]
            if isinstance(r.label, str) and r.label == lab: d[des] = r.credit
            else: d[des] = r.credit_if_absent if lab == 'absent' else r.credit_if_present  # NaN for the ensemble and the independent batch
        rows.append(d)
    return pd.DataFrame(rows)

def compare(T, des):
    v = T[[ENS, des, 'u', 'label']].dropna(); x = (v[ENS] - v[des]).groupby(v.u).sum().to_numpy(); b = x[rng.integers(0, len(x), (20000, len(x)))].sum(1)
    pr, ab = v[v.label == 'present'], v[v.label == 'absent']
    return dict(n=len(v), ens=v[ENS].sum(), base=v[des].sum(), d=x.sum(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), nu=len(x), npr=len(pr), nab=len(ab), ens_p=pr[ENS].sum(), base_p=pr[des].sum(), ens_a=ab[ENS].sum(), base_a=ab[des].sum())

L = [__doc__.strip(), '']
def section(name, labels):
    L.append(f'== {name}')
    cnt = {b: sum(1 for k in labels if k[0] == b) for b in ('panel', 'fourth', 'independent')}; npres = sum(1 for v in labels.values() if v == 'present')
    L.append(f'  labels kept: panel {cnt["panel"]}, fourth {cnt["fourth"]}, independent {cnt["independent"]}; {npres} present, {len(labels) - npres} absent')
    for prec, pools in (('Tier-1', (('panel+fourth', ('panel', 'fourth')), ('panel', ('panel',)), ('fourth', ('fourth',)))), ('JWST', (('all', ('panel', 'fourth', 'independent')), ('outside the panel', ('fourth', 'independent')), ('panel', ('panel',))))):
        T = credit_table(labels, prec)
        for tag, bs in pools:
            q = T[T.batch.isin(bs)]
            for des in DES:
                c = compare(q, des); base_full = q[des].sum(); extra = f'; {des} on all {q[des].notna().sum()} labels incl. those without an ensemble verdict: {base_full:.1f}' if q[des].notna().sum() != c['n'] else ''
                L.append(f'  {prec:<7}{tag:<18}n={c["n"]:<4}planets={c["nu"]:<3}{des:<20}ensemble {c["ens"]:5.1f}  this {c["base"]:5.1f}  diff {c["d"]:+6.2f} ({c["lo"]:+.2f},{c["hi"]:+.2f})'
                         f'  present {c["ens_p"]:4.1f} vs {c["base_p"]:4.1f} of {c["npr"]}  absent {c["ens_a"]:4.1f} vs {c["base_a"]:4.1f} of {c["nab"]}{extra}')
    L.append('')

all_committed = {k: v for k, v in committed.items() if isinstance(v, str)}
section('(a) committed labels: 64 + 24 + 17 (reproduces results/round3/round3_real_new.txt and Table 15)', all_committed)
for name, S in (('(b) 65-label rule-consistent panel set (label_sensitivity.py "corrected") + fourth batch + 17 further', corrected), ('(c) 59-label judgement-free panel set (label_sensitivity.py "strict") + fourth batch + 17 further', strict)):
    labs = {k: v for k, v in all_committed.items() if k[0] != 'panel'}; labs.update({('panel', pl, m): v for (pl, m), v in S.items()}); section(name, labs)
for tag, title in (('literal', '(d) uniform rule, literal: every printed free-chemistry variant on one side of -4 (a limit above -4 counts as crossing)'), ('tolerant', "(d') uniform rule, limits above -4 treated as uninformative rather than crossing")):
    L.append(f'== {title}: cell audit'); L.extend(log[tag]); L.append('')
    section(title, uniform[tag])
L.append('== notes')
L.append('  The ensemble (results/round3/real_C+B-mlp.csv) carries verdicts only on the committed labels, so a label that enters under a rule is scored for the baselines alone and is left out of every paired difference; the baselines\' full-set scores are printed beside.')
L.append('  Variants are pooled per planet across reductions and papers, as the fourth batch did ("any printed free-chemistry variant on the planet\'s JWST data"); the panel\'s committed-unscored cells not listed above stay unscored under both readings (' + STAY + ').')
L.append('  The 17 further labels: the variant lists come from independent_data_survey.md (Tables A1 and B3), not from results/independent_test/, whose SOURCES.md repeats only the V1298 Tau c and HAT-P-14 b values; the survey warns that seven single-planet HST papers were not read, so those cells are kept on the population-study values alone.')
L.append('  Bootstrap: 20,000 resamples of planet units (reductions of one planet kept together), seed 407, sum of per-unit paired differences, as in checks/round3_real_new.py; unlike that script, the HST and JWST reductions of V1298 Tau c form one unit here (47 units over all, not 48), which moves no interval by more than 0.1.')
L.append("  (d') relaxes only the leaving of committed labels; it lets no further cell enter (otherwise the withdrawn K2-18 b H2O labels of the Madhusudhan and Schmidt reductions would re-enter on the Hu reduction's 1-sigma limit alone). The Hu reduction's own H2O label leaves under (d) because the AURA limits in the same paper, -3.52 and -3.57 at 2 sigma, are its reported constraints and lie above -4; it stays under (d').")
t = '\n'.join(L); open('results/round3/real_label_rules.txt', 'w').write(t + '\n'); print(t)
