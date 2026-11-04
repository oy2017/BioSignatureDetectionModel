"""Crossed (planet x source-model) bootstrap of the ensemble-minus-routing gain on the held-out grids
(REVISION_EXPECTATIONS.md, section AD, rows AD2 and AD3). Output: results/round3/grid_crossed_bootstrap.txt

Credit as in review_checks/goyal_score.py and plots/fig7_grids.py: 1 right, 0.5 split or withheld, 0 wrong;
balanced credit = mean of the pooled credit on truly present (v > -4) and on truly absent labels.
Intervals (95 %, 2,000 replicates, fixed seeds) on the difference C+B-mlp minus 'the method':
  (a) planets only: planets (suffixes -alt, -hu, -hm stripped) resampled with replacement;
  (b) source models only: source atmosphere models (truth column `file`) resampled with replacement;
  (c) crossed (pigeonhole / two-way bootstrap, Owen 2007): independent multinomial counts for planets and for
      source models; each grid row (its four molecules and both designs) is weighted by the product.
"""
import os
import numpy as np, pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = lambda s: s.replace('-alt', '').replace('-hu', '').replace('-hm', '')
MOL = ('CH4', 'H2O', 'CO2', 'NH3'); GR = (('local', 'Local condensation'), ('crossfield', 'Photochemical')); PR = ('JWST', 'Tier-1')
NB = 2000; ENS, RT = 'C+B-mlp', 'the method'


def load(tag):
    Y = pd.read_csv(f'results/goyal_test/truth_{tag}.csv')
    Yl = Y.melt(id_vars=['planet', 'row', 'file', 'T', 'met', 'co'], value_vars=[f'v_{m}' for m in MOL], var_name='molecule', value_name='v'); Yl['molecule'] = Yl.molecule.str[2:]
    V = pd.concat([pd.read_csv(f'results/goyal_test/votes_frozen_{tag}.csv.gz').query('design==@RT'),
                   pd.read_csv(f'results/round3/votes_C+B-mlp_{tag}.csv.gz').drop(columns='prob')])
    A = V.merge(Yl, on=['planet', 'row', 'molecule']); A['y'] = A.v > -4
    A['c'] = np.where(A.y, A.present, A.absent) + .5 * (1 - A.present - A.absent); A['u'] = A.planet.map(BASE)
    W = A.pivot_table(index=['precision', 'planet', 'row', 'molecule'], columns='design', values='c').reset_index()
    W = W.merge(A[A.design == RT][['precision', 'planet', 'row', 'molecule', 'y', 'u', 'file', 'T', 'met', 'co']], on=['precision', 'planet', 'row', 'molecule'])
    assert W[[ENS, RT]].notna().all().all()
    return Y, W


class Boot:
    """Precomputed multinomial weights for planets and source models of one grid (shared across all subsets)."""
    def __init__(self, W, seed):
        self.us = sorted(W.u.unique()); self.fs = sorted(W.file.unique())
        r = np.random.default_rng(seed)
        self.Wu_a = self._counts(r, len(self.us)); self.Wf_b = self._counts(r, len(self.fs))
        self.Wu_c = self._counts(r, len(self.us)); self.Wf_c = self._counts(r, len(self.fs))
        self.ui = {u: i for i, u in enumerate(self.us)}; self.fi = {f: i for i, f in enumerate(self.fs)}

    @staticmethod
    def _counts(r, n):
        ix = r.integers(0, n, (NB, n)); out = np.zeros((NB, n), np.float32)
        np.add.at(out, (np.repeat(np.arange(NB), n), ix.ravel()), 1); return out

    def run(self, q):
        """q: label rows of one subset. Returns point diff, point values, and (lo, hi) for a, b, c."""
        g = q.groupby(['u', 'file', 'y'])[[ENS, RT]].agg(['sum', 'count'])
        g.columns = ['es', 'n', 'rs', 'n2']; g = g.reset_index()
        cells = g.pivot_table(index=['u', 'file'], columns='y', values=['es', 'rs', 'n'], fill_value=0)
        ui = np.array([self.ui[u] for u in cells.index.get_level_values(0)]); fi = np.array([self.fi[f] for f in cells.index.get_level_values(1)])
        col = lambda k, y: cells[(k, y)].to_numpy(float) if (k, y) in cells.columns else np.zeros(len(cells))
        S = np.stack([col('es', True), col('rs', True), col('n', True), col('es', False), col('rs', False), col('n', False)])  # 6 x K

        def bal(w):  # w: R x K weights
            with np.errstate(all='ignore'):  # macOS Accelerate BLAS raises spurious overflow warnings; checked equal to einsum
                t = w @ S.T  # R x 6
            e = .5 * (t[:, 0] / t[:, 2] + t[:, 3] / t[:, 5]); r = .5 * (t[:, 1] / t[:, 2] + t[:, 4] / t[:, 5]); return e, r
        e0, r0 = bal(np.ones((1, len(cells))))
        out = {'n': int(S[2].sum() + S[5].sum()), 'ens': e0[0], 'rt': r0[0], 'd': e0[0] - r0[0], 'nu': len(set(ui)), 'nf': len(set(fi))}
        for k, w in (('a', self.Wu_a[:, ui]), ('b', self.Wf_b[:, fi]), ('c', self.Wu_c[:, ui] * self.Wf_c[:, fi])):
            with np.errstate(invalid='ignore', divide='ignore'):
                e, r = bal(w.astype(float))
            d = (e - r)[np.isfinite(e - r)]; out[k] = tuple(np.percentile(d, [2.5, 97.5])); out[k + '_nan'] = NB - len(d)
        return out


def fmt(o, ks='abc'):
    s = f"ens {o['ens']:.3f}  routing {o['rt']:.3f}  diff {o['d']:+.3f}"
    for k in ks:
        lo, hi = o[k]; s += f"  ({k}) {lo:+.3f},{hi:+.3f} w={hi - lo:.3f}"
    return s


def tercile_bins(x, name):
    """Value bins with ties kept together, cut where the cumulative row share crosses 1/3 and 2/3."""
    if name == 'co':  # 78 % of photochemical rows sit at C/O = 0.55: sub-solar / 0.55 / super-solar
        return np.select([x < .549, x < .551], ['<0.55', '=0.55'], '>0.55')
    v = np.sort(np.unique(x)); share = np.array([(x <= t).mean() for t in v])
    c1 = v[np.argmin(np.abs(share - 1 / 3))]; c2 = v[np.argmin(np.abs(share - 2 / 3))]
    lab = lambda lo, hi: f'{lo:g}' if lo == hi else f'{lo:g}-{hi:g}'
    vl, vm, vh = v[v <= c1], v[(v > c1) & (v <= c2)], v[v > c2]
    return np.select([x <= c1, x <= c2], [lab(vl.min(), vl.max()), lab(vm.min(), vm.max())], lab(vh.min(), vh.max()) if len(vh) else 'none')


L = ['Crossed bootstrap of the ensemble (C+B-mlp) minus routing method gain on the held-out grids (AD2, AD3).',
     f'95 % percentile intervals, {NB} replicates; (a) planets, (b) source models, (c) crossed planets x source models (product of multinomial counts). w = width.']
T13 = {('local', 'JWST'): (.711, .574), ('local', 'Tier-1'): (.636, .531), ('crossfield', 'JWST'): (.773, .658), ('crossfield', 'Tier-1'): (.700, .567)}
EXIST = {('local', 'JWST'): (.110, .156), ('local', 'Tier-1'): (.089, .119), ('crossfield', 'JWST'): (.089, .136), ('crossfield', 'Tier-1'): (.120, .144)}
pooled, gas, terc, reuse = [], [], [], []
for gi, (tag, gname) in enumerate(GR):
    Y, W = load(tag); B = Boot(W, 4050 + gi)
    vc = Y.file.value_counts(); vcu = Y.assign(u=Y.planet.map(BASE)).groupby('file').u.nunique()
    reuse.append(f'  {gname:<19} {len(Y)} grid rows, {Y.planet.nunique()} reductions of {Y.planet.map(BASE).nunique()} planets, {len(vc)} distinct source models; '
                 f'rows per model mean {vc.mean():.2f}, median {vc.median():.0f}, max {vc.max()}; models used once {(vc == 1).sum()} ({(vc == 1).mean() * 100:.0f} %); '
                 f'models used by >1 planet {(vcu > 1).sum()} ({(vcu > 1).mean() * 100:.0f} %), planets per model max {vcu.max()}; '
                 f'rows-per-model histogram {dict(sorted(vc.value_counts().to_dict().items()))}')
    for p in PR:
        q = W[W.precision == p]; o = B.run(q); te, tr = T13[(tag, p)]; ex = EXIST[(tag, p)]
        ok = abs(round(o['ens'], 3) - te) < 1e-9 and abs(round(o['rt'], 3) - tr) < 1e-9
        pooled.append(f"  {gname:<19}{p:<7} n={o['n']:<6} {fmt(o)}  ratio c/a {(o['c'][1] - o['c'][0]) / (o['a'][1] - o['a'][0]):.2f}  c/b {(o['c'][1] - o['c'][0]) / (o['b'][1] - o['b'][0]):.2f}"
                      f"\n  {'':<26}Table 13 {te:.3f} vs {tr:.3f}: {'reproduced' if ok else 'NOT reproduced'}; existing planet interval ({ex[0]:+.3f},{ex[1]:+.3f}) w={ex[1] - ex[0]:.3f}; crossed excludes zero: {o['c'][0] > 0 or o['c'][1] < 0}")
        for m in MOL:
            om = B.run(q[q.molecule == m]); z = om['c'][0] <= 0 <= om['c'][1]
            gas.append((z, f"  {gname:<19}{p:<7}{m:<4} {fmt(om)}  c/a {(om['c'][1] - om['c'][0]) / (om['a'][1] - om['a'][0]):.2f}{'  CROSSED INCLUDES ZERO' if z else ''}"))
        if tag == 'crossfield':
            for k, kname in (('T', 'temperature T'), ('met', 'metallicity met'), ('co', 'C/O co')):
                b = tercile_bins(q[k].to_numpy(), k); order = sorted(set(b), key=lambda s: q[k][b == s].min())
                for s in order:
                    qs = q[b == s]; os_ = B.run(qs)
                    terc.append(f"  {p:<7}{kname:<16}{s:<12} n={os_['n']:<6} planets {os_['nu']:<3} models {os_['nf']:<4} {fmt(os_, 'ac')}{'  gain <= 0' if os_['d'] <= 0 else ''}{'  crossed includes zero' if os_['c'][0] <= 0 <= os_['c'][1] else ''}")
                if k == 'met':  # also each metallicity value separately
                    for mv in sorted(q.met.unique()):
                        os_ = B.run(q[q.met == mv]); terc.append(f"  {p:<7}{'  met value':<16}{mv:<12g} n={os_['n']:<6} planets {os_['nu']:<3} models {os_['nf']:<4} {fmt(os_, 'ac')}")
L += ['', '== 1. Pooled gain, each grid x precision (point values against Table 13)'] + pooled
nz = [s for z, s in gas if z]
L += ['', '== 2. Per gas'] + [s for _, s in gas] + [f'  per-gas crossed intervals including zero: {len(nz)} of {len(gas)}']
L += ['', '== 3. Photochemical grid: gain within terciles of source-model T, met, C/O (ties kept together; C/O split <0.55 / 0.55 / >0.55 because 78 % of rows sit at 0.55)'] + terc
L += ['', '== 4. Source-model reuse'] + reuse
t = '\n'.join(L); open('results/round3/grid_crossed_bootstrap.txt', 'w').write(t + '\n'); print(t)
