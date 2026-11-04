"""AD5 (REVISION_EXPECTATIONS.md, section AD; post hoc): the frozen four-learner vote on the broad population with balanced
classes (B+bal-vote), scored on the two held-out grids exactly as Table 13 (plots/fig7_grids.py, checks/goyal_score.py):
balanced credit pooled over the four gases (right 1, split 1/2, wrong 0; mean of present and absent shares), with the
credit on present labels and the false-present rate; 95 % bootstrap over planets (base planet, 4000 resamples, seed 405).
First reproduces Table 13 for the routing method ("the method") and the matched ensemble (C+B-mlp).

Inputs: results/goyal_test/{votes_frozen_<g>,truth_<g>}, results/round3/votes_{C+B-mlp,B+bal-vote}_<g>.csv.gz
(the B+bal-vote held-out votes from `round3_learner.py --cand B+bal-vote --cond local crossfield`).
Writes results/round3/ablation_vote.txt.
"""
import os

import numpy as np
import pandas as pd

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(V3)
MOLS = ["CH4", "H2O", "CO2", "NH3"]; BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
DES = {"the method": "routing method", "C+B-mlp": "matched ensemble", "B+bal-vote": "B+bal-vote"}
TABLE13 = {("local", "JWST"): {"the method": (0.574, 0.31, 0.13), "C+B-mlp": (0.711, 0.60, 0.17)}, ("local", "Tier-1"): {"the method": (0.531, 0.14, 0.05), "C+B-mlp": (0.636, 0.45, 0.17)},
           ("crossfield", "JWST"): {"the method": (0.658, 0.40, 0.06), "C+B-mlp": (0.773, 0.63, 0.08)}, ("crossfield", "Tier-1"): {"the method": (0.567, 0.18, 0.03), "C+B-mlp": (0.700, 0.52, 0.12)},
           ("rainout", "JWST"): {"the method": (0.643, 0.39, 0.07), "C+B-mlp": (0.765, 0.64, 0.11)}, ("rainout", "Tier-1"): {"the method": (0.566, 0.18, 0.03), "C+B-mlp": (0.698, 0.51, 0.11)}}
GRIDS = [("local", "local condensation"), ("crossfield", "photochemical"), ("rainout", "rainout (development)")]


def load(tag):
    V = pd.concat([pd.read_csv(f"results/goyal_test/votes_frozen_{tag}.csv.gz")] + [pd.read_csv(f"results/round3/votes_{d}_{tag}.csv.gz").drop(columns="prob") for d in ("C+B-mlp", "B+bal-vote")]); V = V[V.design.isin(DES)]
    Y = pd.read_csv(f"results/goyal_test/truth_{tag}.csv").melt(id_vars=["planet", "row"], value_vars=[f"v_{m}" for m in MOLS], var_name="molecule", value_name="v"); Y["molecule"] = Y.molecule.str[2:]
    A = V.merge(Y, on=["planet", "row", "molecule"]); A["y"] = A.v > -4; A["c"] = np.where(A.y, A.present, A.absent) + 0.5 * (1 - A.present - A.absent); A["u"] = A.planet.map(BASE); return A


def score(q, ix):
    """{design: (point, bootstrap array, present credit, false-present)} with planets as units."""
    us = sorted(q.u.unique()); S = q.groupby(["design", "u", "y"]).c.agg(["sum", "count"]); out = {}
    for d in DES:
        p = S.loc[d].unstack("y").reindex(us).fillna(0); a, b, c, e = (p["sum"][True].to_numpy(), p["count"][True].to_numpy(), p["sum"][False].to_numpy(), p["count"][False].to_numpy())
        F = lambda i: 0.5 * (a[i].sum(-1) / b[i].sum(-1) + c[i].sum(-1) / e[i].sum(-1)); g = q[q.design == d]
        out[d] = (F(np.arange(len(us))), F(ix), a.sum() / b.sum(), g[~g.y].present.mean(), g[g.y].present.mean())
    return out


def main():
    L = ["AD5: B+bal-vote, frozen, on the held-out grids (Table 13 scoring). Balanced credit (present credit, false-present); 95 % bootstrap over planets.", ""]; summ = []
    for tag, gname in GRIDS:
        A = load(tag); nu = A.u.nunique(); ix = np.random.default_rng(405).integers(0, nu, (4000, nu))
        for prec in ("JWST", "Tier-1"):
            q = A[A.precision == prec]; R = score(q, ix); L.append(f"== {gname}, {prec}: {len(q[q.design == 'the method'])} labels, {nu} planets, {100 * q[q.design == 'the method'].y.mean():.0f} % present")
            for d, lab in DES.items():
                pt, bs, pc, fp, rec = R[d]; ref = TABLE13[(tag, prec)].get(d); chk = "" if ref is None else f"   Table 13 {ref[0]:.3f} ({ref[1]:.2f}, {ref[2]:.2f}) {'reproduced' if (round(pt, 3), round(pc, 2), round(fp, 2)) == ref else 'NOT REPRODUCED'}"
                L.append(f"  {lab:<18} {pt:.3f} ({np.percentile(bs, 2.5):.3f} to {np.percentile(bs, 97.5):.3f})  present credit {pc:.3f}  false-present {fp:.3f}{chk}")
            e_, v_, r_ = R["C+B-mlp"], R["B+bal-vote"], R["the method"]; dv = e_[1] - v_[1]; dr = v_[1] - r_[1]; de = e_[1] - r_[1]
            L.append(f"  ensemble minus B+bal-vote  {e_[0] - v_[0]:+.3f} ({np.percentile(dv, 2.5):+.3f} to {np.percentile(dv, 97.5):+.3f})")
            L.append(f"  B+bal-vote minus routing   {v_[0] - r_[0]:+.3f} ({np.percentile(dr, 2.5):+.3f} to {np.percentile(dr, 97.5):+.3f})")
            L.append(f"  ensemble minus routing     {e_[0] - r_[0]:+.3f} ({np.percentile(de, 2.5):+.3f} to {np.percentile(de, 97.5):+.3f})")
            L.append("  per gas, balanced credit (routing / B+bal-vote / ensemble): " + ", ".join(f"{m} " + "/".join(f"{score(q[q.molecule == m], ix[:1])[d][0]:.3f}" for d in DES) for m in MOLS))
            if tag != "rainout": summ.append((gname, prec, r_[0], v_[0], e_[0], e_[0] - v_[0], np.percentile(dv, 2.5), np.percentile(dv, 97.5)))
            L.append("")
    L.append("AD5 check (held-out grids): routing < B+bal-vote < ensemble, and ensemble - B+bal-vote >= 0.03")
    for g, p, r, v, e, d, lo, hi in summ:
        L.append(f"  {g:<20} {p:<7} routing {r:.3f}  vote {v:.3f}  ensemble {e:.3f}  between: {'yes' if r < v < e else 'no'}  gap {d:+.3f} ({lo:+.3f} to {hi:+.3f}) >= 0.03: {'yes' if d >= 0.03 else 'no'}")
    n_between = sum(r < v < e for _, _, r, v, e, *_ in summ); n_gap = sum(d >= 0.03 for *_, d, _, _ in summ)
    L.append(f"  between in {n_between} of 4; gap >= 0.03 in {n_gap} of 4 -> AD5 {'met' if n_between == 4 and n_gap >= 3 else 'not met'}")
    t = "\n".join(L); open("results/round3/ablation_vote.txt", "w").write(t + "\n"); print(t)


if __name__ == "__main__":
    main()
