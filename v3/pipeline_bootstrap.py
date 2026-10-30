"""Every pipeline on one fixed denominator, with a planet-level bootstrap (replaces the sign tests; review of 2026-10-30).

Scores all labels the literature decides on the real panel, 1 for a right verdict, 0 for a wrong one, and one half for a
split or withheld verdict, so a pipeline cannot improve by leaving labels unscored. Reductions of the same planet are one
bootstrap unit. These are descriptive: the flags and about twenty router variants were chosen on this panel.

Writes results/pipeline_bootstrap.{txt,csv}.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "jwst_panel")
ROWS = [("published classifier", "rr", "none"), ("AIRS-only everywhere", "rr", "AIRS-only everywhere"), ("shape flag -> AIRS-only", "mr", "either -> AIRS-only"),
        ("retrained (haze + deck) everywhere", "mr", "retrained hazedeck everywhere"), ("retrained, shape flag -> its AIRS-only", "mr", "retrained hazedeck everywhere, either -> its AIRS-only"),
        ("census pipeline (needs a composition)", "mr", "muting -> retrained hazedeck, else either -> AIRS-only"), ("scale-height pipeline (haze)", "mr", "proxy -> retrained haze, else either -> AIRS-only"),
        ("scale-height pipeline, two-sided (haze + deck)", "mr", "proxy (two-sided) -> retrained hazedeck, amplified -> withheld, else either -> AIRS-only")]
BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
THIRD = {"WASP-94Ab", "HIP67522b", "HAT-P-12b"}


def main(n=20000):
    src = {"mr": pd.read_csv(os.path.join(OUT, "muting_router.csv")), "rr": pd.read_csv(os.path.join(OUT, "router_real.csv"))}
    cols = {}
    for name, s, r in ROWS:
        q = src[s]; q = q[(q.router == r) & q.label.notna()].set_index(["planet", "molecule"])
        cols[name] = (q.verdict == q.label).astype(float).where(q.verdict != "split", 0.5)
    df = pd.DataFrame(cols); assert not df.isna().any().any(), df[df.isna().any(axis=1)]
    plan = np.array([BASE(p) for p in df.index.get_level_values(0)]); units = sorted(set(plan)); per = df.groupby(plan).sum().loc[units]; cnt = pd.Series(plan).value_counts().loc[units].to_numpy()
    rng = np.random.default_rng(405); idx = rng.integers(0, len(units), (n, len(units)))
    L = [f"Fixed denominator: {len(df)} labelled cells, {len(units)} planets; right 1, split or withheld 0.5, wrong 0; difference in labels on the full-panel scale, 95 % planet-bootstrap interval, {n} resamples; descriptive, the pipelines were chosen on this panel", ""]
    third = df[[p in THIRD for p in df.index.get_level_values(0)]]; rows = []
    L.append(f"{'pipeline':<48} {'score':>6} {'wrong':>6} {'split':>6} {'3rd batch':>10} {'vs published':>22} {'vs AIRS-only':>22}")
    for name, _, _ in ROWS:
        v = df[name]; d = {}
        for ref in ("published classifier", "AIRS-only everywhere"):
            pa, pb = per[name].to_numpy()[idx], per[ref].to_numpy()[idx]; den = cnt[idx].sum(1)
            dif = (pa.sum(1) - pb.sum(1)) / den * len(df); d[ref] = (v.sum() - df[ref].sum(), *np.percentile(dif, [2.5, 97.5]), np.mean(dif > 0))
        rows.append(dict(pipeline=name, score=v.sum(), wrong=int((v == 0).sum()), split=int((v == 0.5).sum()), third=third[name].sum(), **{f"{k} {i}": x for k, t in d.items() for i, x in zip(("diff", "lo", "hi", "p_gt0"), t)}))
        f = lambda t: f"{t[0]:+5.1f} ({t[1]:+5.1f},{t[2]:+5.1f})"
        L.append(f"{name:<48} {v.sum():6.1f} {int((v == 0).sum()):6d} {int((v == 0.5).sum()):6d} {third[name].sum():6.1f}/{len(third):<3d} {f(d['published classifier']):>22} {f(d['AIRS-only everywhere']):>22}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "pipeline_bootstrap.csv"), index=False)
    txt = "\n".join(L); open(os.path.join(RESULTS, "pipeline_bootstrap.txt"), "w").write(txt + "\n"); print(txt)
    print("\nhow each pipeline scores what the third batch contains"); print(third.to_string())


if __name__ == "__main__":
    main()
