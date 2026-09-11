"""Fill {{placeholders}} in manuscript.md from v2/results and v2/data.

Only placeholders whose source exists are filled; the rest are left for a
later run. Prints what remains. Run from nhsjs/: python fill_numbers.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), "v2")
RES = os.path.join(V2, "results")
DATA = os.path.join(V2, "data")


def numbers():
    n = {}
    # grid
    tr = pd.read_parquet(os.path.join(DATA, "train_params.parquet"))
    tests = [len(pd.read_parquet(os.path.join(DATA, f"test{k}_params.parquet"))) for k in range(1, 6)]
    n["n_train"] = f"{len(tr):,}"
    n["n_test_list"] = ", ".join(f"{t:,}" for t in tests[:-1]) + f" and {tests[-1]:,}"
    bulk = ["p_radius", "p_mass", "atm temperature", "atm base_pressure", "atm top_pressure",
            "s temperature", "s radius", "s mass", "sma"]
    c = tr[bulk].corr().abs().to_numpy(); np.fill_diagonal(c, 0)
    n["max_bulk_r"] = f"{c.max():.2f}"
    n["ch4_o3_r"] = f"{tr['atm CH4'].corr(tr['atm O3']):.2f}"
    g = open(os.path.join(RES, "generation.txt")).read()
    dec = [float(x) for x in re.findall(r"decile \d+: ([0-9.]+)", g)]
    kept = (len(tr) + sum(tests)) / 30000
    n["pct_kept"] = f"{kept*100:.0f}"
    n["decile0_kept"] = f"{dec[0]*100:.0f}"
    n["decile1_kept"] = f"{min(dec[1:])*100:.0f}"
    # tuning grids
    import sys
    sys.path.insert(0, V2)
    from pipeline import RF_GRID, XGB_GRID
    n["n_xgb_grid"] = str(int(np.prod([len(v) for v in XGB_GRID.values()])))
    n["n_rf_grid"] = str(int(np.prod([len(v) for v in RF_GRID.values()])))
    # pipeline summary
    p = os.path.join(RES, "ariel_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        n["k_pca"] = str(s["feature_dims"]["pca"])
        for k, m in s["models"].items():
            n[f"{k}_acc"] = f"{m['accuracy'][0]*100:.1f}"
            n[f"{k}_acc_sd"] = f"{m['accuracy'][1]*100:.1f}"
            n[f"{k}_brier"] = f"{m['brier'][0]:.3f}"
            n[f"{k}_ece"] = f"{m['ece'][0]:.3f}"
            n[f"{k}_auc"] = f"{m['auc'][0]:.3f}"
            n[f"{k}_f1"] = f"{m['f1'][0]*100:.1f}"
        n["best_key"] = s["best"]
    for cfg in ("r100", "r200"):
        p = os.path.join(RES, f"{cfg}_summary.json")
        if os.path.exists(p):
            s = json.load(open(p)); m = s["models"][s["best"]]
            n[f"{cfg}_best_acc"] = f"{m['accuracy'][0]*100:.1f}"
            n[f"{cfg}_best_key"] = s["best"]
            n[f"{cfg}_k_pca"] = str(s["feature_dims"]["pca"])
    return n


def main():
    p = os.path.join(HERE, "manuscript.md")
    s = open(p, encoding="utf8").read()
    n = numbers()
    filled = 0
    for k, v in n.items():
        tag = "{{" + k + "}}"
        if tag in s:
            s = s.replace(tag, v); filled += 1
    open(p, "w", encoding="utf8").write(s)
    left = sorted(set(re.findall(r"\{\{([a-z0-9_]+)\}\}", s)))
    print(f"filled {filled}; remaining placeholders: {left}")


if __name__ == "__main__":
    main()
