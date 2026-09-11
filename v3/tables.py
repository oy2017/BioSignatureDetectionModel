"""Assemble the manuscript tables as Markdown from the result files.

  Table 2  model x preprocessing at the Ariel configuration (results/ariel_summary.json)
  ladder   accuracy at ariel / r100 / r200 for the frozen pipeline family (results/{cfg}_summary.json)
  Table 3  seven-axis shift map (results/ariel_shifts.csv)

Usage: python tables.py  -> results/tables.md
"""
import json
import os

import pandas as pd

from common import RESULTS

LABELS = {"raw_xgb": "XGBoost, raw bins", "pca_xgb": "XGBoost, PCA", "norm_xgb": "XGBoost, per-spectrum normalized",
          "raw_rf": "Random Forest, raw bins", "pca_rf": "Random Forest, PCA", "norm_rf": "Random Forest, per-spectrum normalized",
          "pcaw_mlp": "MLP, PCA whitened", "pca_mlp": "MLP, PCA unwhitened"}


def table2(cfg="ariel"):
    s = json.load(open(os.path.join(RESULTS, f"{cfg}_summary.json")))
    rows = ["| Model, features | Accuracy (%) | F1 (%) | Brier | ECE | AUC |", "| :-- | --: | --: | --: | --: | --: |"]
    for k, m in s["models"].items():
        rows.append(f"| {LABELS.get(k, k)} | {m['accuracy'][0]*100:.2f} ± {m['accuracy'][1]*100:.2f} | "
                    f"{m['f1'][0]*100:.2f} ± {m['f1'][1]*100:.2f} | {m['brier'][0]:.4f} | {m['ece'][0]:.3f} | {m['auc'][0]:.3f} |")
    dims = ", ".join(f"{k} {v}" for k, v in s["feature_dims"].items())
    return "\n".join(rows) + f"\n\nFeature dimensions: {dims}. n_train = {s['n_train']}. Best: {s['best']}.\n"


def ladder():
    rows = ["| Configuration | Bins | XGBoost, normalized (%) | MLP, normalized (%) | XGBoost, PCA (%) | Brier (XGBoost, normalized) | ECE |",
            "| :-- | --: | --: | --: | --: | --: | --: |"]
    names = {"ariel": "Ariel delivered (photometry + R 15 / 100 / 30)", "r100": "uniform R = 100", "r200": "uniform R = 200 (earlier study)"}
    for cfg in ("ariel", "r100", "r200"):
        p = os.path.join(RESULTS, f"{cfg}_summary.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))["models"]
        nb = {"ariel": 102, "r100": 275, "r200": 550}[cfg]
        rows.append(f"| {names[cfg]} | {nb} | {m['norm_xgb']['accuracy'][0]*100:.2f} ± {m['norm_xgb']['accuracy'][1]*100:.2f} | "
                    f"{m['norm_mlp']['accuracy'][0]*100:.2f} ± {m['norm_mlp']['accuracy'][1]*100:.2f} | "
                    f"{m['pca_xgb']['accuracy'][0]*100:.2f} ± {m['pca_xgb']['accuracy'][1]*100:.2f} | "
                    f"{m['norm_xgb']['brier'][0]:.4f} | {m['norm_xgb']['ece'][0]:.3f} |")
    return "\n".join(rows) + "\n"


def table3(cfg="ariel"):
    df = pd.read_csv(os.path.join(RESULTS, f"{cfg}_shifts.csv"))
    rows = ["| Axis | Case | Δ accuracy (pts) | Δ Brier | Predicted positive rate | Amplitude ratio |",
            "| :-- | :-- | --: | --: | --: | --: |"]
    for _, r in df.iterrows():
        rows.append(f"| {r['axis']} | {r['case']} | {r['d_acc']:+.1f} | {r['d_brier']:+.3f} | {r['pos_rate']:.3f} | {r['amp_ratio']:.2f} |")
    return "\n".join(rows) + "\n"


def main():
    out = ["## Table 2", table2(), "## Resolution ladder", ladder()]
    if os.path.exists(os.path.join(RESULTS, "ariel_shifts.csv")):
        out += ["## Table 3", table3()]
    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
