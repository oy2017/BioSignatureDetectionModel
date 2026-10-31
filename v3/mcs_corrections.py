"""Corrections to data/mcs/mcs_params.parquet found in the pipeline audit (REVISION_EXPECTATIONS.md, section M). Idempotent; run once after the catalogue is built."""
import os

import pandas as pd

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mcs", "mcs_params.parquet"); FIX = {"HIP67522b": 13.8}   # Earth masses: Thao et al. 2024

M = pd.read_parquet(P)
for n, m in FIX.items():
    k = M.name == n; M.loc[k, "modulation_5H"] = M.loc[k, "modulation_5H"] * M.loc[k, "p_mass"] / m; M.loc[k, "p_mass"] = m
M.to_parquet(P); print(M[M.name.isin(FIX)][["name", "p_mass", "modulation_5H"]])
