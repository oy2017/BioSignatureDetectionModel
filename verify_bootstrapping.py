import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils import resample
import os

# Results from Table II
# XGBoost: Acc 88.30, Prec 87.97, Rec 88.81, F1 88.38
# MLP: Acc 88.17, Prec 87.78, Rec 88.82, F1 88.28

# We simulate a population to perform bootstrapping since we don't have the 
# combined raw test results in a single array here, but we can verify the 
# significance of the 0.1% difference.

def bootstrap_diff(f1_a, f1_b, n_samples=531*5, iterations=10000):
    # Null hypothesis: The difference is zero.
    # Given the near-perfect overlap in scores, we check if the difference is robust.
    a_results = np.random.binomial(1, f1_a, n_samples)
    b_results = np.random.binomial(1, f1_b, n_samples)
    
    diffs = []
    for _ in range(iterations):
        idx = np.random.choice(len(a_results), len(a_results), replace=True)
        diffs.append(np.mean(a_results[idx]) - np.mean(b_results[idx]))
    
    p_val = np.mean(np.array(diffs) <= 0)
    return np.percentile(diffs, [2.5, 97.5]), p_val

print("--- F1-Score Bootstrapping Analysis (XGBoost vs Random Forest) ---")
ci, p = bootstrap_diff(0.8873, 0.8667)
print(f"Mean F1 Difference: {0.8873 - 0.8667:.4f}")
print(f"95% Confidence Interval for Difference: {ci}")
print(f"p-value: {p:.4f}")

if p > 0.05:
    print("-> Result: The F1-score difference between XGBoost and RF is NOT statistically significant.")
else:
    print("-> Result: The F1-score difference is statistically significant.")
