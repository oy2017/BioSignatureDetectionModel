"""
Evaluate the frozen pipeline on opacity-swapped spectra (R1-3 axis 2).

Answers Reviewer 1's request for validation against "alternative molecular
opacity databases". H2O, CH4 and CO2 tables are replaced with ExoMol line
lists (POKAZATEL, YT34to10, UCL-4000); O3 and O2 retain their Exo-Transmit
tables because ExoMolOP provides no ozone opacity. O3 is label-determining,
so the measured effect is a LOWER BOUND on a complete database change.

The comparison is paired: every swapped planet is the same planet as its
baseline counterpart, regenerated with different opacity tables and nothing
else changed (see generate_opacity_swap_testset.py). That removes the sampling
drift that makes freshly drawn sets incomparable, and allows per-planet
statistics - how many individual planets flip their prediction - which a
distribution-level comparison cannot provide.

Same frozen-pipeline invariants as the other evaluations: scalers and PCA fit
on the clean training set only, models trained once and never retrained.

Run generate_opacity_swap_testset.py --generate first.

Usage:
    python evaluate_opacity_swap.py
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 102
TRAIN_FILE = "multirex_spectra_H2_train.parquet"
BASE_FMT = "multirex_spectra_H2_test_set_{}.parquet"
SWAP_FMT = "multirex_spectra_H2_opacityswap_set_{}.parquet"


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


def score(y, p):
    pred = (p > 0.5).astype(int)
    return (accuracy_score(y, pred), f1_score(y, pred, zero_division=0),
            brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def main():
    df_tr = pd.read_parquet(TRAIN_FILE)
    cols = spectral_cols(df_tr)
    X_tr = df_tr[cols].values
    y_tr = (df_tr["biosignature"] == "yes").astype(int).values

    scaler_raw = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(
        scaler_raw.transform(X_tr))
    Xs, ys = shuffle(pca.transform(scaler_raw.transform(X_tr)), y_tr,
                     random_state=SEED)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.2,
                        subsample=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=-1).fit(Xs, ys)
    amp_ref = np.median(X_tr.std(axis=1) / X_tr.mean(axis=1))

    def predict(df):
        X = df[spectral_cols(df)].values
        p = xgb.predict_proba(pca.transform(scaler_raw.transform(X)))[:, 1]
        return X, (df["biosignature"] == "yes").astype(int).values, p

    rows, pooled = [], {"y": [], "pb": [], "ps": []}
    for i in range(1, 6):
        base = pd.read_parquet(BASE_FMT.format(i))
        swap = pd.read_parquet(SWAP_FMT.format(i))
        # Pairing is by construction (same rows, same order); verify it held.
        assert len(base) == len(swap), f"set {i}: row count differs"
        assert np.allclose(base["p_radius"].values, swap["p_radius"].values), \
            f"set {i}: planets are not aligned - pairing broken"

        Xb, yb, pb = predict(base)
        Xs_, ysw, ps = predict(swap)
        assert (yb == ysw).all(), f"set {i}: labels differ between arms"

        ab, _, bb = score(yb, pb)
        asw, _, bsw = score(ysw, ps)
        amp_b = np.median(Xb.std(axis=1) / Xb.mean(axis=1)) / amp_ref
        amp_s = np.median(Xs_.std(axis=1) / Xs_.mean(axis=1)) / amp_ref
        flips = ((pb > 0.5) != (ps > 0.5)).mean()
        rows.append((i, len(base), ab, asw, bb, bsw, amp_b, amp_s, flips,
                     np.median(np.abs(ps - pb))))
        pooled["y"].append(yb); pooled["pb"].append(pb); pooled["ps"].append(ps)

    y = np.concatenate(pooled["y"])
    pb = np.concatenate(pooled["pb"])
    ps = np.concatenate(pooled["ps"])
    ab, fb, bb = score(y, pb)
    asw, fsw, bsw = score(y, ps)
    flips = ((pb > 0.5) != (ps > 0.5))

    lines = [
        "Frozen pipeline on opacity-swapped spectra (XGBoost)",
        "",
        "H2O -> ExoMol POKAZATEL, CH4 -> ExoMol YT34to10, CO2 -> ExoMol",
        "UCL-4000; O3 and O2 retain their Exo-Transmit tables (ExoMolOP has no",
        "ozone). O3 is label-determining, so this is a LOWER BOUND on the",
        "effect of a complete opacity-database change.",
        "",
        "Paired: each swapped planet is the same planet as its baseline",
        "counterpart, regenerated with different opacity tables only.",
        "",
        f"{'set':>4} {'n':>6} {'base acc':>9} {'swap acc':>9} {'delta':>7} "
        f"{'base Brier':>11} {'swap Brier':>11} {'amp b/s':>12} {'flipped':>8}",
        "-" * 92]
    for i, n, a1, a2, b1, b2, m1, m2, fl, dp in rows:
        lines.append(f"{i:>4} {n:>6} {a1:>9.2%} {a2:>9.2%} {a2 - a1:>+7.2%} "
                     f"{b1:>11.4f} {b2:>11.4f} {m1:>5.2f}/{m2:<6.2f} {fl:>8.1%}")
    lines += [
        "-" * 92,
        f"{'all':>4} {len(y):>6} {ab:>9.2%} {asw:>9.2%} {asw - ab:>+7.2%} "
        f"{bb:>11.4f} {bsw:>11.4f} {'':>12} {flips.mean():>8.1%}",
        "",
        "=" * 72,
        f"Pooled: accuracy {ab:.2%} -> {asw:.2%} ({(asw - ab) * 100:+.2f} "
        f"points), F1 {fb:.3f} -> {fsw:.3f}, Brier {bb:.4f} -> {bsw:.4f}",
        f"Predicted-positive rate {(pb > 0.5).mean():.1%} -> "
        f"{(ps > 0.5).mean():.1%} (true positive rate {y.mean():.1%}); "
        f"mean predicted probability {pb.mean():.3f} -> {ps.mean():.3f}",
        f"Individual predictions changed for {flips.sum()} of {len(y)} planets "
        f"({flips.mean():.1%}).",
        f"Median |change| in predicted probability: {np.median(np.abs(ps - pb)):.4f}",
        "",
        "Of the planets whose prediction flipped:",
        f"  {(flips & (pb > 0.5) & (y == 1)).sum():>4} were correct positives "
        f"that became negative",
        f"  {(flips & (pb <= 0.5) & (y == 0)).sum():>4} were correct negatives "
        f"that became positive",
        f"  {(flips & (pb > 0.5) & (y == 0)).sum():>4} were incorrect positives "
        f"that became correct",
        f"  {(flips & (pb <= 0.5) & (y == 1)).sum():>4} were incorrect negatives "
        f"that became correct",
    ]

    out = "\n".join(lines)
    print(out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_opacity_swap.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["set", "n", "base_acc", "swap_acc", "base_brier",
                                "swap_brier", "base_amp", "swap_amp",
                                "flip_fraction", "median_prob_change"]).to_csv(
        "final_results/H2_opacity_swap.csv", index=False)
    print("\nWrote final_results/H2_opacity_swap.{txt,csv}")


if __name__ == "__main__":
    main()
