"""
Whitened-vs-unwhitened MLP robustness, averaged over training restarts.

The R1-8 whitening result (crossover and headroom table) came from a single
training of each network, but run-to-run scatter is +/-1.7 points for the
whitened architecture and +/-5.6 for the unwhitened one, so the quoted gap
and crossover carried no error bars. This retrains BOTH sweep MLPs
N_RESTARTS times on identical frozen features and re-evaluates every
perturbation family, reporting each quantity as mean +/- sd with the
whitened-minus-unwhitened differences computed pairwise within restarts.

Perturbations are generated once with the sweep's own perturb() and rng
seed, identical across restarts, so training is the only varying factor.
XGBoost is deterministic and not re-evaluated.

Usage:
    python domain_shift_mlp_restarts.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from domain_shift_sweep import (N_COMPONENTS, SEED, SWEEPS, build_mlp,
                                load_raw, perturb, score)

N_RESTARTS = 5

# Strength at which each family is unperturbed (perturb() returns X.copy()).
CLEAN_STRENGTH = {"white noise": 15, "correlated noise": 15, "gain ramp": 0.0,
                  "baseline offset": 0.0, "resolution loss": 0,
                  "stellar contamination": 0.0}


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")

    X_train, y_train, tests, wl, _, _ = load_raw()
    scaler_raw = StandardScaler().fit(X_train)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(
        scaler_raw.transform(X_train))
    P_train = pca.transform(scaler_raw.transform(X_train))
    scaler_pca = StandardScaler().fit(P_train)

    # Precompute frozen-pipeline features for every perturbed condition once;
    # identical across restarts (same rng seed as the original sweep).
    print("precomputing perturbed features...", flush=True)
    cond_feats = {}
    for family, strengths in SWEEPS.items():
        for s in strengths:
            rng = np.random.default_rng(SEED)
            cond_feats[(family, s)] = [
                (pca.transform(scaler_raw.transform(
                    perturb(X_raw, wl, family, s, rng))), y)
                for X_raw, y in tests]

    acc = {}   # (family, strength, model) -> list over restarts
    for r in range(N_RESTARTS):
        rs = 1000 + r
        tf.random.set_seed(rs)

        def fit(P):
            Xs, ys = shuffle(P, y_train, random_state=rs)
            m = build_mlp(N_COMPONENTS)
            m.fit(Xs, ys, epochs=200, batch_size=128, validation_split=0.2,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                           restore_best_weights=True)],
                  verbose=0)
            return m

        mlp_w = fit(scaler_pca.transform(P_train))
        mlp_u = fit(P_train)
        for (family, s), sets in cond_feats.items():
            for name, mdl, prep in [("w", mlp_w, scaler_pca.transform),
                                    ("u", mlp_u, lambda P: P)]:
                a = [score(y, mdl.predict(prep(P), verbose=0).ravel())[0]
                     for P, y in sets]
                acc.setdefault((family, s, name), []).append(np.mean(a))
        print(f"restart {r + 1}/{N_RESTARTS}: clean "
              f"W {acc[('white noise', 15, 'w')][-1]:.2%} "
              f"U {acc[('white noise', 15, 'u')][-1]:.2%}", flush=True)

    lines = [f"Whitened vs unwhitened MLP under the domain-shift sweep, "
             f"{N_RESTARTS} training restarts each",
             "",
             "Same frozen preprocessing and perturbations as "
             "domain_shift_sweep.py; only network training varies between "
             "restarts. Differences are paired within restarts.", ""]
    rows = []
    for family, strengths in SWEEPS.items():
        lines.append(f"--- {family} ---")
        lines.append(f"{'strength':>10} {'whitened':>16} {'unwhitened':>16} "
                     f"{'W - U (paired)':>16}")
        for s in strengths:
            w = np.array(acc[(family, s, "w")])
            u = np.array(acc[(family, s, "u")])
            d = w - u
            lines.append(f"{s:>10} {w.mean():>8.2%} ±{w.std():>5.2%} "
                         f"{u.mean():>8.2%} ±{u.std():>5.2%} "
                         f"{d.mean():>+8.2%} ±{d.std():>5.2%}")
            rows.append((family, s, w.mean(), w.std(), u.mean(), u.std(),
                         d.mean(), d.std()))
        # headroom-above-chance fraction lost at the strongest setting
        cw = np.array(acc[(family, CLEAN_STRENGTH[family], "w")])
        cu = np.array(acc[(family, CLEAN_STRENGTH[family], "u")])
        sw = np.array(acc[(family, strengths[-1], "w")])
        su = np.array(acc[(family, strengths[-1], "u")])
        hw = (cw - sw) / (cw - 0.5)
        hu = (cu - su) / (cu - 0.5)
        lines.append(f"{'headroom lost':>10}  W {hw.mean():.0%} ±{hw.std():.0%}"
                     f"   U {hu.mean():.0%} ±{hu.std():.0%}")
        lines.append("")

    # Crossover in the white-noise family, per restart.
    lines.append("White-noise crossover (first strength where unwhitened "
                 "leads), per restart:")
    xs = []
    for r in range(N_RESTARTS):
        cross = next((s for s in SWEEPS["white noise"]
                      if acc[("white noise", s, "u")][r]
                      > acc[("white noise", s, "w")][r]), None)
        xs.append(cross)
        lines.append(f"  restart {r + 1}: SNR {cross}")
    lines.append("")

    out = "\n".join(lines)
    print("\n" + out)
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/H2_whitening_restarts.txt", "w") as fh:
        fh.write(out + "\n")
    pd.DataFrame(rows, columns=["family", "strength", "w_mean", "w_sd",
                                "u_mean", "u_sd", "diff_mean", "diff_sd"]
                 ).to_csv("final_results/H2_whitening_restarts.csv",
                          index=False)
    print("Wrote final_results/H2_whitening_restarts.{txt,csv}")


if __name__ == "__main__":
    main()
