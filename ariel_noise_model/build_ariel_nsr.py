"""Regenerate the Ariel per-Teff NSR curves used by ../evaluate_ariel_noise.py.

Runs ExoRad2 (Mugnai et al. 2020), the open radiometric engine underneath
ArielRad, with an Ariel payload reconstructed from PUBLISHED parameters
(ariel_payload.xml -- Tinetti et al. 2018 Ariel Definition Study Report;
Mugnai et al. 2020; Edwards et al. 2019). This is NOT the consortium's official
ArielRad instrument file, which is not public.

ExoRad is NOT installed in the pinned classification environment (~/tfenv). It
lives in a separate venv so it cannot perturb numpy/xgboost:

    python3.10 -m venv ~/.venvs/exorad
    ~/.venvs/exorad/bin/pip install exorad

Then, from this directory:

    ~/.venvs/exorad/bin/exorad -p ariel_payload.xml -t ariel_targets.csv -o ariel_out.h5
    ~/.venvs/exorad/bin/python build_ariel_nsr.py

which writes ../final_results/ariel_nsr_curves.npz (per-Teff Wavelength + NSR,
the dimensionless total_noise column, h^1/2). ExoRad ignores the K-magnitude
column and sets flux from distance/radius/Teff, so ariel_targets.csv places each
Teff at a common operating point (1 Rsun, 20 pc); the NSR *shape* (coloring) is
what the classification experiment uses, with the overall level swept.
"""
import h5py
import numpy as np

TEFFS = list(range(2500, 7501, 250))
OUT = "../final_results/ariel_nsr_curves.npz"


def main():
    f = h5py.File("ariel_out.h5", "r")
    data = {"teffs": np.array(TEFFS)}
    for T in TEFFS:
        t = f[f"targets/T{T}/table/table"][:]
        o = np.argsort(t["Wavelength"])
        data[f"wl_{T}"] = t["Wavelength"][o].astype(float)
        data[f"nsr_{T}"] = t["total_noise"][o].astype(float)   # dimensionless NSR
    np.savez(OUT, **data)
    print(f"wrote {OUT} with {len(TEFFS)} Teff NSR curves")


if __name__ == "__main__":
    main()
