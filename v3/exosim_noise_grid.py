"""Per-host-star Ariel noise curves from ExoSim2's radiometric model (the adopted noise axis).

RESEARCH_PLAN.md section 5: the radiometric budget depends on the star, not the planet, so
the noise axis is a grid over host T_eff, exactly the structure of the ExoRad curves v2
used (ariel_noise_model/ariel_nsr_curves.npz: teffs, wl_<T>, nsr_<T>). This produces the
same structure from ExoSim2's physics on the reconstructed Ariel-like payload
(v3/exosim_payload/README.md records every placeholder).

Stellar radius and distance are held at the example's values (1.18 R_sun, 47.5 pc); only
T_eff varies, because the study's SNR convention sets the noise *level* per planet and
takes only the wavelength *shape* from these curves. nsr_<T> is ExoSim2's total_noise,
i.e. relative noise for a 1-hour integration (unit hr^1/2).

Runs in the ExoSim2 venv (Python 3.12): ~/exosim2/venv/bin/python exosim_noise_grid.py
Writes results/exosim_nsr_curves.npz and results/exosim_noise_grid.log
"""
import os, re, shutil, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PAYLOAD = os.path.join(HERE, "exosim_payload")
WORK = os.path.expanduser("~/exosim2/noise_grid")
EXOSIM = os.path.expanduser("~/exosim2/venv/bin/exosim")
TEFFS = np.arange(2500, 7501, 250)                       # matches the ExoRad grid

def run(cmd, log):
    with open(log, "a") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode

def main():
    import h5py
    os.makedirs(WORK, exist_ok=True); os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log = os.path.join(HERE, "results", "exosim_noise_grid.log"); open(log, "w").close()
    out = {"teffs": TEFFS.astype(float)}
    for T in TEFFS:
        d = os.path.join(WORK, f"T{T}"); shutil.rmtree(d, ignore_errors=True); shutil.copytree(PAYLOAD, d)
        sky = os.path.join(d, "sky_example.xml"); s = open(sky).read()
        s = re.sub(r'<T unit="K">[^<]*</T>', f'<T unit="K"> {T} </T>', s)      # every source in the sky
        open(sky, "w").write(s)
        fp = os.path.join(d, "fp.h5"); t0 = time.time()
        rc = run([EXOSIM, "focalplane", "-c", "main_example.xml", "-o", fp], log) if os.chdir(d) is None else 1
        rc |= run([EXOSIM, "radiometric", "-c", "main_example.xml", "-o", fp], log)
        if rc != 0 or not os.path.exists(fp):
            print(f"T={T}: FAILED (see log)", flush=True); continue
        with h5py.File(fp) as h: tab = h["radiometric"]["table"][()]
        order = np.argsort(tab["wavelength"])
        out[f"wl_{T}"] = tab["wavelength"][order].astype(float)
        out[f"nsr_{T}"] = tab["total_noise"][order].astype(float)
        out[f"src_{T}"] = tab["source_signal_in_aperture"][order].astype(float)
        print(f"T={T}: {len(tab)} bins, median NSR@1h {np.nanmedian(tab['total_noise']):.2e}, {time.time()-t0:.0f} s", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    np.savez(os.path.join(HERE, "results", "exosim_nsr_curves.npz"), **out)
    print("wrote results/exosim_nsr_curves.npz with", sum(k.startswith("nsr_") for k in out), "curves", flush=True)

if __name__ == "__main__":
    main()
