"""Drive Exo-Transmit on the committed test planets.

Exo-Transmit is the code MultiREx's opacity tables come from - the .dat files in
multirex/data are byte-identical to Exo_Transmit/Opac (verified by md5). Running
it therefore changes the radiative transfer implementation while holding the
opacity data exactly fixed, which is the clean version of R1-3 axis 1.

Configuration matched to MultiREx's make_tm():

  * Absorbers CH4, CO2, H2O, O3 only. MultiREx ships no CO or NH3 tables, so
    those gases contribute mean molecular weight and no opacity; selectChem.in
    switches their opacity off here to match, while their abundances stay in
    the EOS so the mean molecular weight agrees.
  * Collision Induced Absorption OFF - make_tm() adds absorption and Rayleigh
    only.
  * Rayleigh scattering ON, augmentation factor 1.0.
  * 100 isothermal layers between top_pressure and base_pressure, matching
    TauREx's nlayers default.
  * Planet radius at the base of the atmosphere. Exo-Transmit's userInput.in
    documents its radius the same way, so the conventions already agree.
"""
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

SRC = os.path.expanduser("~/exotransmit_src")
REPO = "/mnt/c/Users/owenh/BioSignatureDetectionModel"

R_EARTH_M, R_SUN_M, M_EARTH_KG = 6.3781e6, 6.957e8, 5.972167867791379e24
G_SI = 6.67430e-11
N_LAYERS = 100

GASES = ["H2O", "CO", "CO2", "NH3", "CH4", "O3"]
ABSORBERS = {"CH4", "CO2", "H2O", "O3"}
MOLAR_MASS = {"H2": 2.01588, "H2O": 18.01528, "CO": 28.0101, "CO2": 44.0095,
              "NH3": 17.03052, "CH4": 16.04246, "O3": 47.99820}

EOS_SPECIES = ("C CH4 CO COS CO2 C2H2 C2H4 C2H6 H HCN HCl HF H2 H2CO H2O H2S "
               "He K MgH N N2 NO2 NH3 NO Na O O2 O3 OH PH3 SH SO2 SiH SiO TiO "
               "VO").split()
EOS_T = [3000 - 100 * i for i in range(30)]          # 3000 down to 100
EOS_P = [10.0 ** e for e in range(8, -5, -1)]        # 1e8 down to 1e-4 Pa

SELECT_ORDER = ("CH4 CO2 CO H2O NH3 O2 O3 C2H2 C2H4 C2H6 H2CO H2S HCl HCN HF "
                "MgH N2 NO NO2 OCS OH PH3 SH SiH SiO SO2 TiO VO Na K").split()


def make_workdir(path):
    """A private Exo-Transmit tree; Opac is symlinked since it is 250 MB."""
    os.makedirs(path, exist_ok=True)
    for d in ("T_P", "EOS", "Spectra"):
        os.makedirs(os.path.join(path, d), exist_ok=True)
    opac = os.path.join(path, "Opac")
    if not os.path.exists(opac):
        os.symlink(os.path.join(SRC, "Opac"), opac)
    shutil.copy(os.path.join(SRC, "Exo_Transmit"), path)
    # otherInput.in carries the grid constants. The optical-depth count must
    # equal the number of layers in the T_P file; Exo-Transmit ships 334, but
    # TauREx uses 100, so match TauREx to isolate the radiative transfer code
    # rather than the vertical discretisation.
    other = open(os.path.join(SRC, "otherInput.in")).read().split("\n")
    other[47] = str(N_LAYERS)
    with open(os.path.join(path, "otherInput.in"), "w") as f:
        f.write("\n".join(other))
    return path


def write_selectchem(path):
    lines = ["For each gas, place a 1 after the equals sign if the gas opacity "
             "will be present in your transmission calculations, and 0 if not. "
             "Do not change the order of the gases in this file!"]
    for g in SELECT_ORDER:
        lines.append(f"{g} = {1 if g in ABSORBERS else 0}")
    lines.append("Scattering = 1")
    lines.append("Collision Induced Absorption = 0")
    lines.append("Do not delete this line!")
    with open(os.path.join(path, "selectChem.in"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_tp(path, row, name="t_p_planet.dat"):
    p_top = float(row["atm top_pressure"])
    p_base = float(row["atm base_pressure"])
    T = float(row["atm temperature"])
    ps = np.logspace(np.log10(p_top), np.log10(p_base), N_LAYERS)
    with open(os.path.join(path, "T_P", name), "w") as f:
        f.write("    i\tP\tT\n")
        for i, p in enumerate(ps):
            f.write(f"    {i}\t{p:.7E}\t{T:.7e}\n")
    return "/T_P/" + name


def write_eos(path, row, name="eos_planet.dat"):
    """Constant mixing ratios replicated over the whole T,P grid."""
    vmr = {g: 10.0 ** float(row[f"atm {g}"]) for g in GASES}
    vmr["H2"] = max(1.0 - sum(vmr.values()), 0.0)

    abund = {s: 0.0 for s in EOS_SPECIES}
    for g, v in vmr.items():
        abund[g] = v

    row_vals = "\t".join(f"{abund[s]:.6e}" for s in EOS_SPECIES)
    out = ["T\t\tP\t\t" + "\t\t".join(EOS_SPECIES), ""]
    for p in EOS_P:
        out.append(f"{p:.6e}")
        out.append("")
        for T in EOS_T:
            out.append(f"{float(T):.6e}\t{p:.6e}\t{row_vals}")
        out.append("")
    with open(os.path.join(path, "EOS", name), "w") as f:
        f.write("\n".join(out) + "\n")
    return "/EOS/" + name


def write_userinput(path, row, tp_rel, eos_rel, out_rel="/Spectra/out.dat"):
    r_p = float(row["p_radius"]) * R_EARTH_M
    g = G_SI * float(row["p_mass"]) * M_EARTH_KG / r_p ** 2
    r_s = float(row["s radius"]) * R_SUN_M
    body = [
        "userInput.in - ", "Formatting here is very important.",
        "Exo_Transmit home directory:", path,
        "Temperature-Pressure data file:", tp_rel,
        "Equation of State file:", eos_rel,
        "Output file:", out_rel,
        "Planet surface gravity (in m/s^-2):", f"{g:.6e}",
        "Planet radius (in m):", f"{r_p:.6e}",
        "Star radius (in m):", f"{r_s:.6e}",
        "Pressure of cloud top (in Pa):", "0.0",
        "Rayleigh scattering augmentation factor:", "1.0",
        "End of userInput.in (Do not change this line)",
    ]
    with open(os.path.join(path, "userInput.in"), "w") as f:
        f.write("\n".join(body) + "\n")
    return os.path.join(path, out_rel.lstrip("/"))


def run_planet(path, row):
    tp = write_tp(path, row)
    eos = write_eos(path, row)
    outfile = write_userinput(path, row, tp, eos)
    if os.path.exists(outfile):
        os.remove(outfile)
    r = subprocess.run(["./Exo_Transmit"], cwd=path, capture_output=True,
                       text=True, timeout=1800)
    if not os.path.exists(outfile):
        raise RuntimeError(f"no output.\nstdout:{r.stdout[-800:]}\n"
                           f"stderr:{r.stderr[-800:]}")
    d = np.loadtxt(outfile, skiprows=2)
    wl_um = d[:, 0] * 1e6          # metres -> microns
    depth = d[:, 1] / 100.0        # percent -> fraction
    return wl_um, depth


def bin_to_grid(wl, y, grid):
    order = np.argsort(wl)
    wl, y = np.asarray(wl)[order], np.asarray(y)[order]
    lg = np.log(grid)
    edges = np.empty(len(grid) + 1)
    edges[1:-1] = np.exp(0.5 * (lg[:-1] + lg[1:]))
    edges[0] = np.exp(lg[0] - 0.5 * (lg[1] - lg[0]))
    edges[-1] = np.exp(lg[-1] + 0.5 * (lg[-1] - lg[-2]))
    idx = np.digitize(wl, edges) - 1
    out = np.full(len(grid), np.nan)
    for i in range(len(grid)):
        m = idx == i
        if m.any():
            out[i] = y[m].mean()
    if np.isnan(out).any():
        good = ~np.isnan(out)
        out = np.interp(grid, grid[good], out[good])
    return out


def spectral_cols(df):
    fp = re.compile(r"^-?\d+\.\d+$")
    return sorted([c for c in df.columns
                   if isinstance(c, float) or (isinstance(c, str) and fp.match(c))],
                  key=float)


# --- Validation record ------------------------------------------------------
#
# Rayleigh-only comparison (all molecular opacity off in both codes, pure H2):
#
#   row   depth TauREx    depth ExoT     ratio   amp ratio
#    19   2.367846e-02  2.366041e-02   0.99924      0.9290
#   266   3.743411e-04  3.743358e-04   0.99999      0.9631
#   270   1.947561e-02  1.940230e-02   0.99624      0.8807
#
# Mean transit depth agrees to 0.4%, so geometry, gravity normalisation, mean
# molecular weight and the radius convention are all correct.
#
# Spectral contrast does NOT agree, by 4-12%, and the deficit tracks the
# atmosphere's total vertical extent relative to the planet radius:
#
#   row   thickness/R   amplitude deficit
#   266         0.006               0.037
#    19         0.061               0.071
#   270         0.147               0.119
#
# The cause is that Exo-Transmit assumes CONSTANT gravity through the
# atmosphere while TauREx integrates hydrostatically with gravity falling as
# altitude rises (planet.py calculate_scale_properties calls gravity_at_height
# per layer). These atmospheres span ~11-12 scale heights, so the top sits at
# up to 0.15 planet radii, where g is ~25% lower than at the base; TauREx's
# upper atmosphere is therefore more extended and its spectral contrast larger.
#
# This is a genuine difference between the two codes' physics, not a
# configuration knob - it cannot be matched without editing Exo-Transmit's
# source, which would defeat the purpose of using an independent code. It must
# be reported as the dominant identified difference alongside any accuracy
# result.
