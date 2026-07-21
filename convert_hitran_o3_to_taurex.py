"""
Convert petitRADTRANS' HITRAN ozone cross-sections into TauREx format.

ExoMolOP provides no ozone, so the opacity-swap experiment leaves O3 on its
original tabulation - and O3 is one of the two molecules defining the class
label. This converts the only alternative ozone data available (HITRAN, shipped
with petitRADTRANS as line-by-line cross sections) so that a four-molecule swap
can be run and the lower-bound caveat removed.

The conversion is a format translation, not a physical transformation:

  source  16O3__HITRAN.R1e6_0.3-28mu.xsec.petitRADTRANS.h5
          xsecarr[pressure, temperature, wavenumber], cm^2/molecule,
          p in bar, t in K, bin_edges in cm^-1, R ~ 1e6 (4.5e6 points)

  target  TauREx HDF5, identical key names, units and axis order to the
          ExoMolOP files already used, rebinned to R = 15000 to match them
          and to keep the table loadable alongside three others.

Rebinning averages the source cross-section within each target bin, which
conserves the band-integrated cross-section. That conservation is checked
explicitly and the script fails if it does not hold.

Usage:
    python convert_hitran_o3_to_taurex.py
"""

import os

import h5py
import numpy as np

SRC = os.path.expanduser(
    "~/petitRADTRANS/input_data/opacities/lines/line_by_line/O3/16O3/"
    "16O3__HITRAN.R1e6_0.3-28mu.xsec.petitRADTRANS.h5")
OUT_DIR = os.path.expanduser("~/exomolop_o3")
OUT = os.path.join(OUT_DIR, "16O3__HITRAN.R15000_0.3-28mu.xsec.TauREx.h5")
TARGET_R = 15000
# Tolerance on band-integrated cross-section after rebinning.
INTEGRAL_TOL = 0.02


def make_grid(wn_min, wn_max, resolution):
    """Log-spaced wavenumber grid at constant resolving power."""
    n = int(np.ceil(resolution * np.log(wn_max / wn_min)))
    return np.exp(np.linspace(np.log(wn_min), np.log(wn_max), n))


def bin_slice(wn_src, x_src, edges, n_out):
    """Mean cross-section of the source within each target bin."""
    idx = np.clip(np.digitize(wn_src, edges) - 1, 0, n_out - 1)
    total = np.bincount(idx, weights=x_src, minlength=n_out)
    count = np.bincount(idx, minlength=n_out)
    out = np.zeros(n_out)
    hit = count > 0
    out[hit] = total[hit] / count[hit]
    return out, hit


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(SRC, "r") as h:
        p = h["p"][:].astype(np.float64)          # bar
        t = h["t"][:].astype(np.float64)          # K
        wn_src = h["bin_edges"][:].astype(np.float64)
        n_p, n_t = len(p), len(t)
        print(f"source: {n_p} pressures {p.min():.1e}-{p.max():.1e} bar, "
              f"{n_t} temperatures {t.min():.0f}-{t.max():.0f} K, "
              f"{len(wn_src):,} wavenumbers")

        wn_out = make_grid(wn_src.min(), wn_src.max(), TARGET_R)
        n_out = len(wn_out)
        edges = np.empty(n_out + 1)
        edges[1:-1] = np.sqrt(wn_out[1:] * wn_out[:-1])
        edges[0] = wn_out[0] ** 2 / edges[1]
        edges[-1] = wn_out[-1] ** 2 / edges[-2]
        print(f"target: {n_out:,} wavenumbers at R = {TARGET_R}")

        xsec = np.zeros((n_p, n_t, n_out), dtype=np.float64)
        worst = 0.0
        for ip in range(n_p):
            for it in range(n_t):
                src = h["xsecarr"][ip, it, :].astype(np.float64)
                binned, _ = bin_slice(wn_src, src, edges, n_out)
                xsec[ip, it, :] = binned
                # Conservation check on the band that matters for this work.
                m_src = (wn_src >= 1282) & (wn_src <= 20000)   # 0.5-7.8 um
                m_out = (wn_out >= 1282) & (wn_out <= 20000)
                if m_src.sum() > 2 and m_out.sum() > 2:
                    i_src = np.trapz(src[m_src], wn_src[m_src])
                    i_out = np.trapz(binned[m_out], wn_out[m_out])
                    if i_src > 0:
                        worst = max(worst, abs(i_out - i_src) / i_src)
            print(f"  pressure {ip + 1}/{n_p} done", flush=True)

    print(f"\nworst band-integral deviation after rebinning: {worst:.3%}")
    if worst > INTEGRAL_TOL:
        raise RuntimeError(
            f"rebinning changed the band-integrated cross-section by "
            f"{worst:.2%}, above the {INTEGRAL_TOL:.0%} tolerance - the "
            f"converted table is not trustworthy")

    with h5py.File(OUT, "w") as o:
        o.create_dataset("bin_edges", data=wn_out)
        o.create_dataset("p", data=p)
        o.create_dataset("t", data=t)
        o.create_dataset("xsecarr", data=xsec.astype(np.float32))
        o.create_dataset("mol_name", data=np.array([b"O3"]))
        o.create_dataset("key_iso_ll", data=np.array([b"16O3__HITRAN"]))
        o.create_dataset("DOI", data=np.array([b"10.1016/j.jqsrt.2013.07.002"]))
        o["bin_edges"].attrs["units"] = "wavenumbers"
        o["p"].attrs["units"] = "bar"
        o["t"].attrs["units"] = "kelvin"
        o["xsecarr"].attrs["units"] = "cm^2/molecule"
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
