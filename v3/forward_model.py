"""The forward model: TauREx 3 called directly (2026-10-16; replaces the MultiREx wrapper).

Every spectrum in v3 is an isothermal, constant-composition H2 (or H2/He) atmosphere in TauREx 3
with Exo-Transmit opacity tables, absorption and Rayleigh scattering, an optional grey cloud deck
and an optional Lee et al. (2013) Mie haze. Until 2026-10-16 this was done through a MultiREx fork
(see MULTIREX_FORK.md); the fork added nothing to the physics, only unit conversions and the
aerosol keywords, so it is reproduced here in one module and the fork is retired. The spectra
are bit-identical to the fork's (verified on the consortium populations; see the fork notes).

What is set up, in the order MultiREx did it:
  planet       TauREx Planet; radius and mass given in Earth units, converted with astropy's
               R_jup/R_earth and M_jup/M_earth; orbital distance in AU
  star         BlackbodyStar (temperature K, radius and mass in solar units)
  temperature  Isothermal(T)
  chemistry    TaurexChemistry(fill_gases=...) with one ConstantGas per species at 10**log10 VMR;
               a species without an opacity table acts through the mean molecular weight only
  model        TransmissionModel(atm_max_pressure=base, atm_min_pressure=top) with TauREx's
               default 100 layers; AbsorptionContribution; RayleighContribution;
               SimpleCloudsContribution(clouds_pressure) for a grey deck; LeeMieContribution for
               the haze (radius in um, Q0, mixing ratio; filling the column)
  opacities    Exo-Transmit tables from OPACITY_DIR (default v3/data/opacities; override with
               the ARIEL_OPACITY_DIR environment variable), fetched by fetch_external_data.sh

The Exo-Transmit pressure-unit correction (TauREx <= 3.3.2 reads the tables' Pa as bar; reported
as ucl-exoplanets/taurex3 issue 172, fixed upstream in pull request 174 on 2026-09-14 but not yet
released) is applied here, conditionally: after each table loads, the grid is divided by 1e5 only
if it is 1e5 times the file's second line. forward_model_guard.py checks that this is active.

Validation mirrors MultiREx's, so the same planets fail: temperatures, pressures, radii, masses
and the orbital distance must be positive, the base pressure must exceed the top pressure, and the
sum of the mixing ratios must lie in [0, 1]. A failure raises ValueError; spectrum() turns any
exception, NaN, or a depth outside (0, 1] into None, exactly as the old one_native() did.
"""
import os

import numpy as np
import taurex.log
import taurex.opacity.exotransmit as _et
from astropy.constants import M_earth, M_jup, R_earth, R_jup
from taurex.cache import OpacityCache
from taurex.chemistry import ConstantGas, TaurexChemistry
from taurex.contributions import (AbsorptionContribution, LeeMieContribution, RayleighContribution,
                                  SimpleCloudsContribution)
from taurex.contributions import Contribution
from taurex.data.fittable import fitparam


class PowerLawHazeContribution(Contribution):
    """A scattering haze with a power-law cross-section filling the column: sigma(lambda) = sigma0 (lambda / 1 um)^-alpha,
    in m^2 per molecule of the column (the 'enhanced Rayleigh' parameterisation used in free retrievals of real spectra,
    e.g. Pinhas et al. 2018, MacDonald & Madhusudhan 2017). It is a different family from the Lee et al. (2013) Mie haze
    the departures use, so a fit with it is a test of a misspecified aerosol model, not of knowing the generator."""

    def __init__(self, haze_log_sigma0=-40.0, haze_alpha=4.0):
        super().__init__("PowerLawHaze"); self._log_sigma0 = float(haze_log_sigma0); self._alpha = float(haze_alpha)

    @fitparam(param_name="haze_log_sigma0", param_latex=r"$\log\sigma_0$", default_fit=False, default_bounds=[-40, -20])
    def hazeLogSigma0(self):  # noqa: N802
        return self._log_sigma0

    @hazeLogSigma0.setter
    def hazeLogSigma0(self, v):  # noqa: N802
        self._log_sigma0 = float(v)

    @fitparam(param_name="haze_alpha", param_latex=r"$\alpha$", default_fit=False, default_bounds=[0, 8])
    def hazeAlpha(self):  # noqa: N802
        return self._alpha

    @hazeAlpha.setter
    def hazeAlpha(self, v):  # noqa: N802
        self._alpha = float(v)

    def prepare_each(self, model, wngrid):
        self._nlayers = model.nLayers; self._ngrid = wngrid.shape[0]
        lam_um = 1e4 / np.asarray(wngrid, float)
        sig = 10.0 ** self._log_sigma0 * lam_um ** (-self._alpha)
        self.sigma_xsec = np.repeat(sig[None, :], self._nlayers, axis=0)
        yield "PowerLawHaze", self.sigma_xsec

    @classmethod
    def input_keywords(cls):
        return ("PowerLawHaze",)

from taurex.model import TransmissionModel
from taurex.planet import Planet as TauPlanet
from taurex.stellar import BlackbodyStar
from taurex.temperature import Isothermal

HERE = os.path.dirname(os.path.abspath(__file__))
OPACITY_DIR = os.environ.get("ARIEL_OPACITY_DIR", os.path.join(HERE, "data", "opacities"))
TABLES = ["H2O", "CH4", "CO2", "O2", "O3", "CO", "HCN", "C2H2", "NH3"]     # opac<X>.dat, Exo-Transmit format
WL_MIN, WL_MAX = 0.5, 7.8                                                  # um, Ariel's range

_R_CONV = R_jup.value / R_earth.value      # Earth radii -> Jupiter radii: divide by this
_M_CONV = M_jup.value / M_earth.value

taurex.log.disableLogging()


# ------------------------------------------------------------------ opacity tables
def _apply_pressure_unit_fix():
    """Undo TauREx's bar->Pa conversion of Exo-Transmit tables when, and only when, it was applied."""
    if getattr(_et.ExoTransmitOpacity, "_pa_fix", False):
        return
    orig = _et.ExoTransmitOpacity._load_exo_transmit

    def load_in_pa(self, filename):
        orig(self, filename)
        with open(filename) as fh:
            fh.readline()
            file_grid = np.array([float(c) for c in fh.readline().split()])
        ratio = float(self._pressure_grid.min()) / float(file_grid.min())
        if abs(ratio - 1e5) < 1e-3 * 1e5:                 # TauREx <= 3.3.2 read the grid as bar
            self._pressure_grid = self._pressure_grid / 1e5
            self._min_pressure = self._pressure_grid.min()
            self._max_pressure = self._pressure_grid.max()
        elif abs(ratio - 1.0) > 1e-6:
            raise RuntimeError(f"Exo-Transmit pressure grid of {filename} is {ratio:g} x the file's; expected 1 or 1e5")

    _et.ExoTransmitOpacity._load_exo_transmit = load_in_pa
    _et.ExoTransmitOpacity._pa_fix = True


_apply_pressure_unit_fix()
_configured = {"key": None}


def configure_opacities(path=None):
    """Point TauREx at a directory of opacity tables. Idempotent per (process, path)."""
    path = os.path.abspath(path or OPACITY_DIR)
    key = (os.getpid(), path)
    if _configured["key"] == key:
        return
    if not os.path.isdir(path):
        raise FileNotFoundError(f"opacity directory {path} does not exist; run fetch_external_data.sh")
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(path)
    _configured["key"] = key


def missing_tables(path=None):
    path = path or OPACITY_DIR
    return [m for m in TABLES if not os.path.exists(os.path.join(path, f"opac{m}.dat"))]


# ------------------------------------------------------------------ the model
def _positive(name, value):
    v = float(value)
    if not v > 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return v


def build_model(row, gases, fill_gas, cloud_pressure=None, haze=None):
    """A built TauREx TransmissionModel for one planet.

    row           mapping with 'atm temperature', 'atm base_pressure', 'atm top_pressure' (K, Pa, Pa),
                  'atm <gas>' (log10 volume mixing ratio) for each gas in `gases`, 'p_radius' and
                  'p_mass' (Earth units), 's temperature', 's radius', 's mass' (K, solar units), 'sma' (AU)
    fill_gas      'H2' or ['H2', 'He'] (TauREx's default He/H2 ratio when both are given)
    cloud_pressure grey, opaque deck at this pressure (Pa), or None
    haze          None; or dict(radius=<um>, q=<Q0>, mix_ratio=<particles per m3>, optional topP, bottomP in Pa) for a
                  Lee et al. (2013) Mie haze, one number density in every layer between the two pressures (the whole
                  column by default); or dict(log_sigma0=<log10 m2 per molecule at 1 um>, alpha=<slope>) for a power-law
                  haze at constant mixing ratio
    """
    if _configured["key"] is None or _configured["key"][0] != os.getpid():
        configure_opacities()                     # default tables, unless the caller already chose a directory
    T = _positive("temperature", row["atm temperature"])
    base = _positive("base pressure", row["atm base_pressure"])
    top = _positive("top pressure", row["atm top_pressure"])
    if not base > top:
        raise ValueError("base pressure must be greater than top pressure")
    comp = {g: float(row[f"atm {g}"]) for g in gases}
    total = sum(10.0 ** v for v in comp.values())
    if total > 1 or total < 0:
        raise ValueError(f"the sum of mixing ratios must be between 0 and 1, got {total}")
    planet = TauPlanet(planet_distance=_positive("semi-major axis", row["sma"]),
                       planet_mass=_positive("planet mass", row["p_mass"]) / _M_CONV,
                       planet_radius=_positive("planet radius", row["p_radius"]) / _R_CONV)
    star = BlackbodyStar(temperature=_positive("star temperature", row["s temperature"]),
                         radius=_positive("star radius", row["s radius"]),
                         mass=_positive("star mass", row["s mass"]))
    chem = TaurexChemistry(fill_gases=fill_gas)
    for g, v in comp.items():
        chem.addGas(ConstantGas(molecule_name=g, mix_ratio=10.0 ** v))
    tm = TransmissionModel(planet=planet, temperature_profile=Isothermal(T=T), chemistry=chem, star=star,
                           atm_max_pressure=base, atm_min_pressure=top)
    tm.add_contribution(AbsorptionContribution())
    tm.add_contribution(RayleighContribution())
    if haze is not None and "log_sigma0" in haze:     # constant mixing ratio: cross-section per molecule, scales with the gas density
        tm.add_contribution(PowerLawHazeContribution(haze_log_sigma0=haze["log_sigma0"], haze_alpha=haze["alpha"]))
    elif haze is not None:
        tm.add_contribution(LeeMieContribution(lee_mie_radius=haze.get("radius", 0.01), lee_mie_q=haze.get("q", 40),
                                               lee_mie_mix_ratio=haze.get("mix_ratio", 1e-10),
                                               lee_mie_bottomP=haze.get("bottomP", -1), lee_mie_topP=haze.get("topP", -1)))
    if cloud_pressure is not None:
        tm.add_contribution(SimpleCloudsContribution(clouds_pressure=float(cloud_pressure)))
    tm.build()
    return tm


def transmission(row, gases, fill_gas, cloud_pressure=None, haze=None):
    """(wavelength um ascending within [WL_MIN, WL_MAX], transit depth) on TauREx's native grid."""
    tm = build_model(row, gases, fill_gas, cloud_pressure, haze)
    wn, rprs = tm.model()[:2]
    wl = 1e4 / np.asarray(wn, dtype=np.float64)
    order = np.argsort(wl)
    wl, rprs = wl[order], np.asarray(rprs, dtype=np.float64)[order]
    m = (wl >= WL_MIN) & (wl <= WL_MAX)
    return wl[m], rprs[m]


def spectrum(row, wl_native, gases, fill_gas, cloud_pressure=None, haze=None):
    """Noise-free transit depth on wl_native as float32, or None on failure (any exception, a
    non-finite value, or a depth outside (0, 1]). A model returned on another grid is interpolated."""
    try:
        wl, y = transmission(row, gases, fill_gas, cloud_pressure, haze)
        if y.shape[0] != wl_native.shape[0] or not np.allclose(wl, wl_native):
            y = np.interp(wl_native, wl, y)
        if not np.all(np.isfinite(y)) or np.any(y > 1.0) or np.any(y <= 0):
            return None
        return y.astype(np.float32)
    except Exception:
        return None


def native_wavelengths(gases=("H2O", "CO", "CO2", "NH3", "CH4", "O3"), fill_gas="H2"):
    """TauREx's native wavelength grid (the opacity tables' grid) restricted to [WL_MIN, WL_MAX]."""
    row = {"p_radius": 8.0, "p_mass": 50.0, "atm temperature": 1000.0, "atm base_pressure": 1e5,
           "atm top_pressure": 1.0, "s temperature": 5000.0, "s radius": 1.0, "s mass": 1.0, "sma": 0.1,
           **{f"atm {g}": -5.0 for g in gases}}
    return transmission(row, list(gases), fill_gas)[0]
