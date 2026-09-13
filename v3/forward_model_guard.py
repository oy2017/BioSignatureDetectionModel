"""Fail loudly if the forward model is missing a correction that every v3 result depends on.

Imported by generate_grid.py and shift_opacity.py, so every MultiREx render passes through it.
  * MultiREx fork change 6: Exo-Transmit opacity tables read with pressures in Pa, not bar
    (TauREx 3.3.2 multiplies them by 1e5; see MULTIREX_FORK.md and
    multirex_fork_change6_pressure_units.diff).
  * MultiREx fork change 7: a grey cloud deck given together with a Mie haze is rendered
    (it was silently dropped; multirex_fork_change7_cloud_with_haze.diff).
"""


def check():
    import multirex  # noqa: F401  (applies the fork's patches at import)
    import taurex.opacity.exotransmit as et
    if not getattr(et.ExoTransmitOpacity, "_multirex_pa_fix", False):
        raise RuntimeError("MultiREx fork change 6 (Exo-Transmit pressures in Pa) is not active; "
                           "apply v3/multirex_fork_change6_pressure_units.diff to multirex/spectra.py")
    import inspect, multirex.spectra as ms
    if "Fork change 7" not in inspect.getsource(ms):
        raise RuntimeError("MultiREx fork change 7 (grey deck kept alongside a Mie haze) is not active; "
                           "apply v3/multirex_fork_change7_cloud_with_haze.diff")


check()
