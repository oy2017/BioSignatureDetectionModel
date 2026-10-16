"""Fail loudly if the forward model is not in the state every v3 result depends on.

Imported by generate_grid.py, shift_opacity.py and forward_model.py's other callers, so every
render passes through it.
  * Exo-Transmit opacity tables are read with pressures in Pa, not bar (TauREx <= 3.3.2 multiplies
    them by 1e5; ucl-exoplanets/taurex3 issue 172, fixed upstream in pull request 174 and applied
    here conditionally by forward_model.py; MULTIREX_FORK.md records the history).
  * All nine opacity tables are present in forward_model.OPACITY_DIR (fetch_external_data.sh).
  * A grey deck and a Mie haze given together are both rendered: forward_model.build_model adds
    both contributions (the MultiREx fork once dropped the deck; MULTIREX_FORK.md, change 7).
"""


def check():
    import taurex.opacity.exotransmit as et
    import forward_model as FM
    if not getattr(et.ExoTransmitOpacity, "_pa_fix", False):
        raise RuntimeError("the Exo-Transmit pressure-unit correction is not active; import forward_model first")
    missing = FM.missing_tables()
    if missing:
        raise RuntimeError(f"opacity tables missing from {FM.OPACITY_DIR}: {missing}; run fetch_external_data.sh")
    import inspect
    src = inspect.getsource(FM.build_model)
    if "LeeMieContribution" not in src or src.index("LeeMieContribution") > src.index("SimpleCloudsContribution"):
        raise RuntimeError("forward_model.build_model must add the haze before the grey deck and add both")


check()
