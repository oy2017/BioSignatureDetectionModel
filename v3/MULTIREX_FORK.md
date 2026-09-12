# The MultiREx fork used by this study (checked 2026-09-12 against upstream 0.3.1)

The installed `multirex` (0.3.1) is a local fork (pip records `file:///…/multirex_fork`).
`diff` against the upstream 0.3.1 wheel shows five changes:

1. **Seeding bug fix (real, demonstrated).** Upstream seeds every `Atmosphere`, `Planet`, `Star`
   and `System` with `int(time.time())` and reseeds the *global* `np.random` with it. Any two
   objects built within the same wall-clock second therefore draw identical "random" parameters.
   Worse, `clone_shuffled()` passes the parent's seed into the clone, so a "reshuffled" clone
   reproduces the parent's draws exactly; `explore_multiverse()` is built on `clone_shuffled`.
   Demonstration with upstream code (scratchpad, 2026-09-12): three Systems built in the same
   second → identical temperature, composition and radius; `clone_shuffled()` twice → identical.
   The fork seeds with `time.time_ns() % (2**32-1)` and stops passing the parent seed to clones.
   **v2 and v3 never use `clone_shuffled` or `explore_multiverse`** (parameters are drawn with
   NumPy and passed explicitly, independence verified in v2), so their results do not depend on
   this fix either way. `study1_R200` used `explore_multiverse` and is the run the fix was made for.
2. **Aerosols (feature).** `Atmosphere(cloud_pressure=…)` adds a grey `SimpleCloudsContribution`;
   `Atmosphere(cloud_model={"type": "lee_mie"|"flat_mie"|"simple", …})` adds the corresponding
   TauREx Mie contribution. Used by every cloud/haze axis and by the Alfnoor rebuild.
3. **Failure handling in `explore_multiverse`.** NaN spectra and RuntimeWarnings (overflows) are
   caught, logged to `failed_parameters.log`, and dropped instead of propagating. Not used by v2/v3.
4. **PHOENIX flag propagated in `clone`** (upstream lost the `phoenix` setting on cloning).
5. **`utils.get_gases` path handling** (creates the directory; `os.path.join`).

Opacity tables added to `multirex/data/` by this study (Exo-Transmit format, from
`~/exotransmit_src/Opac/`): `opacCO.dat`, `opacHCN.dat`, `opacC2H2.dat`, `opacNH3.dat` (the last
on 2026-09-12; before it NH3 acted through mean molecular weight only — see TRUST_IDEA §4g).

To do: report item 1 upstream (D4san/MultiREx) with the three-line reproduction above.
