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

## Changes made during the 2026-09-12 audit (all v3 results after that date depend on them)

6. **Exo-Transmit opacity pressures in Pa (bug fix, upstream in TauREx 3.3.2).** TauREx's
   `ExoTransmitOpacity._load_exo_transmit` multiplies the pressure line of each `opac*.dat` table
   by 1e5, reading it as bar. Exo-Transmit's tables are in Pa: its reader computes the number density
   as P/(k_B T) with k_B in J/K (`readopactable.c`), and its manual states that all quantities are SI.
   Unpatched, every cross section is looked up at a pressure 1e5 times too low. On test1 the
   clean-trained screen went from 96.3 % (as rendered before) to 90.0 % on spectra rendered with the
   correction (`results/check_pressure_units.txt`). The fork now divides the grid by 1e5 at load
   (`multirex_fork_change6_pressure_units.diff`). To report upstream (TauREx 3).
7. **Grey deck kept alongside a Mie haze (bug fix).** `make_tm` added the grey `SimpleClouds`
   contribution only in an `elif` after `cloud_model`, so `Atmosphere(cloud_pressure=…,
   cloud_model={…})` rendered the haze and silently dropped the deck. Affected the randomized grid's
   haze-plus-cloud planets and the consortium rebuild's haze case
   (`multirex_fork_change7_cloud_with_haze.diff`).

`forward_model_guard.py` (imported by `generate_grid.py` and `shift_opacity.py`) refuses to render
if either change is missing.
