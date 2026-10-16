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
   (`multirex_fork_change6_pressure_units.diff`). Reported upstream 2026-09-13: https://github.com/ucl-exoplanets/taurex3/issues/172 (evidence and reproduction in `upstream_reports/taurex3_issue172/`).
7. **Grey deck kept alongside a Mie haze (bug fix in this fork's own aerosol feature, change 2; upstream MultiREx has no cloud or haze support, so nothing to report upstream).** `make_tm` added the grey `SimpleClouds`
   contribution only in an `elif` after `cloud_model`, so `Atmosphere(cloud_pressure=…,
   cloud_model={…})` rendered the haze and silently dropped the deck. Affected the randomized grid's
   haze-plus-cloud planets and the consortium rebuild's haze case
   (`multirex_fork_change7_cloud_with_haze.diff`).

`forward_model_guard.py` (imported by `generate_grid.py` and `shift_opacity.py`) refuses to render
if either change is missing.

## 2026-10-16: changes 6 and 7 pushed to the public fork; change 6 made conditional

Both corrections now live on `oy2017/MultiREx-public` as commits `9376f4a` (change 6) and
`83f739c` (change 7), and `requirements.txt` pins that commit; the diff files in this directory
are regenerated from those commits. Until then the two changes existed only as local patches,
and a fresh install from the pinned commit could not render a cloud or haze case.

TauREx fixed issue 172 upstream on 2026-09-14 (pull request 174, "Fix exotransmit pressure
units"), with a unit test; the latest release, 3.3.2 of 2025-12-16, does not contain it. Change 6
therefore now checks, after each table is loaded, whether the loaded pressure grid is 1e5 times
the file's second line, and divides only in that case; it raises if the ratio is neither 1 nor
1e5. The same code is correct under 3.3.2 and under a TauREx that includes the fix, and
`forward_model_guard.py` is unchanged.

Fresh-install recipe (macOS arm64 or Linux, Python 3.10): `pip install -r requirements.txt`,
then copy `opacCO.dat`, `opacHCN.dat`, `opacC2H2.dat` and `opacNH3.dat` from
https://github.com/elizakempton/Exo_Transmit/tree/master/Opac into the installed
`multirex/data/`; FastChem's `asplund_2009.dat` and `logK.dat` go in `~/fastchem_input/`; the
STScI `phoenixm00_*.fits` atlas goes in `v2/phoenix/`; the Mission Candidate Sample table
`Ariel_MCS_Known_2026-05-11.csv` goes in `v3/data/mcs/`. Verified 2026-10-16 on macOS arm64: the
faithful populations re-rendered with identical failure counts and the reproduction fit matched the
committed result file to 0.05 points on average (maximum 1.2 on one ensemble-vote cell).

## 2026-10-16, later: the fork retired; v3 calls TauREx 3 directly

MultiREx added nothing to the physics: it converted units, built the TauREx objects and, in the
fork, added the aerosol keywords. `v3/forward_model.py` now does exactly that in one module
(planet and star units, isothermal profile, constant chemistry with the fill gas, absorption and
Rayleigh contributions, grey deck, Lee Mie haze, 100 layers, the conditional pressure-unit
correction), reading the Exo-Transmit tables from `v3/data/opacities/` (`fetch_external_data.sh`;
all nine tables are byte-identical to Exo-Transmit's `Opac/` on GitHub, including the five the
fork bundled). `generate_grid.py`, `shift_aerosol.py`, `shift_absorbers.py`, `shift_opacity.py`,
`alfnoor_screen.py` and `alfnoor_trust.py` call it; `generate_grid.py` no longer imports v2's
generator, and `forward_model_guard.py` checks the new module. `requirements.txt` no longer
installs MultiREx; v2 (superseded) still needs the fork at commit `83f739c` to re-render.

Verification, 2026-10-16, with MultiREx uninstalled from the environment, against the spectra
rendered through the fork on 2026-10-15 (same machine): the native wavelength grid is identical;
the faithful consortium populations (POP-I clean, haze 3e7, deck 1e2 Pa, HCN+C2H2 with deck; POP-III;
240 planets each), the first-reading populations (H2 fill, 1 Pa top; 200 planets, four cases) and
the Mission Candidate Sample set (400 planets) are bit-identical, with the same planets failing.
`check_pressure_units.py` remains as the audit's record and still names the fork; it is not run.

One hazard worth knowing when testing in a single process: joblib's loky reuses worker processes,
and `alfnoor_faithful._worker` rebinds `generate_grid.FILL_GAS` and `shift_aerosol.FILL_GAS` to
`["H2", "He"]` inside them. A later render in the same process that expects the default `"H2"`
fill inherits the H2/He fill in the reused workers. The pipeline scripts each run in their own
process, so none is affected; a comparison script that renders both kinds in one process must
start a fresh process (or `Parallel` pool) between them.
