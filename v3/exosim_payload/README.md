# Ariel-like ExoSim2 payload (reconstruction)

Built 2026-09-11 from the public ExoSim2 example (`arielmission-space/ExoSim2-public`,
v2.2.1). **This is a reconstruction, not the mission's payload file**, which is held in a
restricted repository. Every number below that is Ariel's comes from the published
layout (Tinetti et al. 2018; ArielRad, Mugnai et al. 2020); everything else is the ExoSim2
example reused as a placeholder. The paper must say so.

Run (Python >= 3.12 venv with `exosim` 2.2.1 installed from the clone, plus `setuptools`):

    exosim focalplane  -c main_example.xml -o /abs/path/fp.h5      # ~20 s
    exosim radiometric -c main_example.xml -o /abs/path/fp.h5      # ~30 s  (absolute -o path required)

The radiometric table (`radiometric/table` in fp.h5) is the product: per bin, source
signal and the photon / read / dark / foreground noise terms. `total_noise` is the
*relative* noise for a 1-hour integration in units hr^1/2, so sigma_rel(t) =
total_noise / sqrt(t / 1 hr).

## What is Ariel's (published)
- six channels: VISPhot 0.50-0.60, FGS1 0.60-0.80, FGS2 0.80-1.10 (photometers);
  NIRSpec 1.10-1.95 R=15; AIRS-CH0 1.95-3.90 R=100; AIRS-CH1 3.90-7.80 R=30
- A_tel = 0.63 m^2

## Every deviation from the shipped example, and why
1. `main_example.xml` working grid widened 0.7-4.0 -> 0.45-8.0 um (the example's grid
   could not contain AIRS-CH1).
2. `QE.ecsv` and `qe_map.h5` are keyed by channel name: six columns / groups added as
   copies of the example's Photometer and Spectrometer entries.
3. Spectrometer channels use `EstimateApertures` instead of the example's fixed 52-row
   `LoadApertures` table, which was sized for the example's own bins.
4. Each spectrometer has its own linear wavelength solution `spec-wl_sol_<ch>.ecsv`
   mapping its band across the detector; the example's mapped 0.95-3.9 only.
5. Each channel has its own flat bandpass dichroic `D1_<ch>.ecsv` (0.90 inside its band,
   0 outside). The example's D1 was a 1.0-3.5 / 0.7-1.0 dichroic; photometer bands are
   set by this filter, not by wl_min/wl_max.
6. `M1.ecsv` and `QE.ecsv` extended to 0.45-8.0 by flat extrapolation of edge values.
7. The two placeholder `custom_noise` terms (test1=20, test2=30) removed from the
   spectrometers' radiometric block; they dominated `total_noise`.
8. Photometer radiometric blocks given `dark_current` and `read_noise` (the example
   photometer reported NaN noise).
9. **`earthsky` foreground removed from `sky_example.xml`.** It is a MODTRAN atmospheric
   transmission for a 38 km balloon line of sight, nonzero only 0.5-5.0 um, and it
   multiplies into every path. Ariel is a space telescope. This was the last thing
   zeroing AIRS-CH1 and it is a physics correction, not a placeholder choice.

## Validation against the ExoRad curve v2 used (same 6086 K star)
Per-bin noise-to-signal shape: Spearman 0.981 over all 104 bins; per channel +0.978 to
+0.987. Absolute level a uniform factor 2.0-2.2 higher in every channel (integration
convention and placeholder throughputs); the SNR convention in the study sets the level,
so the shape is what matters and it agrees.
