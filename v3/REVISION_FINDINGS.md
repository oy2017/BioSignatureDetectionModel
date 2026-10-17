# Findings of the revision, as they come in

Dated entries; each names the result file, the control that supports it, and the caveats that still
apply. Expectations are scored against `REVISION_EXPECTATIONS.md`. Everything here is preliminary
until the entry says otherwise.

## 2026-10-16: first pass over the real JWST spectra

Inputs: WASP-39 b from the NIRISS SOSS, NIRCam F322W2 and NIRSpec G395H ERS products (PRISM, MIRI
and the Carter benchmark still downloading); eight panel planets from the files listed in
`results/jwst_panel/panel_summary.txt` (TOI-1231 b and the original K2-18 b and TOI-270 d spectra
are on OSF, which is down until 2026-10-19; WASP-166 b is still downloading). Conversion as in
`jwst_to_ariel.py` with modes aligned through their overlaps and uncovered bins filled flat at the
measured mean. Verdicts: `wasp39_assemble.py`, `jwst_panel.py`; controls: `jwst_controls.py`.

### F1. WASP-39 b: the classifier is right where the literature is unambiguous

At JWST precision, with all 104 inputs, all four learners call H₂O present and CH₄ absent at every
threshold that the literature decides, and CO₂ absent at 10⁻⁴ but present at 10⁻⁵, which is where
the published abundance sits (10⁻⁴·⁵ ± a decade across reductions). Under Ariel Tier-1 noise at
WASP-39 b's one transit, H₂O is detected in 77–96 % of draws and CO₂ in 0–41 %. The band metric of
Mugnai et al. agrees (M_H2O = 3.0 against their flag at 0.8; M_CH4 < 0). NH₃ is the one open verdict:
it flips with how the still-uncovered 5–8 µm region is filled, so it waits for the MIRI data.
(`results/wasp39/verdicts_prelim_nirHiss.txt`.) Expectations A1 (except NH₃, pending), A2 and A4
met so far; A3 not observed: no learner called H₂O absent with all inputs, so the optical points did
not trigger the haze mode on this planet.

Control: the clear H₂/He twin of WASP-39 b (MCS bulk parameters, published abundances) through the
same mask gives the same verdicts, and its H₂O is detected in 100 % of draws even at twice the real
errors. The real spectrum's dispersion over the measured bins is 321 ppm against 1,213 ppm for the
twin: the real atmosphere's features are muted four-fold (clouds, metallicity), and it is this
muting, not noise, that lowers the Tier-1 detection frequency from 100 % to 74–96 %.

### F2. The two real CH₄-bearing planets are missed, and the distance rule flags both

TOI-270 d (CH₄ at 9.4σ, log VMR −1.6) and K2-18 b (CH₄ at 4–11σ in three analyses, log VMR −1.7 to
−1.1) are called CH₄-absent by the classifier with all inputs (1 of 4 learners present on each), without
the optical points, and in the AIRS-only variant (TOI-270 d split 2–2). Their 10-nearest-neighbour
distances from the training set, 12.7 and 12.3, exceed the threshold that declines 10 % of clean
simulated planets (11.25): both are flagged as outside the training distribution.

Control (`results/jwst_panel/controls.txt`): the clear H₂/He twins with the same published abundances,
through the same coverage mask and the same flat fill, scattered with the real per-bin JWST errors
(30 and 45 ppm after binning), are detected as CH₄-bearing in 68 % and 96 % of 200 draws and are
never flagged; at twice the real errors they are still detected in 43 % and 60 %. So the miss is not
the noise, not the coverage, and not the classifier's in-simulator skill: it is the shape of the real
spectra. Both planets are temperate sub-Neptunes around M dwarfs with high mean molecular weight and
clouds or haze, a kind of atmosphere the consortium grid (H₂/He, four absorbers, gray cloud) does not
contain. In the editor's taxonomy this is a recoverable error: the information is in the spectrum
(JWST sees it), the classifier's training prior does not reach it, and a rule the note already
tested would have declined both planets. Expectation P3 met; P2 not met (the no-optical and AIRS-only
variants do not recover the misses, so this is not the haze mode of Section 3.3).

Caveats. TOI-270 d is the Felix et al. 2025 reanalysis at native resolution, not Benneke et al.'s
product; K2-18 b is a copy of the 2023 spectrum with a −41 ppm offset applied to one instrument by a
third party; 17 and 5 of 104 bins are flat-filled. The originals are on OSF and will be run when it is
back. The labels are the literature's; the CH₄ detections are among the most secure in the JWST
literature.

### F3. With AIRS-only coverage the classifier cannot test H₂O, and says so by construction

WASP-107 b, HAT-P-26 b and GJ 3470 b (NIRCam or G395H only, 50–69 of 104 bins flat-filled) are all
called H₂O-absent although H₂O is present at 6–21σ. Their twins through the same mask are also missed
(1 of 4 learners), while the same twins with full coverage are detected by all four: the H₂O bands
at 1.4, 1.9 and 2.7 µm sit in the uncovered NIRSpec region. This is a coverage artifact, not a
classifier failure, and the paper scores these planets only on CH₄ and CO₂, whose bands the data
cover. On those: CH₄ absent (correct, all three), CO₂ present for WASP-107 b (3 of 4; literature
ambiguous), HAT-P-26 b (2–3 of 4; present) and GJ 3470 b (2 of 4 all-input, 3 of 4 AIRS-only; present).

### F4. On real spectra the classifier's default is "absent"

Over the 22 panel labels the literature decides, the all-input classifier is right on 16 (73 %),
which is close to the share of absent labels (15 of 22). It reproduced one of the seven "present"
labels. Even where it is right, the Tier-1 detection frequencies of present molecules are low
(CO₂ on WASP-107 b 49 %; everything else below 30 %). Together with F1's four-fold muting, this says
the real spectra carry smaller features, relative to their continua, than the clear-grid training
spectra, and the classifier's operating point is tuned to the latter. Expectation P1 (≥ 75 % of
scored labels) marginally not met at 73 %; P6 met (no NH₃ false positive on any panel planet).

### What these add up to, provisionally

A simulator-trained classifier that reproduces its published in-simulator accuracies is right on the
one real hot Saturn in its training domain and wrong on both real planets that carry the molecule it
was built to find, because those planets lie outside the domain; a cheap distance rule, useless for
ranking errors inside the simulator (the note's Section 3.4), identifies exactly those planets on
real data. The missing piece for the paper is the identifiability test (running), which will say
whether Ariel at Tier 1 could decide these labels at all.
