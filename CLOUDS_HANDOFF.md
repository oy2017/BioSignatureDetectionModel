# Cloud and haze prescriptions — state and next steps

Working notes for continuing R1-3 axis 3 ("different cloud and haze prescriptions")
in a fresh session. Everything below was established empirically; the failures are
reproducible.

## Goal

Reviewer 1 asked for *different cloud and haze prescriptions*, plural, naming hazes
specifically. His exact wording is in `revision/scholastica_reviews_verbatim.txt`
(gitignored, local-only) — see the R1-3 passage. A grey deck is done and verified. Hazes are not, and hazes are the
scientifically interesting case — see the prediction at the bottom.

Current honest coverage of R1-3: **about 1.5 of 7 axes.**

## What works — do not re-litigate this

`SimpleCloudsContribution`, an optically thick grey deck, is implemented and
verified end to end.

* MultiREx fork `oy2017/MultiREx-public` at **2234e2c** carries it.
* `generate_cloudy_testset.py` generates the datasets; `evaluate_cloudy.py`
  evaluates them against the frozen pipeline.
* Five datasets at cloud-top pressures 1e5 … 1e1 Pa are committed.
* Verified suppression is monotonic and matches physics:
  4.4% / 18.8% / 43.1% / 70.3% / 94.6% at 1e5 / 1e4 / 1e3 / 1e2 / 1e1 Pa.

## What is blocked

Both Mie contributions are unusable as currently called. Plumbing is in place and
appears correct; the TauREx parameters are the problem.

**`LeeMieContribution`** — wavelength-dependent scattering, i.e. the haze model.
Zero effect at every mixing ratio from 1e-8 to 1e0. Output is bit-identical to
cloud-free.

> **Lead:** `LeeMieContribution(...).fitting_parameters()` returns
> `lee_mie_radius`, `lee_mie_q`, `lee_mie_topP`, `lee_mie_bottomP` — and
> **`lee_mie_mix_ratio` is absent from that list.** The constructor accepts it, but
> it does not appear as a fitting parameter, which suggests the value never reaches
> the opacity calculation and must instead be set through TauREx's parameter
> system after construction.

**`FlatMieContribution`** — finite grey opacity. Saturates to 100% suppression at
every opacity from 1e-20 to 1e-2. A 1e-20 opacity blanking a spectrum is not
physical, so it is being called wrong rather than being genuinely that strong.

## Where the code is

| Item | Location |
| :-- | :-- |
| Mie plumbing patch (9 sites, correct, inert for LeeMie) | `~/multirex-mie-plumbing.patch` |
| Pre-Mie installed package, working grey deck | `~/spectra.py.bak` |
| Installed package (has the Mie plumbing) | `~/tfenv/lib/python3.10/site-packages/multirex/spectra.py` |
| Clean rollback | `cp ~/spectra.py.bak <installed path>` |

The Mie plumbing is **not** committed to the fork, deliberately — it is untested
and LeeMie is inert. Do not push it until a haze demonstrably changes a spectrum.

The plumbing adds a `cloud_model` dict parameter to `Atmosphere`:

```python
cloud_model={"type": "simple",   "pressure": 1e3}
cloud_model={"type": "flat_mie", "mix_ratio": 1e-10, "bottomP": 1e5, "topP": 1e0}
cloud_model={"type": "lee_mie",  "radius": 0.1, "q": 40,
             "mix_ratio": 1e-10, "bottomP": 1e5, "topP": 1e0}
```

## Suggested next steps

1. Read TauREx's `LeeMieContribution` source, specifically how `lee_mie_mix_ratio`
   is consumed in the opacity calculation versus how it is stored. Compare against
   `SimpleCloudsContribution`, which works, to see what differs in how a parameter
   reaches `contribute()`.
2. Test the contribution **in isolation** on a bare TauREx `TransmissionModel`
   before going anywhere near MultiREx, so the failure surface is one library not
   two.
3. Only once a haze visibly changes a spectrum, wire it through MultiREx and verify
   **through `explore_multiverse`** — see the warning below.
4. Calibrate the haze opacity to match the grey deck's feature-suppression levels
   (0.92 / 0.73 / 0.50 / 0.27 of clear amplitude) so the two prescriptions are
   compared at equal muting and only the wavelength dependence differs.

## When the haze result exists, update FOUR places

`revision_plan.md` serves two deliverables at once — the paper is written from the
rewrite instructions, organised by manuscript section, while the editor's
comment-indexed response document is assembled from the R1-n sections. A new result
belongs in both halves, and they do not update each other.

This was already missed once: the cloud results were written into the R1-3 response
section and the rewrite instructions were left describing only the injected-
systematics sweep, so the writing half of the document did not know clouds existed.

| # | Location | What to change |
| :-- | :-- | :-- |
| 1 | `revision_plan.md` → R1-3 → "Clouds — results" | Add the haze results table alongside the grey-deck one |
| 2 | `revision_plan.md` → R1-3 → coverage table, row 3 | Update status, and the "about 1.5 of seven axes" figure in the section header **and** in the document header near the status table |
| 3 | `revision_plan.md` → `## §4 — new Robustness subsection` → "Clouds" block | Add draft text for the haze result; replace the closing prediction paragraph with what the experiment actually found |
| 4 | `revision_plan.md` → `## §3 — new Assumptions subsection` → "No clouds or hazes in training" row | Update, since hazes would no longer be untested |

Also update this file, and delete it once hazes are done and nothing is outstanding.

## Two traps that cost real time here

**Verify through `explore_multiverse`, never through `generate_spectrum`.** The
cloud feature was verified on the direct path and appeared to work, but
`explore_multiverse` calls `clone_shuffled()`, which rebuilds the `Atmosphere` from
`original_params` and silently dropped the parameter. Three commits and five
datasets were produced before this surfaced. Every generated dataset was
cloud-free while appearing correct.

**Measure suppression scale-free.** Use per-spectrum scatter divided by that
spectrum's own mean depth. Absolute scatter scales with transit depth and therefore
with radius squared, and radius spans 1–26 R⊕, so an absolute measure is dominated
by the radius draw and hides the cloud effect entirely.

## The prediction worth testing

§4.2 established that the classifier ignores PC0 and PC1, which capture mean
transit depth and continuum slope, because they carry no label information. The
systematics sweep corroborated this: an additive baseline offset, which moves only
PC0, produced no degradation at all.

A grey deck suppresses feature amplitude uniformly and is highly damaging — up to
33 accuracy points. A haze acts principally as a wavelength-dependent continuum
tilt, which is close to a PC1-like mode.

**So a haze should degrade performance substantially less than a grey deck at equal
feature suppression.** If that holds, the claim becomes far more precise than
"clouds hurt": the pipeline is vulnerable to opacity that suppresses high-frequency
structure and insensitive to opacity that only tilts the continuum. That is a real
result, and it is worth the debugging.
