# When can a simulator-trained classifier be trusted?

> **The current paper and its supplementary material are in [`ariel_tier1_trust/`](ariel_tier1_trust/).**
> Start there. Everything else in this repository is the development history that led to it.

**TL;DR.** Machine-learning classifiers have been proposed to sort the thousand planets of
Ariel's quick-look survey by composition. They are trained and tested on simulated spectra,
because no real ones exist yet, so nobody knows what they do when a real atmosphere differs
from the simulator. We rebuilt the classifier the Ariel consortium proposed (its code was
never released), confirmed it reproduces all 48 published accuracies within 3.7 points, and
stress-tested it against fifteen ways the simulator can be wrong. It could be trusted when the
opacity database, the radiative-transfer code or two minor absorbers changed. It could not be
trusted under haze: at a haze comparable to the least hazy hot Jupiters observed, it declared
methane and water absent on nearly every planet that had them, grew *more* confident, and gave
no warning, although the molecular bands were intact. The cause was its three optical
photometric inputs; removing them largely removed the failure. A second classifier, built for
Ariel's carbon-to-oxygen question, failed in the opposite places. So the answer to "can this
classifier be trusted?" does not carry over from one classifier to another: each one needs its
own stress test.

## Why this matters, from the beginning

**What Ariel is.** Ariel is a European Space Agency telescope, launching in 2029, that will
study the atmospheres of about a thousand planets around other stars. It watches each planet
pass in front of its star and measures how much starlight the planet's atmosphere absorbs at
each wavelength. The result is a *transmission spectrum*, a short curve whose bumps say which
molecules are present: water, methane, carbon dioxide.

**The problem of a thousand planets.** Ariel cannot study every planet deeply. Every planet gets
a quick look (Tier 1), and a subset gets deeper observations (Tier 2 and beyond). Somebody has
to decide, from the quick look, which planets deserve more time. The Ariel consortium proposed,
among other tools, machine-learning classifiers that read the quick-look spectrum and flag
which molecules are there (Mugnai et al. 2021).

**The catch: there is nothing real to train on.** Ariel has not flown. The classifiers are
trained on simulated spectra, tens of thousands of imaginary planets from a computer model,
and tested on more spectra from the same model. That test gives a reassuring number. But the
simulator makes hundreds of choices (which molecules exist, whether there is haze, how star
spots contaminate the signal, how noisy the detector is), and the real universe will make them
differently. The standard test cannot see that, because training and testing share every
assumption.

**Why it is worth worrying about.** A classifier that quietly fails does not look like a
failure. It returns confident answers, and deep observations go to the wrong planets. The
haze result above is exactly that kind of failure: hazy planets, which are common, would be
written off as molecule-free with full confidence.

## What was done

1. **Rebuilt the published classifier** from the paper, element by element, and checked it
   against every published accuracy. An earlier, mistaken reading of the recipe missed by 10
   points on average, so the match is a real test of faithfulness.
2. **Stress-tested it.** For each way the simulator can be wrong (clouds, haze, star spots,
   noise, a gain ramp, another opacity database, another radiative-transfer code, missing
   absorbers) we measured the accuracy lost, how much of it any retraining could recover,
   whether a randomized training set absorbed it, and whether confidence scores, distance
   scores or conformal prediction flagged the errors.
3. **Found the cause of the haze failure** with a controlled test: identical classifiers with
   different inputs. Normalisation was not the cause; the optical photometric points were.
4. **Asked whether the answer transfers** by giving a second classifier, for a different Ariel
   science question, the same test. One counterexample is enough to show it does not.

## What was found

| | Consortium classifier | Carbon-rich classifier |
| :-- | :-- | :-- |
| Reproduction of published accuracies | 48 of 48 within 3.7 points | (built for this work) |
| Other opacity database / other code | −2.9 / −3.5 points | −9.0 / −0.7 points |
| HCN and C₂H₂ omitted from training | −0.1 points | −26 to −30 points on carbon-rich real targets |
| Haze, 3 × 10⁷ m⁻³ (weak end of observed hot-Jupiter hazes) | −17.7 points; methane reported on 0–5 % of planets | −5.1 points at full resolution; −29.1 at Tier 1 |
| Cause of the haze loss | the three optical photometric inputs (removing them: −5.6) | at Tier 1, the same points (removing them: −7.4, but −18.6 clean) |
| Does it know? | confidence rises; no decline rule helps | distance scores flag the missing absorbers; confidence does not |

Every number and the scripts that produce it: [`ariel_tier1_trust/README.md`](ariel_tier1_trust/README.md).

## Along the way

- **A unit bug in TauREx**, the radiative-transfer code behind these spectra, found and reported:
  it reads Exo-Transmit opacity tables at pressures 10⁵ times too low
  ([issue 172](https://github.com/ucl-exoplanets/taurex3/issues/172)). Correcting it made
  TauREx agree with Exo-Transmit's own code.
- **Expectations were committed before the stress test** and compared with the outcomes; several
  were wrong, and the misses are reported.
- **A full audit before the final run** found and fixed errors in our own pipeline; the corrections
  are recorded in [`v3/MULTIREX_FORK.md`](v3/MULTIREX_FORK.md) and the git history, and the
  pre-audit state is archived.

## Layout

```
ariel_tier1_trust/  The current paper's supplementary material: results, figures, faithfulness
                    argument, expectations, and the commands that reproduce everything.
v3/                 Where the scripts run. v3/jhss/ holds the manuscript source and build.
v2/, study1_R200/   Earlier studies. Superseded; kept for provenance.
nhsjs/              Submission package of an earlier paper. Not under active work.
ariel_noise_model/  ExoRad payload and noise curves.
reference papers/   Third-party PDFs cited.
```

## Environment

Python 3.10; `pip install -r requirements.txt`. The forward model is TauREx 3 through a
MultiREx fork whose changes are documented in [`v3/MULTIREX_FORK.md`](v3/MULTIREX_FORK.md),
including the Exo-Transmit pressure-unit correction; [`v3/forward_model_guard.py`](v3/forward_model_guard.py)
refuses to render without it. Re-rendering additionally needs Exo-Transmit, the ExoMolOP
tables, the PHOENIX atlas, FastChem and an ExoSim 2 environment. Result files are committed,
so every number can be checked without regenerating spectra.

## License

The code, result files and figures in this repository are released under the [MIT License](LICENSE).
The license does not cover third-party material kept for reference: the PDFs in `reference papers/`, journal
guidelines, templates and editorial correspondence, and files derived from third-party tools (TauREx 3, BSD 3-Clause;
MultiREx, MIT; ExoRad and ExoSim 2), which remain under their owners' terms. Manuscript text is excluded, since its
copyright is governed by the publishing journal.

