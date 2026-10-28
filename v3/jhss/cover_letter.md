# Cover letter for the resubmission

Manuscript: "Can a simulator-trained classifier for Ariel Tier-1 spectra be trusted?" (JHSS; decision of 17 October 2026: revise and resubmit, after the editor's discussion message of 15 October setting out the additions required).

Resubmitted as an original article under the title "Recoverable and irrecoverable errors of a simulator-trained classifier for Ariel Tier-1 spectra, with an external test on JWST observations".

[PENDING: the free-haze retrieval of the clean population completes 31 October; it replaces the clean-side stand-in in Table 9 and the bracketed note in Section 2.7. No number quoted below changes in a way that affects a conclusion.]

---

## Message to the editor on resubmission

Dear Dr Apte,

Thank you for reading the manuscript and for the discussion with the reviewer. I accepted the suggestions in full and have done the work. The revised manuscript is resubmitted as an original article, and the four requests of your message are each answered with a measurement:

1. **External validation on WASP-39 b.** The JWST benchmark spectrum of WASP-39 b (the four Early Release Science modes re-reduced uniformly by Carter & May et al. 2024, extended with the MIRI spectrum of Powell et al. 2024) was converted to the classifier's 104-bin Tier-3 input by a documented procedure, and the procedure was validated on a spectrum whose answer is known: the benchmark's own best-fit model, pushed through the conversion, differs from the same model binned directly by 22 ppm rms and changes no verdict. A 24-configuration sensitivity matrix (each mode's individual reductions, two NIRISS pipelines, offsets on and off, masked lines, with and without MIRI) measures how much each verdict depends on the reduction. At JWST precision the classifier reproduces the published composition: three of four learners call H₂O present, all four call CH₄, CO₂ and NH₃ absent at the 10⁻⁴ threshold, and CO₂ appears at 10⁻⁵ where its published abundance puts it. The one wrong learner is wrong for a reason the paper locates in the real spectrum, not in the conversion. Because one planet is a weak test, ten further planets with public JWST spectra were converted the same way (Section 2.6; Section 3.5).

2. **Identifiability.** A nested-sampling retrieval of every one of the classifier's 955 test planets, under the consortium's own test prior (POP-I) at each planet's Tier-1 transit count, gives for every label the posterior probability of the true class. Under that named population, a third of the CH₄ and NH₃ labels, 44 % of the CO₂ labels and 54 % of the H₂O labels cannot be decided at Tier-1 precision (posterior of the true class below 0.95), only 17 % of planets have all four labels decidable, and the retrieval's own accuracy, 92 to 94 %, is the ceiling the spectra allow, 9 to 16 points above the classifier (Section 2.4; Section 3.3; Table 5).

3. **A remedy with measurable value.** The value is measured as Tier-2 targets lost at a fixed 60 % selection, against the existing methods: the classifier alone, the five decline rules of the original note, and a retrieval of every planet. The result has three parts. On clean spectra, the retrieval of every planet is the selection oracle at 365 core-hours, and no cheap rule allocates a smaller retrieval budget better than chance. Under haze, the existing retrieval fails more confidently than the classifier. And on a population in which a fifth of the planets are hazed, a one-number score on the three optical inputs, routed to a retrieval with the haze density free, allocates a 20 % retrieval budget as well as an oracle that knows which planets are hazed, recovering 80 % of the hazed methane targets against 3 % for the classifier alone, 17 % for a random budget and 3 % for the confidence rules; without any retrieval, the same score routed to the AIRS-only variant recovers a third of the targets haze removes, at no retraining, and is never worse than the classifier (Sections 2.7 and 3.6; Tables 8 and 9).

4. **The two classes of error.** Every classifier error, on clean spectra and under each departure, was assigned to a class by the retrieval (recoverable: the spectrum decides the label; irrecoverable: it does not). Three quarters of the classifier's errors on clean spectra are irrecoverable. Haze and unocculted spots leave the labels decidable, and 57 to 87 % of the errors they introduce are recoverable; a deep cloud deck removes information on absent labels; threefold noise doubles the undecidable fraction and its added errors are mostly irrecoverable. The same two classes appear on the real spectra: the classifier missed methane on five of six JWST reductions of the three methane-bearing sub-Neptunes while the same design restricted to its AIRS spectrometer inputs found it on five of six (recoverable), and the five real warm giants are muted beyond anything the simulator population can produce at their bulk parameters (a prior limit no input-space rule can see). A retrieval that models the haze with one free parameter recovers 222 of the 224 methane labels the clean spectra decide; the same retrieval without that parameter is confidently wrong on 99 % of them (Sections 2.5, 3.4 and 3.5; Table 6; Figure 4).

The reviewer's further points (the scope of the "trust" claim, the mission-level decision problem, attribution to the re-implementation, and the compression of the second classifier and the decline-rule results) are each answered in the separate response file, which lists every comment of the reviewer and of your discussion message verbatim with how and where the revision addresses it. Every expectation was committed to the public repository before the corresponding run, and the outcomes, including the ones not met, are in Table 10. All code, result tables and the conversion of the JWST spectra are public.

I would be grateful if the revised manuscript could be considered as an original article.

With thanks,
Owen Yang

---
