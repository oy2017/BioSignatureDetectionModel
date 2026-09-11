# NHSJS submission package (paper 1)

Built 2026-09-09 from the rebuilt study in `../v2/` (plan: `REWRITE_PLAN.md`).

## Files

- `manuscript.md` — the manuscript source. Citations as `[n]`; the builder
  renumbers by first mention. Figures are referenced from `../v2/results/figures/`.
- `build_docx.py` — builds both submission files from the official templates in
  `templates/`:
  - `Which simulation choices matter ... spectra.docx` — Standard citations
    (superscript numbers), blind (no authors, no acknowledgments). File name = title.
  - `Which simulation choices matter ... spectra - online.docx` — Online citations
    (full reference in double parentheses at every use, superscript commas
    between adjacent citations), with the author block from `authors.md` and
    the acknowledgments from `acknowledgments.md`.
  - `supplementary_figures.zip` — the five figure PNGs, for the form's third slot.
- `fill_numbers.py` — fills `{{placeholders}}` in the source from `../v2/results/`
  (all filled as of this build; re-run after any recompute).
- `templates/` — the journal's templates and CSL files, downloaded 2026-09-08.
- `REWRITE_PLAN.md` — the plan the study followed (version 2, full regeneration).

Superseded, kept for reference: `NHSJS_standard.md`, `citation_mapping.txt`,
`REBUILD_pending.docx`, and the DOCX named after the old title.

## Before submitting

1. Open both DOCX files in Word. Confirm the page count (estimate 16; limit 20),
   12-point Times New Roman, single spacing, and that the two template
   instruction pages are absent (the builder starts from an empty body).
2. Read `manuscript.md` end to end. Every number came from a file under
   `../v2/results/`; the Results and Discussion were drafted from them.
3. Decide the title (Section 5 of the plan offers an alternative).
4. Form answers: type = Research Article; number of citations = 28; prior
   submission: rejected by JHSS (not a publication) and a non-archival ML4PS
   2026 workshop submission of the earlier R = 200 study.

## Reproducing the numbers

All computation lives in `../v2/`; see `../v2/README.md`.
