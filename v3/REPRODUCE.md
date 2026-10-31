# Reproducing the manuscript

Every table and figure of `jhss/manuscript.md`, the script that writes it and the committed result file. Run from `v3/` with `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` and the pinned environment. Spectra and trained models are regenerated, not committed (`data/`); `fetch_jwst_data.sh` fetches the JWST spectra, and `cia_effect.py` states where the HITRAN tables come from.

| Result | Script | Result file |
| :-- | :-- | :-- |
| Spectra and learners | `alfnoor_faithful.py --render --jobs 8`, then `alfnoor_faithful.py --fit --natives faithful --layout tier3_r20 --noise radiometric` | `data/alfnoor_faithful/` |
| Table A2, Figure 1 | as above, `reproduction_seeds.py`, `plots/note_fig1_reproduction.py` | `results/alfnoor_faithful_faithful_tier3_r20_radiometric.csv`, `results/reproduction_seeds.txt` |
| Table 1, Table 5, Table D3 | `alfnoor_trust.py --render --jobs 8`, `alfnoor_trust.py --fit`, `threshold_holdout.py` | `results/alfnoor_trust.txt` |
| Table 6, Figure 2 | `alfnoor_haze_mechanism.py`, `plots/note_fig3_haze.py` | `results/alfnoor_haze_mechanism.txt` |
| Confined hazes | `haze_confined.py`, `haze_top_test.py` | `results/haze_top_test.txt`, `results/pipeline_sim.txt` |
| Collision-induced absorption | `cia_effect.py <tables> [--population]` | `results/cia_effect.txt` |
| Table 4 | `identifiability_calibrate.py --n 1000 --jobs 9`, `--noisy --random-sample --n 200`, `--prior population --random-sample --n 200`, `noisy_vs_noisefree.py`, `selection_checks.py` | `results/identifiability_calibration*.csv`, `results/selection_checks.txt` |
| Table 7, Figure 3 | `identifiability.py`, `alfnoor_diagnostic.py`, `plots/fig4_identifiability.py` | `results/diagnostic_departure_classes.csv`, `results/diagnostic_retrieval_classes.csv`, `results/diagnostic.txt` |
| Table 2, Table 8 | `jwst_panel.py`, `jwst_muting_census.py`, `muting_bootstrap.py`, `jwst_controls.py`, `panel_mechanism.py` | `results/jwst_panel/` |
| Figure 4 | `plots/fig5_muting.py` | `results/jwst_panel/muting_census.csv`, `muting_population.csv`, `muting_bootstrap.csv` |
| Label audit | by reading the sources; `label_sensitivity.py` | `results/jwst_panel/label_sources.csv`, `label_sensitivity.txt` |
| Table 11 | `alfnoor_retrain.py`, `muting_router.py`, `amplitude_flag.py`, `router_real.py` | `results/jwst_panel/muting_router.csv`, `amplitude_flag_router.csv`, `router_real.csv` |
| Table 12 | `amplitude_flag.py`, `panel_tier1.py`, `panel_mechanism.py`, `panel_control.py`, `pipeline_table.py` | `results/pipeline_table.md`, `results/jwst_panel/panel_tier1.txt`, `panel_control.txt` |
| Table 13, Table 14 | `pipeline_sim.py`, `pipeline_table.py`, `table11_spread.py` | `results/pipeline_sim.txt`, `results/pipeline_table.md`, `results/table11_spread.txt` |
| Table 16 | `identifiability_calibrate.py --departure haze3e7 [--free-haze | --aerosol powerlaw]`, `powerlaw_summary.py` | `results/identifiability_calibration_haze3e7*.csv`, `results/powerlaw_summary.txt` |
| Table 17, Table C2, Figure 5 | `e3_budget.py --fallback freehaze --q 0.1 0.2 0.5 --draws 20`, `triage_spread.py`, `plots/fig6_budget.py` | `results/e3_budget_freehaze.txt`, `results/triage_spread.txt` |
| Tables D1, D2 | `pipeline_table.py` | `results/pipeline_table.md` |
| Numbers with no table of their own | `misc_numbers.py`, `decidability_cheap.py` | `results/misc_numbers.txt`, `results/decidability_cheap.txt` |
| Real panel retrieved | `panel_retrieval.py --jobs 9`, then `--summarize` | `results/jwst_panel/panel_retrieval*.csv`, `panel_retrieval.txt` |
| Table {panelret} | `panel_retrieval.py`, `--summarize` | `results/jwst_panel/panel_retrieval.txt` |
| Table {missed}, existing rules as routers, transplant specificity, prior reach, classes against levels | `review_checks/*.py` | `results/review_checks/` |
| Tables {panel4}, {fourth}: the independent test | `jwst_panel.FOURTH`, `fourth_batch.py`; survey `jwst_fourth_batch_survey.md` | `results/jwst_panel_fourth/`, `results/jwst_panel/fourth_batch_second_reader.csv` |
The manuscript, the letters and the journal's Word files: `jhss/build_manuscript.py both`, `jhss/build_manuscript.py docx response_to_reviewer_comments.md`, `jhss/build_manuscript.py docx cover_letter.md`; separate JPEG figures: `jhss/export_figures.py`.

## The corrections and the matched ensemble (Sections 2.10 and 3.7)

The audit's corrections are in the code, except the catalogue mass, which is applied once to the built catalogue: `python mcs_corrections.py`.

Data, downloaded by hand into these folders; neither is committed:
- Goyal et al. (2019) generic grid, Google Drive folder 1ZFbkPdqg37_Om7ECSspSpEp5QrUMfA9J: `data/goyal2019/{chem_rainout.tar, trans_rainout.tar, chem_local.tar, trans_local.tar}`. Extract the rainout tars into `data/goyal2019/x/`. The local chemistry tar uses the same folder and file names as the rainout one, so extract it elsewhere and move its `chemfiles_txt` to `data/goyal2019/x/chemfiles_local`; `trans_local.tar` goes into `x/` as it is.
- Crossfield et al. (2025) grid, Zenodo 17101615, `final_model_grid.tar.gz`: extract into `data/crossfield2025/x/`, leaving out `*.vul` and `*mmr*`.

Order, each result under `results/`:

| Result | Command | Output |
|---|---|---|
| Populations | `broad_population.py 4`, `correlated_population.py 6` | `data/broad/`, `data/correlated/` |
| Rainout grid, older designs | `goyal_test.py --cond rainout` | `goyal_test/` |
| Candidates | `round3_learner.py --cand C-mlp`, `--cand C+B-mlp`, `--cand B+bal-vote` | `round3/` |
| Held-out grids | `goyal_test.py --cond local`, `crossfield_test.py --build`, `--score`, `round3_learner.py --cand C+B-mlp --cond rainout local crossfield` | `round3/`, `goyal_test/` |
| Grid scores | `review_checks/goyal_score.py <cond>` | `goyal_test/goyal_<cond>.txt` |
| Seeds | `round3_learner.py --cand C+B-mlp --cond rainout local crossfield --seed-offset 1`, `2` | `round3/seed1`, `seed2` |
| Band removal | `band_control.py`, then `review_checks/round3_checks.py` | `round3/round3_checks.txt` |
| Opacity removed | `crossfield_counterfactual.py` | `round3/counterfactual.txt` |
| Ariel coverage, Table 21 | `round3_ariel.py` | `round3/round3_ariel.txt` |
| Confidence | `review_checks/round3_confidence.py` | `round3/round3_confidence.txt` |
| Figure 6 | `plots/fig7_grids.py` | `figures/fig7_grids.png` |
| Earlier forms | `matched_learner.py`, `matched_ariel.py`, `broad_learner.py`, `independent_test.py`, `review_checks/matched_score.py`, `review_checks/broad_score.py` | `matched_learner/`, `goyal_test/`, `independent_test/` |

The real panel's dependent tables are rerun after a change of conversion in this order: `jwst_panel.py`, `alfnoor_retrain.py --panel` (remove `results/jwst_panel_{haze,hazedeck,union}` first; `WASP-39b_*` there comes from the benchmark assembly), `router_real.py`, `muting_router.py`, `amplitude_flag.py`, `panel_tier1.py`, `fourth_batch.py`, `jwst_muting_census.py`, `muting_bootstrap.py`, `jwst_controls.py`, `panel_mechanism.py`, `panel_control.py`, `label_sensitivity.py`, `pipeline_table.py`, `review_checks/*.py`.
