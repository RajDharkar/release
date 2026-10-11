# Round 5 camera-ready release audit

## Scope and starting state

[VERIFIED FROM ARTIFACT] Work was performed in the existing nested public-release clone at `/Users/rdharkar/dev/paper/data/release`; its `origin` fetch and push URL was `https://github.com/RajDharkar/release.git`, branch `main`, and starting commit `8fedbb1`. Evidence: `.git/config:lines 6-8`; `.git/HEAD:line 1`; pre-change `git rev-parse --short HEAD` output.

[COMPUTED DIAGNOSTIC] The exact pre-change output of `git status --short --untracked-files=all` in the release clone was empty (zero lines). Evidence: command capture made before the first edit; starting commit `8fedbb1`.

## A1 — origin and support for 1927

[VERIFIED FROM ARTIFACT] The Hughes loader itself assigns the value: `VOLUME_YEAR_ESTIMATE = 1927`, then emits that value for every title in its 24-title explicit list. Its adjacent note calls the value a best estimate tied to *Fine Clothes to the Jew* and says it is not verified from per-poem metadata. Evidence: `src/loaders/source6_pd_poetry_hughes.py:lines 55-68,70-96,129-145`.

[COMPUTED DIAGNOSTIC] The 24 Hughes rows carrying 1927 are exactly—not merely the same count as—the 24 rows whose label basis is `Hughes density-ranked explicit title list (overlapping narrow marker inventory)`; the exact-set equality over all 229 Hughes rows is true. Evidence: `data/analysis/hughes_year_assignment_audit.csv:rows 2-230`; `data/analysis/hughes_source_structure_audit.csv:rows 6-7`; verification assertion `scripts/verify_release.py:lines 223-226`.

[VERIFIED FROM ARTIFACT] The value therefore travels with the explicit ranked-list loader path, while the 14 later Hughes reclassifications retain blank numeric years and the source-bounded range in `year_basis`. Evidence: `src/loaders/reclassify_pd_poetry_density.py:lines 51-61,108-119`; `src/loaders/reclassify_hughes_2pct_and_fix_ids.py:lines 81-109`; `data/analysis/hughes_year_assignment_audit.csv:rows 26-39`.

[COMPUTED DIAGNOSTIC] The Standard Ebooks single-page source has 229 sibling `bodymatter` poem/song elements, zero collection-container elements, no exact `Fine Clothes to the Jew` collection-title string, and only the aggregate 1921–1928 publication range in its colophon. Evidence: `data/analysis/hughes_source_structure_audit.csv:rows 2-5`; original source locations recorded in that table are `data/raw/pd_poetry/hughes_se.html:lines 1249-9270,9276-9278`.

[NOT RECOVERABLE] No additional per-poem or per-collection publication year is recoverable from the fetched Standard Ebooks single-page artifact because it does not encode collection boundaries or per-poem years; resolving more rows would require a different bibliographic source and an authors' decision. The release consequently preserves the 24 loader estimates as explicitly unverified and leaves the other 205 Hughes numeric years blank. Evidence: `data/analysis/hughes_source_structure_audit.csv:rows 2-7`; `data/analysis/hughes_year_assignment_audit.csv:rows 2-230`.

## A2 — generator-license quotations

| Evidence | Generator / clause | Quotation | Location |
|---|---|---|---|
| [VERIFIED FROM ARTIFACT] | Llama 4 Community License §1(b)(i) | “copy of this Agreement”; “Built with Llama”; “any output or results”; “AI model”; “distributed or made available”; “include ‘Llama’ at the beginning”. | `DATA_TERMS.md:line 38` |
| [VERIFIED FROM ARTIFACT] | Llama 4 acceptable-use reference | “Acceptable Use Policy”. | `DATA_TERMS.md:line 39` |
| [VERIFIED FROM ARTIFACT] | Apache License 2.0 §4 for Qwen | “give any other recipients of the Work or Derivative Works a copy of this License”; “retain all copyright, patent, trademark, and attribution notices”. | `DATA_TERMS.md:line 40` |
| [VERIFIED FROM ARTIFACT] | MIT License for DeepSeek | “included in all copies or substantial portions of the Software”. | `DATA_TERMS.md:line 41` |

## B — data corrections

[COMPUTED DIAGNOSTIC] Every Dunbar row in the corpus, SAE-fold, and natural-score artifacts now uses numeric `volume_year=1922`; `year_basis` separately preserves 1913 as the latest component copyright year. Evidence: `data/corpus/dataset_poem_provenance.csv:rows 2-535`; `data/folds/rq1_full_poetry_sae_folds.csv:rows 2-475`; `data/scores/rq1_full_poetry_natural_scores.csv:rows 2-10792`; verification assertions `scripts/verify_release.py:lines 228-236`.

[VERIFIED FROM ARTIFACT] The mixed `pub_year`/`publication_era` fields were replaced in fold and natural-score tables by numeric `volume_year` plus textual `year_basis`; `volume_year` is documented as the parsed volume's publication year, not a composition date. Evidence: `data/folds/rq1_full_poetry_sae_folds.csv:row 1`; `data/scores/rq1_full_poetry_natural_scores.csv:row 1`; `README.md:lines 21-27`.

[COMPUTED DIAGNOSTIC] The corpus now contains 5 excluded Gullah rows with the exact reason `Gullah, excluded as a distinct variety`, while the retained 733 rows split into 212 AAVE and 521 SAE. The inclusive 80–200-word `rq4_in_band` flag yields 102 retained AAVE and 219 retained SAE rows. Evidence: `data/corpus/dataset_poem_provenance.csv:rows 2-739`; verification assertions `scripts/verify_release.py:lines 213-222`.

[COMPUTED DIAGNOSTIC] All 16 CSV tables containing `generator` use only the short lowercase join keys and also contain `model_requested` and `model_returned`; DeepSeek's requested/returned difference remains explicit. Evidence: schema scan implementation and zero-problem result, `scripts/verify_release.py:lines 189-210`; identifier documentation, `README.md:lines 29-34`.

[VERIFIED FROM ARTIFACT] The three `final_rq2_*` tables now name their LR/RF field `classifier`, not `model`. Evidence: `data/results/final_rq2_poetry_only_paired_changes.csv:row 1`; `data/results/final_rq2_poetry_only_seed_metrics.csv:row 1`; `data/results/final_rq2_poetry_only_summary.csv:row 1`.

[COMPUTED DIAGNOSTIC] A scan of all 50 CSV headers found no remaining ambiguous bare `model` column and no generator table missing either model identifier; no additional name/content mismatch was found. Evidence: `scripts/verify_release.py:lines 189-210`; command output `schema scan: 50 CSV files; 16 generator tables; 0 problems`.

[VERIFIED FROM ARTIFACT] Corpus provenance now carries `volume_year` and `year_basis`, while rewrite provenance and both generation manifests carry the source poem's numeric volume year; the 14 reclassified Hughes generation rows correctly remain blank in that numeric field. Evidence: `data/corpus/dataset_poem_provenance.csv:row 1`; `data/generation/rewrite_run_provenance.csv:row 1`; `data/transforms/sae_mgt_generation_manifest.csv:row 1`; `data/transforms/sae_mgt_generation_api_log_matches.csv:row 1`.

## C — released per-item artifacts and code

| Evidence | Paper-reported material | Released artifact and rows |
|---|---|---|
| [VERIFIED FROM ARTIFACT] | Quality battery | `data/analysis/quality_battery_per_item.csv:rows 2-940` (939 Pair A/B/C rows). |
| [VERIFIED FROM ARTIFACT] | Residual grammar | `data/analysis/residual_grammar_per_rewrite.csv:rows 2-210` (209 rewrites). |
| [VERIFIED FROM ARTIFACT] | Eye-dialect rate | `data/analysis/eye_dialect_rate_per_rewrite.csv:rows 2-210` (source and rewrite rate). |
| [VERIFIED FROM ARTIFACT] | Multi-VALUE descriptives | `data/analysis/multivalue_descriptives_per_poem.csv:rows 2-731` (521 human plus 209 machine rows). |
| [VERIFIED FROM ARTIFACT] | 35-feature measured profile | `data/analysis/measured_dialect_profiles_per_item.csv:rows 2-1255`; `data/analysis/measured_dialect_feature_directions.csv:rows 2-36`; `data/analysis/measured_dialect_profile_distances.csv:rows 2-64`. |
| [VERIFIED FROM ARTIFACT] | Poem-length sensitivity | `data/analysis/poem_length_sensitivity_summary.csv:rows 2-13`; `data/analysis/poem_length_balance_diagnostics.csv:rows 2-17`; `data/analysis/poem_length_baseline_verification.csv:rows 2-13`. |
| [VERIFIED FROM ARTIFACT] | Rewrite word counts | `data/analysis/rewrite_word_counts_per_item.csv:rows 2-210`. |
| [VERIFIED FROM ARTIFACT] | Loaders and analysis code | `src/loaders/source6_pd_poetry.py:lines 1-288`; `src/loaders/source6_pd_poetry_sae.py:lines 1-261`; `src/:39 tracked Python files across released source packages`; module map at `src/README.md:lines 1-22`. |
| [VERIFIED FROM ARTIFACT] | Previously held text | `data/text/sae_mgt_rewrites.csv:rows 2-210`; `data/transforms/rq1_proxy_sae521_multivalue_canonical.csv:rows 2-522`; `data/transforms/rq3_sae_mgt_multivalue_v013.csv:rows 2-210`; `data/transforms/rq3_sae_mgt_multivalue_v03.csv:rows 2-210`. |

[COMPUTED DIAGNOSTIC] The residual-grammar table reproduces 105 nonzero rewrites: 40 `llama`, 33 `qwen3`, and 32 `deepseek`. Evidence: `data/analysis/residual_grammar_per_rewrite.csv:rows 2-210`; assertions `scripts/verify_release.py:lines 255-262`.

[COMPUTED DIAGNOSTIC] Median source-to-rewrite eye-dialect rates are 0.0976828714→0 for `deepseek`, 0.0909090909→0 for `llama`, and 0.1058823529→0 for `qwen3`. Evidence: `data/analysis/eye_dialect_rate_per_rewrite.csv:rows 2-210` grouped by `generator`.

[COMPUTED DIAGNOSTIC] Multi-VALUE per-poem values reproduce the reported human-SAE and SAE-MGT descriptives: 521/209 pairs, zero byte-identical pairs, mean distinct features 9.080614/40.473684, and mean rule executions 26.510557/177.655502. Evidence: `data/analysis/multivalue_descriptives_per_poem.csv:rows 2-731` grouped by `corpus`.

[COMPUTED DIAGNOSTIC] The released per-item rewrite counts reproduce medians 190 before and 194 after machine Multi-VALUE transformation. Evidence: `data/analysis/rewrite_word_counts_per_item.csv:rows 2-210`; assertions `scripts/verify_release.py:lines 268-270`.

[COMPUTED DIAGNOSTIC] Pair A candidate hashes and Pair C reference hashes each resolve exactly to the same 209 `completion` texts in the released rewrite file. Evidence: `data/analysis/quality_battery_per_item.csv:rows 2-940`; `data/text/sae_mgt_rewrites.csv:rows 2-210`; hash-link assertions `scripts/verify_release.py:lines 244-253`.

[VERIFIED FROM ARTIFACT] The README records both SequenceMatcher `autojunk` values, the Levenshtein denominator, BERTScore layer/baseline settings, AlignScore `nli_sp` mode and both checkpoint hashes, and the 512-subword context-only truncation setting. Evidence: `README.md:lines 55-75`.

## D — documentation and annotation provenance

[VERIFIED FROM ARTIFACT] `HELD_BACK.md` was deleted; `README.md` and `DATA_TERMS.md` now describe released text, mixed provenance, generator-specific terms, and the unresolved overall data-license placeholder. Evidence: deleted tracked path `HELD_BACK.md`; `README.md:lines 1-12,36-53`; `DATA_TERMS.md:lines 1-10,27-51`.

[VERIFIED FROM ARTIFACT] The code-only MIT `LICENSE` is now accurate because the release contains loader and analysis source code, while data terms remain separate. Evidence: `LICENSE:lines 1-5`; `src/README.md:lines 1-22`; `DATA_TERMS.md:lines 1-10`.

[VERIFIED FROM ARTIFACT] The blinded 60-item annotation set was constructed from `data/processed/aave_poetry.csv`, specifically the 24 ranked plus 36 reclassified processed-corpus rows, and not from the public release repository. Evidence: `data/analysis/annotation_set_provenance_audit.csv:rows 2-7`; underlying build audit `reports/round4.md:lines 5,510-520` in the analysis workspace.

## Verification and delivery record

[COMPUTED DIAGNOSTIC] The round-5 data assertions pass: retained corpus 212/521, in-band 102/219, quality 939, rewrites 209, both 209-row trace links exact, residual grammar 105, Multi-VALUE 521+209, and word-count medians 190/194. Evidence: `scripts/verify_release.py:lines 213-275`; successful `--data-check` output.

[COMPUTED DIAGNOSTIC] All 39 released Python modules parsed and compiled without syntax errors. Evidence: `src/:39 Python files`; successful `python -m compileall -q src scripts` output with cache directed outside the repository.

[COMPUTED DIAGNOSTIC] The credential scan checked all 105 tracked files using seven high-confidence secret patterns and reported zero matches. Evidence: `scripts/verify_release.py:lines 51-60,174-186`; successful output `credential scan: 105 tracked files checked; 0 matches`.

[COMPUTED DIAGNOSTIC] All 104 tracked payloads except the self-referential manifest are listed with current row/column counts and SHA-256; verification reports an exact path-set and digest match. Evidence: `MANIFEST.csv:rows 2-105`; `scripts/verify_release.py:lines 153-171`; successful output `manifest verified: 104 of 104 payload files`.

[VERIFIED FROM ARTIFACT] The remote repository identifies itself as public, and no visibility-setting operation was performed. Evidence: `https://github.com/RajDharkar/release` repository header; `git remote -v` origin URL.

[COMPUTED DIAGNOSTIC] The substantive work is grouped into commits `c2dd180` (code) and `0fc477e` (corrected schemas and artifacts), followed by the documentation/audit commit containing this report. Evidence: `git log --oneline`; the report commit cannot include its own final hash without creating a self-reference.

[COMPUTED DIAGNOSTIC] The final post-push output of `git status --short --untracked-files=all` is empty (zero lines). Evidence: final command capture after the last push.
