# Held-back files

The following source artifacts are not included in this repository:

- `data/interim/aave_poetry_mgt_modeb_run1.csv` — 210 generation records, of which 209 SAE machine-generated rewrites were retained.
- `data/analysis/rq1_proxy_sae521_multivalue_canonical.csv` — Multi-VALUE transformed outputs for 521 human-written SAE poems.
- `data/interim/arm_d_poetry_aave_mgt_run1.csv` — Multi-VALUE transformed outputs for 209 SAE machine-generated rewrites using the historical transformation version.
- `outputs/pre_rewrite_methodology_resolution_2026-10-01/rq3_transformed_sae_mgt_multivalue_v03.csv` — Multi-VALUE transformed outputs for the same 209 SAE machine-generated rewrites using version 0.3.

These files are held back because the permissibility of redistributing model
outputs under the Llama 4 Community License has not been resolved. The release
therefore keeps provenance, hashes, detector scores, and aggregate result
tables without distributing the rewrite bodies or their transformed text.

Full poem text from the AAVE and SAE source pools is also absent. The repository
contains poem identifiers, labels, word counts, and source-volume metadata only.
The source-text redistribution question is documented in `DATA_TERMS.md`.

