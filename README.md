# Voice poetry benchmark data

This repository contains the metadata and frozen numerical artifacts for an
evaluation of machine-text detectors on poetry written in African American
English and Standard American English. It is intended to make the evaluation
structure inspectable without distributing source poems or generated rewrites.

The corpus inventory records poem IDs, source volumes, variety labels, the
basis for each label, per-poem word counts, retention decisions, and exclusion
reasons. The numerical files provide fold assignments, calibration thresholds,
detector scores, operating-point counts, transformation provenance, and frozen
result tables. No detector was rerun to prepare this repository; the files are
copies of the frozen analysis artifacts or metadata extracted from stored
responses.

## Source volumes

The source inventory covers four volumes:

- *The Complete Poems of Paul Laurence Dunbar*, Project Gutenberg ebook 18338.
- *Plantation Echoes*, Project Gutenberg ebook 66997.
- *The Book of American Negro Poetry*, Project Gutenberg ebook 11986.
- *Poetry* by Langston Hughes, Standard Ebooks identifier `langston-hughes/poetry`.

`DATA_TERMS.md` records the distinct source and model terms. Full poem text is
not included.

## Files

`MANIFEST.csv` is the complete payload inventory, with row and column counts,
SHA-256 digests, and short descriptions. The main groups are:

| Directory | Contents | Files |
|---|---|---:|
| `data/corpus/` | 738 poem-level provenance rows with IDs, labels, label basis, and word counts | 1 |
| `data/generation/` | 210 rewrite-provenance rows and 210 safe response-metadata rows | 2 |
| `data/folds/` | 521 SAE fold assignments | 1 |
| `data/thresholds/` | 75 primary threshold rows and 450 operating-point threshold rows | 2 |
| `data/scores/` | Natural, proxy, paired, and transformed detector-score tables | 4 |
| `data/results/` | Frozen RQ1, RQ2, RQ3, operating-point, and regression tables | 19 |
| `data/transforms/` | Batch order, provenance, reproducibility checks, hashes, and generation manifests | 7 |

The release preserves source row order. CSV counts in the manifest exclude the
header. A JSON object is counted as one row, and its top-level keys are counted
as columns. Text and Markdown files are counted by physical line with one text
column. The manifest does not list itself because a file cannot contain its own
stable cryptographic digest.

## Held-back material

The 209 retained SAE machine-generated rewrites and the Multi-VALUE transformed
human and machine outputs are not included. `HELD_BACK.md` names the exact
source files. They remain withheld because the Llama 4 Community License
question for redistribution of model outputs has not been resolved.

The raw request and response logs are also not distributed. The safe metadata
table records the response fields needed for provenance without prompts,
completions, authorization material, or other credentials.

## Using the tables

Join poem-level files on `poem_id` or `source_doc_id` as indicated by each
header. Detector scores use the stored orientation, thresholds, and boolean
decisions; consumers should not assume that a higher raw score has the same
meaning for every detector. Transformation tables use hashes and canonical
batch indices to link derived measurements while keeping text out of the
repository.

