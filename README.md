# Voice poetry benchmark release

This repository contains the camera-ready data, text artifacts, provenance,
frozen detector outputs, and analysis code for an evaluation of machine-text
detectors on poetry written in African American English and Standard American
English. The release includes all 209 retained SAE machine-generated rewrites,
the 521 human Multi-VALUE outputs, and both 209-row machine Multi-VALUE outputs.
No detector or transformation was rerun while assembling this bundle.

`DATA_TERMS.md` records the distinct source and generator terms. The repository
does not assert one blanket license over its mixed-provenance contents. `LICENSE`
applies only to source code.

## Corpus and publication metadata

The corpus inventory contains 738 poems: 212 retained AAVE, 521 retained SAE,
and 5 excluded Gullah poems. `retained`, `exclusion_reason`, and `rq4_in_band`
make those analysis decisions explicit. The RQ4 band is 80--200 stored words,
inclusive, and contains 102 retained AAVE and 219 retained SAE poems.

`volume_year` is the publication year of the parsed volume, not a poem's
composition date. `year_basis` records how the value was established. The
parsed Dunbar volume's title page is 1922; 1913 is retained in `year_basis` as
the latest component copyright year. For Hughes, the 24 explicit ranked-list
rows retain the loader's 1927 estimate and are explicitly marked as not
source-verified. The other Hughes rows have a blank numeric year and a
source-bounded 1921--1928 basis.

## Generator identifiers

`generator` is the lowercase join key: `qwen3`, `llama`, or `deepseek`. Every
table containing it also carries `model_requested` and `model_returned`.
DeepSeek was requested as `deepseek-ai/DeepSeek-V3.2` and returned as
`deepseek/deepseek-v3.2`; the distinction is preserved on every applicable row.

## Contents

| Directory | Contents |
|---|---|
| `data/corpus/` | Poem-level provenance, labels, retention, word counts, and volume-year basis. |
| `data/text/` | All 209 retained rewrites, with source poem and generator linkage. |
| `data/analysis/` | Per-item quality, residual grammar, eye-dialect, Multi-VALUE, measured-profile, length, and word-count artifacts. |
| `data/folds/` | Frozen SAE fold assignments. |
| `data/generation/` | Rewrite-run provenance and safe provider-response metadata. |
| `data/transforms/` | Human and machine Multi-VALUE outputs plus transformation provenance. |
| `data/scores/`, `data/thresholds/`, `data/results/` | Frozen detector scores, thresholds, and reported results. |
| `src/` | Corpus loaders, transformations, feature extraction, and analysis programs. |
| `scripts/` | Deterministic round-5 release assembly and verification support. |

`MANIFEST.csv` inventories every payload file except itself, with row and column
counts, SHA-256 digests, and one-line descriptions. CSV row counts exclude the
header. A JSON object counts as one row, and Markdown/text/Python files count
physical lines as rows.

## Quality-battery settings

`data/analysis/quality_battery_per_item.csv` contains all 939 Pair A/B/C rows.
Character edit percentage is reported with `difflib.SequenceMatcher` under both
`autojunk=True` and `autojunk=False`. Character Levenshtein distance is divided
by `max(len(reference), len(candidate))`.

BERTScore uses `roberta-large`, layer 17, `idf=False`. Raw P/R are recovered
from the frozen rescaled results and the package baseline; raw F1 is the frozen
raw score. Rescaled P/R/F1 use `rescale_with_baseline=True` and
`bert_score/rescale_baseline/en/roberta-large.tsv` (SHA-256
`08e2248310d0c25d8e22ef65e0a6be15060269f1a9084e560c258f09a9e122ae`).

AlignScore uses evaluation mode `nli_sp`. The base result uses the official
`AlignScore-base.ckpt` (SHA-256
`6aedb637f0596ab29baef91e94466a57f032e02feea654978518919fe0981607`)
with RoBERTa-base; the large result uses the official `AlignScore-large.ckpt`
(SHA-256
`ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d`)
with RoBERTa-large. AlignScore tokenization uses a 512-subword budget with
context-only truncation (`truncation='only_first'`).

## Rebuilding the poetry pools

Install `requirements.txt`, then place the four source downloads and Dunbar
index at the paths below:

```text
data/raw/pd_poetry/dunbar_complete.txt       Project Gutenberg 18338
data/raw/pd_poetry/dunbar_dialect_index.html paullaurencedunbar.org/2022/plangbd
data/raw/pd_poetry/henderson1904.txt         Project Gutenberg 66997
data/raw/pd_poetry/johnson1922.txt           Project Gutenberg 11986
data/raw/pd_poetry/hughes_se.html            Standard Ebooks single-page edition
```

From the repository root, run:

```bash
python -m src.loaders.source6_pd_poetry
python -m src.loaders.source6_pd_poetry_sae
python -m src.loaders.reclassify_pd_poetry_density
python -m src.loaders.reclassify_hughes_2pct_and_fix_ids
```

The first two commands parse the explicit source lists and complements; the
last two apply the documented 5% and Hughes 2% reclassifications that produce
the released 212/521 retained pools. `src/README.md` maps the released analysis
programs to their artifacts. Multi-VALUE reproduction additionally requires
the pinned environment recorded in the transformation provenance JSON files.

The raw provider request/response logs are not distributed. The response
metadata table excludes authorization material and other credentials.
