# Data terms and provenance

There is no single license for everything in this repository. The `LICENSE`
file applies only to source code. The tables here combine factual metadata,
derived measurements, hashes, and provenance from sources with different
terms. Anyone redistributing or extending the data should review the terms for
each source and model involved.

## Source poems

Three source volumes came from Project Gutenberg:

- *The Complete Poems of Paul Laurence Dunbar*, ebook 18338.
- *Plantation Echoes*, ebook 66997.
- *The Book of American Negro Poetry*, ebook 11986.

Each downloaded ebook says that it may be copied, given away, or reused under
the Project Gutenberg License. The license explains that works not protected
by United States copyright law may be copied and distributed in the United
States without a fee, while uses retaining the Project Gutenberg name and
materials remain subject to the license and trademark conditions. The current
license is at https://www.gutenberg.org/policy/license.

The fourth source was Standard Ebooks' *Poetry* by Langston Hughes, identifier
`langston-hughes/poetry`. Its Uncopyright page says that the writing and artwork
are believed to be in the United States public domain, warns users outside the
United States to check local law, and places the edition contributors' own work
under CC0 1.0. That dedication does not change the copyright status of the
source text or artwork. The edition page is at
https://standardebooks.org/ebooks/langston-hughes/poetry.

No poem full text is included in this repository. The release does not make a
legal determination about later redistribution of that text.

## Model-produced material

The generation provenance names three requested models:

- `Qwen/Qwen3.5-9B`, whose model card identifies Apache-2.0.
- `meta-llama/Llama-4-Scout-17B-16E-Instruct`, distributed under the Llama 4 Community License.
- `deepseek-ai/DeepSeek-V3.2`, whose model card identifies MIT.

The model cards and licenses govern the models and may impose conditions that
are not captured by the code license in this repository. The rewrite bodies
and both sets of Multi-VALUE transformed text are held back while the Llama 4
Community License redistribution question remains unresolved. `HELD_BACK.md`
names the exact source artifacts.

## Included measurements and metadata

The included CSV and JSON files contain identifiers, labels, word counts,
folds, thresholds, hashes, detector scores, response metadata, and frozen
aggregate results. They do not include authorization headers, credentials,
generation prompts, completion text, or full source-poem text. No blanket
license is asserted over these data artifacts.

