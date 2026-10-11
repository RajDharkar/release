# Data terms and provenance

There is no single license for everything in this repository. `LICENSE` applies
only to source code. The data combine source poems, model outputs, factual
metadata, derived measurements, hashes, and provenance under different terms.

> **OVERALL DATA LICENSE: TO BE DECIDED BY THE AUTHORS.**

This placeholder is intentional and does not grant or deny rights beyond the
terms attached to each underlying source.

## Source poems

The Dunbar, Henderson, and Johnson volumes come from Project Gutenberg ebooks
18338, 66997, and 11986. Project Gutenberg's current license and trademark
terms are at <https://www.gutenberg.org/policy/license>. The source files state
that they may be copied, given away, or reused under that license; use retaining
the Project Gutenberg name or materials remains subject to its conditions.

The Hughes source is Standard Ebooks' *Poetry*, identifier
`langston-hughes/poetry`: <https://standardebooks.org/ebooks/langston-hughes/poetry>.
Its Uncopyright page describes the writing and artwork as believed to be in the
United States public domain, tells users outside the United States to check
local law, and dedicates the edition contributors' work under CC0 1.0. That
dedication does not change the status of the source writing or artwork.

Full source text occurs in the released transformation pairs and, where embedded
in a prompt, in the rewrite table. These source-specific terms therefore remain
attached to those columns.

## Generator terms and exact excerpts

The excerpts below are transcribed from the linked license texts. They are
provided without a repository-level compliance conclusion.

| Generator and released rows | Governing text | Exact excerpt relevant to redistribution or downstream model work |
|---|---|---|
| `llama` — `meta-llama/Llama-4-Scout-17B-16E-Instruct` | [Llama 4 Community License, section 1(b)(i)](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE) | “copy of this Agreement”; “Built with Llama”; “any output or results”; “AI model”; “distributed or made available”; “include ‘Llama’ at the beginning”. |
| `llama` acceptable-use reference | [Llama 4 Acceptable Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama4/USE_POLICY.md) | “Acceptable Use Policy”. |
| `qwen3` — `Qwen/Qwen3.5-9B` | [Model card](https://huggingface.co/Qwen/Qwen3.5-9B), license `apache-2.0`; [Apache License 2.0, section 4](https://www.apache.org/licenses/LICENSE-2.0) | “give any other recipients of the Work or Derivative Works a copy of this License”; “retain all copyright, patent, trademark, and attribution notices”. |
| `deepseek` — requested `deepseek-ai/DeepSeek-V3.2`, returned `deepseek/deepseek-v3.2` | [Model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), license `mit`; [MIT License](https://opensource.org/license/mit) | “included in all copies or substantial portions of the Software”. |

## Included metadata and generated text

The bundle includes identifiers, labels, word counts, folds, thresholds, hashes,
detector scores, safe response metadata, all 209 retained rewrites, and the human
and machine Multi-VALUE outputs. It does not include authorization headers,
credentials, or raw provider request/response logs.

No blanket license is asserted over the data artifacts or model outputs. Review
the source and generator terms above before redistribution or downstream use.
