"""Freeze SAE-MGT provenance and derive the RQ3 Multi-VALUE v0.3 condition.

This producer is deliberately inference-free.  It first reproduces a fixed
prefix of the canonical RQ2 human transformation byte-for-byte, then applies
that exact retained implementation to the frozen 209 SAE-MGT rewrites.
Historical inputs and v0.1.3 outputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perturb.rq1_proxy_sae521_multivalue import (
    build_dialect,
    ordered_population,
    sha256_file,
    sha256_text,
    transform_row,
    verify_environment,
)


OUT = ROOT / "outputs" / "pre_rewrite_methodology_resolution_2026-10-01"
RAW = ROOT / "data/interim/aave_poetry_mgt_modeb_run1.csv"
SOURCE = ROOT / "data/processed/aave_poetry.csv"
CANONICAL = ROOT / "data/analysis/rq1_proxy_sae521_multivalue_canonical.csv"
FOLDS = ROOT / "data/analysis/rq1_full_poetry_natural_scores.csv"
RQ2_MANIFEST = ROOT / "data/analysis/rq1_proxy_sae521_multivalue_provenance_manifest.json"
REGRESSION = OUT / "rq2_v03_transformation_regression.csv"
REGRESSION_STATE = OUT / "rq2_v03_transformation_regression.json"
TRANSFORMED = OUT / "rq3_transformed_sae_mgt_multivalue_v03.csv"
MANIFEST = OUT / "sae_mgt_generation_manifest.csv"
PROVENANCE = OUT / "sae_mgt_generation_provenance.md"
N_REGRESSION = 10


def write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite preserved artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite preserved artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def valid_mask(raw: pd.DataFrame) -> pd.Series:
    return (
        raw["completion"].astype(str).str.strip().ne("")
        & raw["error"].astype(str).str.strip().eq("")
        & ~raw["refusal_bool"].astype(bool)
    )


def generation_manifest() -> pd.DataFrame:
    raw = pd.read_csv(RAW, keep_default_na=False)
    source = pd.read_csv(SOURCE, keep_default_na=False)
    source = source[["id", "poet", "source_dataset", "source_subset", "poem_title"]]
    usable = valid_mask(raw)
    frame = raw.merge(source, left_on="source_poem_id", right_on="id", how="left", validate="many_to_one")
    frame["source_author"] = frame["poet"]
    frame["source"] = frame["source_dataset"] + ":" + frame["source_subset"]
    frame["generator"] = frame["model_short_name"]
    frame["prompt_template_id"] = "poetry_b_" + frame["template_version"].astype(str)
    frame["usable"] = usable.to_numpy()
    frame["exclusion_reason"] = ""
    frame.loc[frame["completion"].astype(str).str.strip().eq(""), "exclusion_reason"] = "empty completion"
    frame.loc[frame["error"].astype(str).str.strip().ne(""), "exclusion_reason"] = "stored generation error"
    frame.loc[frame["refusal_bool"].astype(bool), "exclusion_reason"] = "frozen poetry refusal/filter flag"
    frame["final_analysis_included"] = frame["usable"]
    frame["prompt_sha256"] = frame["prompt"].map(sha256_text)
    frame["completion_sha256"] = frame["completion"].map(sha256_text)
    return frame[[
        "generation_id", "source_poem_id", "source_author", "source", "poem_title",
        "generator", "model_full_id", "prompt_template_id", "prompt_sha256",
        "completion_sha256", "usable", "exclusion_reason", "final_analysis_included",
    ]].sort_values("generation_id").reset_index(drop=True)


def provenance_markdown(manifest: pd.DataFrame) -> str:
    raw = pd.read_csv(RAW, keep_default_na=False)
    excluded = raw.loc[~valid_mask(raw)].iloc[0]
    counts = manifest.loc[manifest["final_analysis_included"]].groupby(["generator", "model_full_id"]).size()
    count_lines = "\n".join(f"- `{model}` (`{short}`): {int(n)}" for (short, model), n in counts.items())
    literal = (
        "Rewrite the following poem in standard written English. Convert any non-standard spellings, grammar, and vocabulary into standard forms, while keeping the meaning, imagery, and emotional weight intact. Output only the rewritten poem.\n\n"
        "--- SOURCE TEXT ---\n{SOURCE_POEM}\n--- END SOURCE TEXT ---\n\n"
        "Keep the rewrite close to the original's length."
    )
    return f"""# SAE-MGT generation provenance

## Status and accurate label

**VERIFIED:** “The SAE-MGT poetry condition consists of machine-generated standard-English rewrites of human-authored AAVE source poems.” The repository's source table identifies the inputs as human AAVE poetry; the literal prompt asks for “standard written English” but does not use `SAE`, `Standard American English`, `AAVE`, `dialect`, or `vernacular`. The precise manuscript label should therefore be **SAE MGT rewrites** (or **machine-generated standard-English rewrites**), with the literal prompt wording disclosed. The artifacts support saying the models were not explicitly told the source variety; they do not support a broader claim about what a model inferred from the text.

## Exact workflow

`src/generate/run_poetry_modeb_run1.py` loaded the 212 `variety == AAVE` and `label == human` rows from `data/processed/aave_poetry.csv` (five Gullah rows excluded), selected the 210 IDs with the smallest SHA-256 digests, and assigned them round-robin in processing order to Qwen, Llama, and DeepSeek (70 attempts each). It rendered `poetry_b_v3`, inserted the full source poem, and sent one user message through the Hugging Face OpenAI-compatible chat-completions endpoint. There was **no system prompt**.

## Literal v3 base user template

```text
{literal}
```

For Qwen only, the stored/sent user content appended a literal space plus `/no_think`; the client also set `chat_template_kwargs={{"enable_thinking": false}}`. Llama and DeepSeek received the base template without that suffix. All 210 rows record template version `v3`; no other template contributed to the retained 209.

## Models and retained counts

{count_lines}

The generation configuration was temperature `0.9`, maximum output tokens `1200`, and no explicit `top_p`. The request contained `messages=[{{"role":"user","content":...}}]`, the exact model ID, `temperature`, and `max_tokens`; Qwen additionally received the thinking-control field above. Model revision hashes were not recorded and cannot be recovered from the generation table.

## Cleaning and exclusion

The response content was retained as the completion. The shared client removes a trailing `/no_think` artifact from Qwen output if present; no other prose, whitespace, spelling, or punctuation normalization is performed. Density and similarity fields are derived measurements, not filters. Empty text, provider errors, thinking leakage during the live run, and poetry-local refusal flags were failure gates.

The raw file has 210 attempts. Exactly one row was excluded by the frozen refusal decision: `{excluded['generation_id']}` (`{excluded['model_full_id']}`). Its stored completion is nonempty and its stored error is empty, but `refusal_bool=True`; the poetry heuristic flags a refusal-pattern match in a response shorter than 40 whitespace words or within the first 20 characters. The decision is preserved even though later audit notes identify this as a conservative false positive. Thus the final counts are Qwen 69, Llama 70, DeepSeek 70 = 209.

## Authoritative repository artifacts

- Raw generation table: `data/interim/aave_poetry_mgt_modeb_run1.csv`
- Source-poem table and metadata: `data/processed/aave_poetry.csv`
- Generation producer and selection/filter logic: `src/generate/run_poetry_modeb_run1.py`
- Literal template renderer/version: `src/generate/templates.py`
- Request construction, response capture, and Qwen cleanup: `src/generate/harness.py`
- Model configuration schema: `src/generate/config.py`
- API request/response logs: `data/interim/live_raw/*.json`
- Append-only call ledger: `data/interim/live_calls_log.csv`
- Historical v0.1.3 transformed-MGT mapping: `data/interim/arm_d_poetry_aave_mgt_run1.csv`
- Frozen analysis inclusion/text table: `data/interim/rq5_subsample.csv`
- New audited inclusion manifest: `outputs/pre_rewrite_methodology_resolution_2026-10-01/sae_mgt_generation_manifest.csv`

## Recoverability limits

The model repository IDs, prompt contents, parameters, token counts, finish reasons, timestamps, source mappings, git commit, and dirty-worktree flag are recorded. Provider-side infrastructure state, exact model weight revisions, nondeterministic sampling seeds, and any undocumented provider defaults are not recoverable. The MGT text is frozen and was not regenerated in this audit.
"""


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = generation_manifest()
    assert len(manifest) == 210 and int(manifest["final_analysis_included"].sum()) == 209
    assert manifest.loc[manifest["final_analysis_included"]].groupby("generator").size().to_dict() == {
        "deepseek": 70, "llama": 70, "qwen3": 69,
    }
    write_csv_new(MANIFEST, manifest)
    write_text_new(PROVENANCE, provenance_markdown(manifest))
    print(f"wrote provenance and {len(manifest)}-attempt manifest ({manifest.final_analysis_included.sum()} included)")


def regress() -> None:
    env = verify_environment()
    rows = ordered_population().head(N_REGRESSION)
    frozen = pd.read_csv(CANONICAL, keep_default_na=False).set_index("source_doc_id")
    dialect = build_dialect()
    records = []
    for _, row in rows.iterrows():
        rec = transform_row(dialect, row)
        expected = frozen.loc[rec["source_doc_id"]]
        records.append({
            "canonical_batch_index": rec["canonical_batch_index"],
            "source_doc_id": rec["source_doc_id"],
            "expected_sha256": expected["transformed_sha256"],
            "rerun_sha256": rec["transformed_sha256"],
            "byte_exact": rec["transformed_text"] == expected["transformed_text"],
        })
    result = pd.DataFrame(records)
    passed = bool(result["byte_exact"].all() and (result["expected_sha256"] == result["rerun_sha256"]).all())
    write_csv_new(REGRESSION, result)
    state = {
        "status": "PASS" if passed else "FAIL",
        "sample_definition": f"first {N_REGRESSION} rows of canonical SHA-256 batch order",
        "sample_n": N_REGRESSION,
        "byte_exact_n": int(result["byte_exact"].sum()),
        "environment": env,
        "canonical_file_sha256": sha256_file(CANONICAL),
        "rq2_provenance_manifest_sha256": sha256_file(RQ2_MANIFEST),
    }
    write_text_new(REGRESSION_STATE, json.dumps(state, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("Canonical RQ2 v0.3 regression failed; RQ3 transformation is prohibited")
    print(f"PASS: {N_REGRESSION}/{N_REGRESSION} canonical human outputs reproduced byte-for-byte")


def transform() -> None:
    state = json.loads(REGRESSION_STATE.read_text(encoding="utf-8"))
    if state.get("status") != "PASS":
        raise RuntimeError("No passing RQ2 transformation regression; refusing RQ3 transformation")
    verify_environment()
    raw = pd.read_csv(RAW, keep_default_na=False)
    raw = raw.loc[valid_mask(raw)].copy()
    folds = pd.read_csv(FOLDS, keep_default_na=False)[["poem_id", "evaluation_fold"]].drop_duplicates()
    assert folds["poem_id"].nunique() == 733
    raw = raw.merge(folds, left_on="source_poem_id", right_on="poem_id", validate="many_to_one")
    raw["ordering_sha256"] = raw["generation_id"].map(sha256_text)
    raw = raw.sort_values(["ordering_sha256", "generation_id"], kind="stable").reset_index(drop=True)
    dialect = build_dialect()
    records = []
    config = {
        "distribution": "value-nlp 0.3", "dialect": "AfricanAmericanVernacular",
        "coreference": "disabled by AfricanAmericanVernacularNoCoref",
        "instance_policy": "one dialect instance for full batch",
        "batch_order": "ascending SHA-256(generation_id)",
        "seed": "int(sha256(generation_id)[:8],16) or 1",
        "input_normalization": "none", "output_normalization": "none",
    }
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    for i, row in raw.iterrows():
        payload = pd.Series({
            "canonical_batch_index": i,
            "source_doc_id": row["generation_id"],
            "poem_id": row["source_poem_id"],
            "sae_fold": int(row["evaluation_fold"]),
            "text": row["completion"],
        })
        rec = transform_row(dialect, payload)
        if not rec["transformation_success"]:
            raise RuntimeError(f"Transformation failed for {row['generation_id']}: {rec['error_message']}")
        records.append({
            "transformed_generation_id": "rq3_mv03_" + row["generation_id"],
            "generation_id": row["generation_id"],
            "source_poem_id": row["source_poem_id"],
            "generator": row["model_short_name"],
            "model_full_id": row["model_full_id"],
            "evaluation_fold": int(row["evaluation_fold"]),
            "canonical_batch_index": i,
            "original_sae_mgt_text": row["completion"],
            "transformed_sae_mgt_text": rec["transformed_text"],
            "transformation_version": "Multi-VALUE/value-nlp 0.3",
            "transformation_configuration_metadata": config_json,
            "transformation_seed": rec["seed"],
            "candidate_rule_count": rec["candidate_rule_count"],
            "executed_rule_count": rec["executed_rule_count"],
            "input_sha256": rec["original_sha256"],
            "output_sha256": rec["transformed_sha256"],
        })
        if (i + 1) % 10 == 0 or i + 1 == len(raw):
            print(f"transformed {i + 1}/{len(raw)}", flush=True)
    out = pd.DataFrame(records)
    assert len(out) == 209 and out["generation_id"].nunique() == 209
    assert out["input_sha256"].eq(out["original_sae_mgt_text"].map(sha256_text)).all()
    assert out["output_sha256"].eq(out["transformed_sae_mgt_text"].map(sha256_text)).all()
    write_csv_new(TRANSFORMED, out)
    print(f"wrote {TRANSFORMED} sha256={sha256_file(TRANSFORMED)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "regress", "transform"])
    args = parser.parse_args()
    {"prepare": prepare, "regress": regress, "transform": transform}[args.phase]()


if __name__ == "__main__":
    main()
