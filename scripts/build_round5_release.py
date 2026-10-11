#!/usr/bin/env python3
"""Build the camera-ready round-5 release artifacts from frozen workspace files.

The script never reruns a detector or a transformation. It normalizes schemas,
joins frozen per-item measurements, and asserts every count and linkage requested
for the public release.
"""
from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


RELEASE = Path(__file__).resolve().parents[1]
WORK = RELEASE.parents[1]

MODEL_IDS = {
    "qwen3": ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
    "llama": (
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ),
    "deepseek": ("deepseek-ai/DeepSeek-V3.2", "deepseek/deepseek-v3.2"),
}
GENERATOR_ALIASES = {
    "Qwen": "qwen3",
    "Qwen/Qwen3.5-9B": "qwen3",
    "qwen3": "qwen3",
    "Llama": "llama",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama",
    "llama": "llama",
    "DeepSeek": "deepseek",
    "deepseek-ai/DeepSeek-V3.2": "deepseek",
    "deepseek/deepseek-v3.2": "deepseek",
    "deepseek": "deepseek",
}


def read_work(relative: str) -> pd.DataFrame:
    return pd.read_csv(WORK / relative)


def read_release(relative: str) -> pd.DataFrame:
    return pd.read_csv(RELEASE / relative)


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    path = RELEASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def normalize_generator(values: pd.Series) -> pd.Series:
    out = values.map(GENERATOR_ALIASES)
    if out.isna().any():
        raise AssertionError(f"unknown generator labels: {values[out.isna()].unique()}")
    return out


def add_model_ids(frame: pd.DataFrame, generator_col: str = "generator") -> pd.DataFrame:
    frame = frame.copy()
    frame[generator_col] = normalize_generator(frame[generator_col])
    frame["model_requested"] = frame[generator_col].map(lambda x: MODEL_IDS[x][0])
    frame["model_returned"] = frame[generator_col].map(lambda x: MODEL_IDS[x][1])
    return frame


def insert_after(frame: pd.DataFrame, after: str, names: list[str]) -> pd.DataFrame:
    columns = list(frame.columns)
    for name in names:
        columns.remove(name)
    at = columns.index(after) + 1
    columns[at:at] = names
    return frame[columns]


def publication_fields(corpus: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    year = pd.Series(pd.NA, index=corpus.index, dtype="Int64")
    basis = pd.Series("", index=corpus.index, dtype="string")

    masks = {
        "The Complete Poems of Paul Laurence Dunbar": (
            1922,
            "volume title page (1922); latest component copyright year 1913",
        ),
        "Plantation Echoes": (1904, "volume title page (1904)"),
        "The Book of American Negro Poetry": (1922, "volume title page (1922)"),
    }
    for volume, (value, reason) in masks.items():
        m = corpus["source_volume"].eq(volume)
        year.loc[m] = value
        basis.loc[m] = reason

    hughes = corpus["source_volume"].eq("Poetry (Standard Ebooks)")
    ranked = corpus["labeling_basis"].eq(
        "Hughes density-ranked explicit title list (overlapping narrow marker inventory)"
    )
    year.loc[hughes & ranked] = 1927
    basis.loc[hughes & ranked] = (
        "loader estimate tied to Fine Clothes to the Jew (1927); not source-verified"
    )
    basis.loc[hughes & ~ranked] = (
        "source-bounded 1921–1928; per-poem/per-collection year unavailable"
    )

    if (basis == "").any():
        raise AssertionError("publication basis missing")
    return year, basis


def correct_core_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = read_release("data/corpus/dataset_poem_provenance.csv")
    corpus["volume_year"], corpus["year_basis"] = publication_fields(corpus)
    corpus["retained"] = ~corpus["gullah_flag"].astype(bool)
    corpus["exclusion_reason"] = np.where(
        corpus["gullah_flag"].astype(bool),
        "Gullah, excluded as a distinct variety",
        "",
    )
    corpus["rq4_in_band"] = corpus["n_words"].between(80, 200, inclusive="both")
    corpus = insert_after(corpus, "source_volume", ["volume_year", "year_basis"])
    corpus = insert_after(corpus, "n_words", ["rq4_in_band"])
    assert len(corpus) == 738
    assert corpus["retained"].sum() == 733
    assert corpus.loc[corpus.retained, "variety_label"].value_counts().to_dict() == {
        "SAE": 521,
        "AAVE": 212,
    }
    assert corpus.loc[
        corpus.rq4_in_band & corpus.retained, "variety_label"
    ].value_counts().to_dict() == {
        "SAE": 219,
        "AAVE": 102,
    }
    assert corpus.loc[
        corpus["labeling_basis"].str.startswith("Hughes density-ranked"), "volume_year"
    ].eq(1927).all()
    assert len(
        corpus.loc[
            corpus["labeling_basis"].str.startswith("Hughes density-ranked")
        ]
    ) == 24
    write_csv(corpus, "data/corpus/dataset_poem_provenance.csv")

    hughes_audit = corpus.loc[
        corpus["source_volume"].eq("Poetry (Standard Ebooks)"),
        [
            "poem_id",
            "variety_label",
            "labeling_basis",
            "volume_year",
            "year_basis",
        ],
    ].copy()
    assert len(hughes_audit) == 229
    write_csv(hughes_audit, "data/analysis/hughes_year_assignment_audit.csv")

    source_text = (WORK / "data/raw/pd_poetry/hughes_se.html").read_text(
        encoding="utf-8"
    )
    source_audit = pd.DataFrame(
        [
            {
                "check": "bodymatter poem/song elements",
                "value": 229,
                "evidence_location": "data/raw/pd_poetry/hughes_se.html:1249-9270",
            },
            {
                "check": "collection container elements",
                "value": 0,
                "evidence_location": "full source scan for collection containers",
            },
            {
                "check": "exact collection title Fine Clothes to the Jew",
                "value": source_text.lower().count("fine clothes to the jew"),
                "evidence_location": "full source exact-string scan",
            },
            {
                "check": "aggregate publication range",
                "value": "1921-1928",
                "evidence_location": "data/raw/pd_poetry/hughes_se.html:9276-9278",
            },
            {
                "check": "ranked-list rows carrying 1927",
                "value": 24,
                "evidence_location": "data/analysis/hughes_year_assignment_audit.csv:2-25",
            },
            {
                "check": "1927 set equals ranked-list set",
                "value": bool(
                    (
                        hughes_audit["volume_year"].eq(1927).fillna(False).astype(bool)
                        == hughes_audit["labeling_basis"].str.startswith(
                            "Hughes density-ranked"
                        ).astype(bool)
                    ).all()
                ),
                "evidence_location": "computed exact set equality over 229 rows",
            },
        ]
    )
    write_csv(source_audit, "data/analysis/hughes_source_structure_audit.csv")

    year_lookup = corpus.set_index("poem_id")[["volume_year", "year_basis"]]
    for relative, key in [
        ("data/folds/rq1_full_poetry_sae_folds.csv", "poem_id"),
        ("data/scores/rq1_full_poetry_natural_scores.csv", "poem_id"),
    ]:
        frame = read_release(relative)
        frame = frame.drop(
            columns=["pub_year", "publication_era", "volume_year", "year_basis"],
            errors="ignore",
        )
        frame = frame.join(year_lookup, on=key, validate="many_to_one")
        frame = insert_after(frame, "poem_title", ["volume_year", "year_basis"])
        assert frame["year_basis"].notna().all()
        write_csv(frame, relative)

    provenance = read_release("data/generation/rewrite_run_provenance.csv")
    if "model" in provenance.columns:
        provenance = provenance.rename(columns={"model": "model_requested"})
    provenance["generator"] = normalize_generator(provenance["model_requested"])
    provenance["model_returned"] = provenance["generator"].map(lambda x: MODEL_IDS[x][1])
    provenance["source_volume_year"] = provenance["source_poem_id"].map(
        corpus.set_index("poem_id")["volume_year"]
    )
    # The 14 Hughes poems moved by reclassification retain the source-bounded
    # range only; prose is deliberately not inserted into this numeric field.
    assert provenance["source_volume_year"].isna().sum() == 14
    provenance = insert_after(
        provenance,
        "source_poem_id",
        ["source_volume_year", "generator", "model_requested", "model_returned"],
    )
    write_csv(provenance, "data/generation/rewrite_run_provenance.csv")

    metadata = read_release("data/generation/raw_response_metadata.csv").rename(
        columns={
            "requested_model": "model_requested",
            "returned_model": "model_returned",
        }
    )
    metadata["generator"] = normalize_generator(metadata["generator"])
    metadata = insert_after(
        metadata, "generator", ["model_requested", "model_returned"]
    )
    write_csv(metadata, "data/generation/raw_response_metadata.csv")

    manifest = read_release("data/transforms/sae_mgt_generation_manifest.csv").rename(
        columns={"model_full_id": "model_requested"}
    )
    manifest = add_model_ids(manifest)
    manifest["source_volume_year"] = manifest["source_poem_id"].map(
        corpus.set_index("poem_id")["volume_year"]
    )
    assert manifest["source_volume_year"].isna().sum() == 14
    manifest = insert_after(
        manifest,
        "source_poem_id",
        ["source_volume_year"],
    )
    manifest = insert_after(
        manifest, "generator", ["model_requested", "model_returned"]
    )
    write_csv(manifest, "data/transforms/sae_mgt_generation_manifest.csv")

    log_matches = read_release("data/transforms/sae_mgt_generation_api_log_matches.csv").rename(
        columns={"model_full_id": "model_requested"}
    )
    ids = manifest.set_index("generation_id")
    log_matches["generator"] = log_matches["generation_id"].map(ids["generator"])
    log_matches["model_returned"] = log_matches["generation_id"].map(ids["model_returned"])
    log_matches = insert_after(
        log_matches, "generation_id", ["generator", "model_requested", "model_returned"]
    )
    write_csv(log_matches, "data/transforms/sae_mgt_generation_api_log_matches.csv")

    for relative in [
        "data/results/rq3_v03_gpt2_detector_regression.csv",
        "data/results/rq3_v03_opt_detector_regression.csv",
        "data/results/rq3_v03_pretrained_detector_regression.csv",
        "data/scores/rq3_v03_detector_scores_long.csv",
    ]:
        frame = add_model_ids(read_release(relative))
        frame = insert_after(frame, "generator", ["model_requested", "model_returned"])
        write_csv(frame, relative)

    for relative in [
        "data/results/final_rq2_poetry_only_paired_changes.csv",
        "data/results/final_rq2_poetry_only_seed_metrics.csv",
        "data/results/final_rq2_poetry_only_summary.csv",
    ]:
        frame = read_release(relative).rename(columns={"model": "classifier"})
        write_csv(frame, relative)

    return corpus, provenance


def release_text_artifacts(
    corpus: pd.DataFrame, provenance: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retained_ids = set(provenance.loc[provenance.retained, "generation_id"])
    rewrites = read_work("data/interim/aave_poetry_mgt_modeb_run1.csv")
    rewrites = rewrites[rewrites["generation_id"].isin(retained_ids)].copy()
    rewrites = rewrites.rename(
        columns={
            "model_short_name": "generator",
            "model_full_id": "model_requested",
        }
    )
    rewrites["generator"] = normalize_generator(rewrites["generator"])
    rewrites["model_returned"] = rewrites["generator"].map(lambda x: MODEL_IDS[x][1])
    rewrites["source_volume_year"] = rewrites["source_poem_id"].map(
        corpus.set_index("poem_id")["volume_year"]
    )
    rewrites = insert_after(
        rewrites,
        "source_poem_id",
        ["source_volume_year", "generator", "model_requested", "model_returned"],
    )
    assert len(rewrites) == 209 and rewrites["source_volume_year"].isna().sum() == 14
    write_csv(rewrites, "data/text/sae_mgt_rewrites.csv")

    human = read_work("data/analysis/rq1_proxy_sae521_multivalue_canonical.csv")
    assert len(human) == 521
    write_csv(human, "data/transforms/rq1_proxy_sae521_multivalue_canonical.csv")

    machine_old = read_work("data/interim/arm_d_poetry_aave_mgt_run1.csv").rename(
        columns={
            "source_model_short_name": "generator",
            "source_model_full_id": "model_requested",
        }
    )
    machine_old = add_model_ids(machine_old)
    machine_old = insert_after(
        machine_old, "generator", ["model_requested", "model_returned"]
    )
    assert len(machine_old) == 209
    write_csv(machine_old, "data/transforms/rq3_sae_mgt_multivalue_v013.csv")

    machine_new = read_work(
        "outputs/pre_rewrite_methodology_resolution_2026-10-01/"
        "rq3_transformed_sae_mgt_multivalue_v03.csv"
    ).rename(columns={"model_full_id": "model_requested"})
    machine_new = add_model_ids(machine_new)
    machine_new = insert_after(
        machine_new, "generator", ["model_requested", "model_returned"]
    )
    assert len(machine_new) == 209
    write_csv(machine_new, "data/transforms/rq3_sae_mgt_multivalue_v03.csv")
    return rewrites, human, machine_old, machine_new


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def quality_battery(rewrites: pd.DataFrame) -> pd.DataFrame:
    old = read_work("data/analysis/beemo_quality_per_item.csv")
    pairs = old[old.record_type.eq("pair")].copy()
    text_rows = old[old.record_type.eq("text")].copy()
    rescaled = read_work("data/analysis/beemo_bertscore_rescaled_per_item.csv")
    align = read_work("data/analysis/beemo_alignscore_large_per_item.csv")

    assert len(pairs) == len(rescaled) == len(align) == 939
    key = ["pair", "generation_id"]
    right = rescaled.merge(
        align[key + ["alignscore", "alignscore_large"]],
        on=key,
        how="left",
        validate="one_to_one",
    )
    assert right[["alignscore", "alignscore_large"]].notna().all().all()
    right = right.rename(columns={"pair": "set_name", "generation_id": "item_id"})
    keep = [
        "set_name",
        "item_id",
        "reference_n_words",
        "candidate_n_words",
        "previous_raw_f1",
        "full_precision",
        "full_recall",
        "full_f1",
        "alignscore",
        "alignscore_large",
    ]
    out = pairs.merge(right[keep], on=["set_name", "item_id"], validate="one_to_one")

    # bert_score's English roberta-large baseline, layer 17.
    baseline_p, baseline_r, baseline_f = 0.83150584, 0.83149410, 0.83122575
    out["raw_bertscore_precision"] = out.full_precision * (1 - baseline_p) + baseline_p
    out["raw_bertscore_recall"] = out.full_recall * (1 - baseline_r) + baseline_r
    reconstructed_f = out.full_f1 * (1 - baseline_f) + baseline_f
    assert (reconstructed_f - out.previous_raw_f1).abs().max() < 2e-6
    out["raw_bertscore_f1"] = out.previous_raw_f1
    out = out.rename(
        columns={
            "full_precision": "rescaled_bertscore_precision",
            "full_recall": "rescaled_bertscore_recall",
            "full_f1": "rescaled_bertscore_f1",
            "alignscore": "alignscore_base",
            "alignscore_large": "alignscore_large",
            "edit_percentage_char": "edit_percentage_autojunk_true",
            "normalized_levenshtein_distance_char": "normalized_levenshtein",
        }
    )
    out["edit_percentage_autojunk_false"] = [
        (1 - SequenceMatcher(None, a, b, autojunk=False).ratio()) * 100
        for a, b in zip(out.reference_text, out.candidate_text)
    ]
    out["levenshtein_denominator"] = out[
        ["reference_character_count", "candidate_character_count"]
    ].max(axis=1)

    metrics = text_rows.set_index("candidate_sha256")
    assert metrics.index.is_unique
    for side in ["reference", "candidate"]:
        hashes = out[f"{side}_sha256"]
        out[f"{side}_tokens"] = hashes.map(metrics["td_n_tokens"])
        out[f"{side}_stopwords"] = hashes.map(metrics["td_n_stop_words"])
        out[f"{side}_stopword_proportion"] = hashes.map(
            metrics["td_stopword_proportion"]
        )
        out[f"{side}_flesch_kincaid_grade"] = hashes.map(
            metrics["td_flesch_kincaid_grade"]
        )
    out["generator"] = out["generator"].fillna("")
    has_generator = out["generator"].ne("")
    out.loc[has_generator, "generator"] = normalize_generator(out.loc[has_generator, "generator"])
    out["model_requested"] = out["generator"].map(
        lambda x: MODEL_IDS[x][0] if x else ""
    )
    out["model_returned"] = out["generator"].map(
        lambda x: MODEL_IDS[x][1] if x else ""
    )

    columns = [
        "set_name",
        "item_id",
        "source_id",
        "generator",
        "model_requested",
        "model_returned",
        "reference_sha256",
        "candidate_sha256",
        "reference_n_words",
        "candidate_n_words",
        "reference_tokens",
        "candidate_tokens",
        "reference_stopwords",
        "candidate_stopwords",
        "reference_stopword_proportion",
        "candidate_stopword_proportion",
        "reference_flesch_kincaid_grade",
        "candidate_flesch_kincaid_grade",
        "edit_percentage_autojunk_true",
        "edit_percentage_autojunk_false",
        "levenshtein_distance_char",
        "levenshtein_denominator",
        "normalized_levenshtein",
        "raw_bertscore_precision",
        "raw_bertscore_recall",
        "raw_bertscore_f1",
        "rescaled_bertscore_precision",
        "rescaled_bertscore_recall",
        "rescaled_bertscore_f1",
        "alignscore_base",
        "alignscore_large",
    ]
    out = out[columns]
    assert out.notna().all().all()

    rewrite_hash = rewrites.set_index("generation_id")["completion"].map(sha256_text)
    pair_a = out[out.set_name.eq("Pair_A")].set_index("item_id")
    pair_c = out[out.set_name.eq("Pair_C")].set_index("item_id")
    assert pair_a.index.equals(pair_c.index)
    assert pair_a["candidate_sha256"].equals(rewrite_hash.loc[pair_a.index])
    assert pair_c["reference_sha256"].equals(rewrite_hash.loc[pair_c.index])

    write_csv(out, "data/analysis/quality_battery_per_item.csv")
    return out


def per_item_diagnostics(
    rewrites: pd.DataFrame, human: pd.DataFrame, machine_old: pd.DataFrame
) -> None:
    common = [
        "generation_id",
        "source_poem_id",
        "generator",
        "model_requested",
        "model_returned",
    ]
    grammar = rewrites[common + ["output_grammatical_final"]].rename(
        columns={"output_grammatical_final": "ten_feature_residual_grammar_rate"}
    )
    assert grammar.ten_feature_residual_grammar_rate.gt(0).sum() == 105
    assert grammar.loc[grammar.ten_feature_residual_grammar_rate.gt(0), "generator"].value_counts().to_dict() == {
        "llama": 40,
        "qwen3": 33,
        "deepseek": 32,
    }
    write_csv(grammar, "data/analysis/residual_grammar_per_rewrite.csv")

    eye = rewrites[
        common + ["seed_orthographic_density", "output_orthographic_density"]
    ].rename(
        columns={
            "seed_orthographic_density": "source_eye_dialect_rate",
            "output_orthographic_density": "rewrite_eye_dialect_rate",
        }
    )
    write_csv(eye, "data/analysis/eye_dialect_rate_per_rewrite.csv")

    human_rows = []
    for row in human.itertuples(index=False):
        executed = json.loads(row.executed_rules_json)
        types = {x["type"] for x in executed}
        human_rows.append(
            {
                "corpus": "human_SAE",
                "item_id": row.poem_id,
                "source_poem_id": row.poem_id,
                "generator": "",
                "model_requested": "",
                "model_returned": "",
                "distinct_features": len(types),
                "rule_executions": row.executed_rule_count,
                "byte_identical": row.original_text == row.transformed_text,
            }
        )
    machine_rows = []
    for row in machine_old.itertuples(index=False):
        fired = json.loads(row.features_fired_json)
        machine_rows.append(
            {
                "corpus": "SAE_MGT",
                "item_id": row.arm_d_id,
                "source_poem_id": row.source_poem_id,
                "generator": row.generator,
                "model_requested": row.model_requested,
                "model_returned": row.model_returned,
                "distinct_features": sum(float(v) != 0 for v in fired.values()),
                "rule_executions": row.modification_counter_total,
                "byte_identical": row.sae_mgt_text == row.aave_mgt_text,
            }
        )
    multivalue = pd.DataFrame(human_rows + machine_rows)
    assert len(multivalue) == 730 and not multivalue.byte_identical.any()
    write_csv(multivalue, "data/analysis/multivalue_descriptives_per_poem.csv")

    counts = machine_old[
        [
            "source_generation_id",
            "source_poem_id",
            "generator",
            "model_requested",
            "model_returned",
            "sae_mgt_text",
            "aave_mgt_text",
        ]
    ].copy()
    counts = counts.rename(columns={"source_generation_id": "generation_id"})
    counts["rewrite_word_count"] = counts.sae_mgt_text.str.split().str.len()
    counts["transformed_rewrite_word_count"] = counts.aave_mgt_text.str.split().str.len()
    counts["word_count_change"] = (
        counts.transformed_rewrite_word_count - counts.rewrite_word_count
    )
    counts = counts.drop(columns=["sae_mgt_text", "aave_mgt_text"])
    assert counts.rewrite_word_count.median() == 190
    assert counts.transformed_rewrite_word_count.median() == 194
    write_csv(counts, "data/analysis/rewrite_word_counts_per_item.csv")

    for source, target in [
        (
            "data/analysis/rq1_measured_dialect_profiles.csv",
            "data/analysis/measured_dialect_profiles_per_item.csv",
        ),
        (
            "data/analysis/rq1_measured_dialect_feature_summary.csv",
            "data/analysis/measured_dialect_feature_directions.csv",
        ),
        (
            "data/analysis/rq1_measured_dialect_profile_distances.csv",
            "data/analysis/measured_dialect_profile_distances.csv",
        ),
        (
            "data/analysis/rq1_length_sensitivity_summary.csv",
            "data/analysis/poem_length_sensitivity_summary.csv",
        ),
        (
            "data/analysis/rq1_length_balance_diagnostics.csv",
            "data/analysis/poem_length_balance_diagnostics.csv",
        ),
        (
            "data/analysis/rq1_length_sensitivity_baseline_verification.csv",
            "data/analysis/poem_length_baseline_verification.csv",
        ),
    ]:
        frame = read_work(source)
        if "generator" in frame.columns:
            frame = add_model_ids(frame)
            frame = insert_after(
                frame, "generator", ["model_requested", "model_returned"]
            )
        write_csv(frame, target)


def main() -> None:
    corpus, provenance = correct_core_tables()
    rewrites, human, machine_old, _machine_new = release_text_artifacts(
        corpus, provenance
    )
    quality_battery(rewrites)
    per_item_diagnostics(rewrites, human, machine_old)
    print("round-5 data build completed")


if __name__ == "__main__":
    main()
