#!/usr/bin/env python3
"""Update/check MANIFEST.csv and scan tracked files for high-confidence secrets."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.csv"

DESCRIPTIONS = {
    ".gitignore": "Ignore local environments, caches, credential files, and checkpoint formats.",
    "README.md": "Release overview, schema definitions, quality settings, and reproduction commands.",
    "DATA_TERMS.md": "Mixed source and generator terms with no blanket data license.",
    "LICENSE": "MIT terms for repository source code only.",
    "requirements.txt": "Pinned direct Python dependencies for the released code.",
    "data/analysis/quality_battery_per_item.csv": "Per-pair edit, Levenshtein, BERTScore, AlignScore, token, stopword, and readability measurements.",
    "data/analysis/residual_grammar_per_rewrite.csv": "Ten-feature residual-grammar rate for each retained rewrite.",
    "data/analysis/eye_dialect_rate_per_rewrite.csv": "Source and rewrite eye-dialect rates for each retained rewrite.",
    "data/analysis/multivalue_descriptives_per_poem.csv": "Per-poem Multi-VALUE distinct-feature counts, execution counts, and byte-identity flags.",
    "data/analysis/measured_dialect_profiles_per_item.csv": "Per-item values for the 35-feature measured dialect profile.",
    "data/analysis/measured_dialect_feature_directions.csv": "Per-feature centroid direction and summary statistics for the measured dialect profile.",
    "data/analysis/measured_dialect_profile_distances.csv": "Per-condition centroid distances, cosine similarities, and bootstrap intervals.",
    "data/analysis/poem_length_sensitivity_summary.csv": "Poem-length sensitivity results matching the appendix summary table.",
    "data/analysis/poem_length_balance_diagnostics.csv": "Poem-length balance diagnostics matching the appendix table.",
    "data/analysis/poem_length_baseline_verification.csv": "Verification of baseline values used by the poem-length sensitivity analysis.",
    "data/analysis/rewrite_word_counts_per_item.csv": "Original and transformed word counts for every retained rewrite.",
    "data/analysis/hughes_year_assignment_audit.csv": "All Hughes rows with label basis, numeric volume year, and year basis.",
    "data/analysis/hughes_source_structure_audit.csv": "Checks of the Standard Ebooks Hughes source structure and publication-range metadata.",
    "data/analysis/annotation_set_provenance_audit.csv": "Audit showing the blinded annotation set was derived from the processed corpus.",
    "data/text/sae_mgt_rewrites.csv": "All 209 retained rewrites with source poem, generator, requested model, and returned model.",
    "data/transforms/rq1_proxy_sae521_multivalue_canonical.csv": "All 521 canonical human-SAE Multi-VALUE input/output pairs with rule provenance.",
    "data/transforms/rq3_sae_mgt_multivalue_v013.csv": "All 209 historical machine-rewrite Multi-VALUE input/output pairs with fired-feature provenance.",
    "data/transforms/rq3_sae_mgt_multivalue_v03.csv": "All 209 version-0.3 machine-rewrite Multi-VALUE input/output pairs with rule counts.",
    "data/transforms/mgt_output_manifest.csv": "Frozen file-level manifest for the machine-rewrite transformation analysis.",
    "reports/round5_release.md": "Tagged camera-ready release audit, investigation results, and verification record.",
    "scripts/build_round5_release.py": "Deterministic assembly of corrected schemas and round-5 per-item artifacts.",
    "scripts/verify_release.py": "Manifest hash verification and high-confidence credential scanning utility.",
    "src/README.md": "Map from released source modules to corpus and analysis artifacts.",
}

SECRET_PATTERNS = {
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "openai_style_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "slack_token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "assigned_secret": re.compile(
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|secret)\b\s*[:=]\s*['\"][^'\"\r\n]{12,}['\"]"
    ),
}


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        return len(frame), len(frame.columns)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return 1, len(value)
        if isinstance(value, list):
            if not value:
                return 0, 0
            columns = len(value[0]) if isinstance(value[0], dict) else 1
            return len(value), columns
    return len(path.read_text(encoding="utf-8").splitlines()), 1


def description(relative: str, prior: dict[str, str]) -> str:
    if relative in DESCRIPTIONS:
        return DESCRIPTIONS[relative]
    if relative in prior:
        return prior[relative]
    if relative.endswith(".py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(source)) or "Released analysis implementation."
        return doc.splitlines()[0].rstrip(".") + "."
    if relative.startswith("data/results/"):
        return "Frozen reported result artifact."
    if relative.startswith("data/scores/"):
        return "Frozen per-item detector score artifact."
    if relative.startswith("data/thresholds/"):
        return "Frozen detector-threshold artifact."
    if relative.startswith("data/generation/"):
        return "Generation provenance and safe response metadata artifact."
    if relative.startswith("data/folds/"):
        return "Frozen evaluation-fold assignment artifact."
    if relative.startswith("data/transforms/"):
        return "Transformation provenance or output artifact."
    if relative.startswith("data/corpus/"):
        return "Poem-level corpus provenance artifact."
    return "Release payload file."


def update_manifest() -> None:
    prior: dict[str, str] = {}
    if MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            prior = {row["path"]: row["description"] for row in csv.DictReader(handle)}
    rows = []
    for relative in tracked_paths():
        if relative == "MANIFEST.csv":
            continue
        path = ROOT / relative
        n_rows, n_columns = shape(path)
        rows.append(
            {
                "path": relative,
                "rows": n_rows,
                "columns": n_columns,
                "sha256": sha256(path),
                "description": description(relative, prior),
            }
        )
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "rows", "columns", "sha256", "description"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest updated: {len(rows)} payload files")


def check_manifest() -> None:
    manifest = pd.read_csv(MANIFEST, dtype={"path": str, "sha256": str})
    expected = sorted(p for p in tracked_paths() if p != "MANIFEST.csv")
    actual = sorted(manifest.path.tolist())
    if expected != actual:
        raise SystemExit("manifest path set differs from tracked payload set")
    mismatches = []
    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        n_rows, n_columns = shape(path)
        if (
            sha256(path) != row.sha256
            or n_rows != row.rows
            or n_columns != row.columns
        ):
            mismatches.append(row.path)
    if mismatches:
        raise SystemExit(f"manifest mismatches: {mismatches}")
    print(f"manifest verified: {len(manifest)} of {len(manifest)} payload files")


def credential_scan() -> None:
    paths = tracked_paths()
    matches = []
    for relative in paths:
        data = (ROOT / relative).read_bytes()
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                matches.append((relative, name))
    print(f"credential scan: {len(paths)} tracked files checked; {len(matches)} matches")
    for relative, name in matches:
        print(f"{relative}: {name}")
    if matches:
        raise SystemExit(1)


def schema_scan() -> None:
    csv_paths = sorted(ROOT.rglob("*.csv"))
    generator_tables = []
    problems = []
    for path in csv_paths:
        columns = set(pd.read_csv(path, nrows=0).columns)
        relative = str(path.relative_to(ROOT))
        if "generator" in columns:
            generator_tables.append(relative)
            missing = {"model_requested", "model_returned"} - columns
            if missing:
                problems.append(f"{relative}: missing {sorted(missing)}")
        if "model" in columns:
            problems.append(f"{relative}: ambiguous bare model column")
    print(
        f"schema scan: {len(csv_paths)} CSV files; "
        f"{len(generator_tables)} generator tables; {len(problems)} problems"
    )
    for problem in problems:
        print(problem)
    if problems:
        raise SystemExit(1)


def data_check() -> None:
    corpus = pd.read_csv(ROOT / "data/corpus/dataset_poem_provenance.csv")
    retained = corpus.loc[corpus.retained].variety_label.value_counts().to_dict()
    in_band = corpus.loc[corpus.retained & corpus.rq4_in_band].variety_label.value_counts().to_dict()
    assert retained == {"SAE": 521, "AAVE": 212}
    assert in_band == {"SAE": 219, "AAVE": 102}
    assert (~corpus.retained).sum() == 5
    assert corpus.loc[~corpus.retained, "exclusion_reason"].eq(
        "Gullah, excluded as a distinct variety"
    ).all()
    hughes = corpus[corpus.source_volume.eq("Poetry (Standard Ebooks)")]
    ranked = hughes.labeling_basis.str.startswith("Hughes density-ranked")
    assert ranked.sum() == 24
    assert hughes.volume_year.eq(1927).equals(ranked)

    for relative in [
        "data/folds/rq1_full_poetry_sae_folds.csv",
        "data/scores/rq1_full_poetry_natural_scores.csv",
    ]:
        frame = pd.read_csv(ROOT / relative)
        assert "volume_year" in frame and "year_basis" in frame
        assert "pub_year" not in frame and "publication_era" not in frame
        dunbar = frame.source_subset.eq("dunbar")
        assert frame.loc[dunbar, "volume_year"].eq(1922).all()

    quality = pd.read_csv(ROOT / "data/analysis/quality_battery_per_item.csv")
    assert quality.set_name.value_counts().to_dict() == {
        "Pair_B": 521,
        "Pair_A": 209,
        "Pair_C": 209,
    }
    rewrites = pd.read_csv(ROOT / "data/text/sae_mgt_rewrites.csv")
    assert len(rewrites) == 209
    rewrite_hash = rewrites.set_index("generation_id").completion.map(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )
    pair_a = quality[quality.set_name.eq("Pair_A")].set_index("item_id")
    pair_c = quality[quality.set_name.eq("Pair_C")].set_index("item_id")
    assert pair_a.index.equals(pair_c.index)
    assert pair_a.candidate_sha256.equals(rewrite_hash.loc[pair_a.index])
    assert pair_c.reference_sha256.equals(rewrite_hash.loc[pair_c.index])

    grammar = pd.read_csv(ROOT / "data/analysis/residual_grammar_per_rewrite.csv")
    positive = grammar[grammar.ten_feature_residual_grammar_rate.gt(0)]
    assert len(positive) == 105
    assert positive.generator.value_counts().to_dict() == {
        "llama": 40,
        "qwen3": 33,
        "deepseek": 32,
    }
    multivalue = pd.read_csv(ROOT / "data/analysis/multivalue_descriptives_per_poem.csv")
    assert multivalue.corpus.value_counts().to_dict() == {
        "human_SAE": 521,
        "SAE_MGT": 209,
    }
    counts = pd.read_csv(ROOT / "data/analysis/rewrite_word_counts_per_item.csv")
    assert counts.rewrite_word_count.median() == 190
    assert counts.transformed_rewrite_word_count.median() == 194
    print(
        "data check: corpus 212 AAVE/521 SAE retained; band 102/219; "
        "quality 939; rewrites 209; trace links 209+209; grammar 105; "
        "Multi-VALUE 521+209; word-count medians 190/194"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--credential-scan", action="store_true")
    parser.add_argument("--schema-scan", action="store_true")
    parser.add_argument("--data-check", action="store_true")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("select at least one operation")
    if args.update_manifest:
        update_manifest()
    if args.check_manifest:
        check_manifest()
    if args.credential_scan:
        credential_scan()
    if args.schema_scan:
        schema_scan()
    if args.data_check:
        data_check()


if __name__ == "__main__":
    main()
