"""Generate the frozen 521-poem Multi-VALUE paired synthetic arm (RQ1 Step 3).

This producer deliberately has four separate phases.  ``prepare`` freezes and
hashes the input/order before any transformation; ``generate`` writes the
canonical single-instance arm and the fresh-instance state diagnostic;
``rerun`` must be invoked in a fresh Python process; and ``finalize`` writes
the provenance manifest and report after verifying the rerun.

No detector is imported or invoked here.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import gc
import hashlib
import json
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
import spacy
import spacy.cli


# Multi-VALUE 0.3 performs network downloads at import time.  The exact model
# and NLTK resources are already present and verified, so suppress only those
# import-time download calls before importing the unmodified package.
spacy.cli.download = lambda *args, **kwargs: None
nltk.download = lambda *args, **kwargs: True

from multivalue import BaseDialect as BD  # noqa: E402
from multivalue import Dialects  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ANALYSIS = ROOT / "data" / "analysis"
REPORTS = ROOT / "reports"
ENV_ROOT = ROOT / ".venv-armc-py31020-validation"

SOURCE = ROOT / "data" / "processed" / "sae_poetry.csv"
FOLDS = ANALYSIS / "rq1_full_poetry_sae_folds.csv"
STEP1_SCORES = ANALYSIS / "rq1_full_poetry_natural_scores.csv"
STEP1_CACHES = [
    ANALYSIS / "rq1_full_poetry_gpt2_local_cache.csv",
    ANALYSIS / "rq1_full_poetry_opt_local_cache.csv",
    ANALYSIS / "rq1_full_poetry_pretrained_local_cache.csv",
]
STEP25_ENV = ANALYSIS / "armc_exact_runtime_environment.txt"
STEP25_HASHES = ANALYSIS / "armc_exact_runtime_hashes.txt"

ORDER = ANALYSIS / "rq1_proxy_sae521_batch_order.csv"
PREFLIGHT = ANALYSIS / "rq1_proxy_sae521_multivalue_preflight.json"
CANONICAL = ANALYSIS / "rq1_proxy_sae521_multivalue_canonical.csv"
FRESH = ANALYSIS / "rq1_proxy_sae521_multivalue_fresh_instance_sensitivity.csv"
RERUN = ANALYSIS / "rq1_proxy_sae521_multivalue_canonical_rerun_hashes.csv"
RERUN_CHECK = ANALYSIS / "rq1_proxy_sae521_multivalue_rerun_verification.json"
MANIFEST = ANALYSIS / "rq1_proxy_sae521_multivalue_provenance_manifest.json"
REPORT = REPORTS / "rq1_proxy_sae521_transformation.md"

COMMANDS = {
    "prepare": (
        "NLTK_DATA=/Users/rdharkar/dev/paper/.venv-armc-py31020-validation/nltk_data "
        ".venv-armc-py31020-validation/bin/python "
        "src/perturb/rq1_proxy_sae521_multivalue.py prepare"
    ),
    "generate": (
        "NLTK_DATA=/Users/rdharkar/dev/paper/.venv-armc-py31020-validation/nltk_data "
        ".venv-armc-py31020-validation/bin/python "
        "src/perturb/rq1_proxy_sae521_multivalue.py generate"
    ),
    "rerun": (
        "NLTK_DATA=/Users/rdharkar/dev/paper/.venv-armc-py31020-validation/nltk_data "
        ".venv-armc-py31020-validation/bin/python "
        "src/perturb/rq1_proxy_sae521_multivalue.py rerun"
    ),
    "finalize": (
        "NLTK_DATA=/Users/rdharkar/dev/paper/.venv-armc-py31020-validation/nltk_data "
        ".venv-armc-py31020-validation/bin/python "
        "src/perturb/rq1_proxy_sae521_multivalue.py finalize"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def compute_seed(source_doc_id: str) -> int:
    return int(hashlib.sha256(str(source_doc_id).encode()).hexdigest()[:8], 16) or 1


def stable_folds(ids: pd.Series, n_folds: int = 5) -> pd.Series:
    order = sorted(ids.astype(str).unique(), key=lambda x: sha256_text(x))
    mapping = {doc: i % n_folds for i, doc in enumerate(order)}
    return ids.astype(str).map(mapping).astype(int)


def imported_multivalue_hashes() -> dict[str, str]:
    package = Path(BD.__file__).resolve().parent
    observed: dict[str, str] = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            observed[str(path.relative_to(package))] = sha256_file(path)
    assert len(observed) == 39, f"Expected 39 imported Multi-VALUE files, found {len(observed)}"

    expected: dict[str, str] = {}
    in_section = False
    for line in STEP25_HASHES.read_text(encoding="utf-8").splitlines():
        if line == "IMPORTED MULTIVALUE PACKAGE FILES (each must equal retained payload)":
            in_section = True
            continue
        if in_section and not line.strip():
            break
        if in_section and re.match(r"^[0-9a-f]{64}  ", line):
            digest, rest = line.split("  ", 1)
            path_text = rest.split("  retained_match=", 1)[0]
            expected[Path(path_text).name if "/resources/" not in path_text and "/inflect/" not in path_text else str(Path(path_text).relative_to(Path(path_text).parents[1]))] = digest

    # Compare directly against the absolute paths retained in the Step 2.5 file;
    # this is clearer than relying on the relative-key normalization above.
    expected_absolute: dict[str, str] = {}
    in_section = False
    for line in STEP25_HASHES.read_text(encoding="utf-8").splitlines():
        if line.startswith("IMPORTED MULTIVALUE PACKAGE FILES"):
            in_section = True
            continue
        if in_section and not line.strip():
            break
        match = re.match(r"^([0-9a-f]{64})  (.*?)  retained_match=yes$", line)
        if in_section and match:
            expected_absolute[str(Path(match.group(2)).resolve())] = match.group(1)
    actual_absolute = {
        str(path.resolve()): sha256_file(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert actual_absolute == expected_absolute, "Imported Multi-VALUE payload differs from Step 2.5"
    return observed


def verify_environment() -> dict[str, Any]:
    assert platform.python_version() == "3.10.20"
    assert spacy.__version__ == "3.7.4"
    assert importlib_metadata.version("en-core-web-sm") == "3.7.1"
    assert importlib_metadata.version("value-nlp") == "0.3"
    import en_core_web_sm

    assert en_core_web_sm.load().meta["version"] == "3.7.1"
    assert ENV_ROOT.resolve() in Path(sys.prefix).resolve().parents or Path(sys.prefix).resolve() == ENV_ROOT.resolve()

    value_versions = sorted(
        {dist.version for dist in importlib_metadata.distributions() if dist.metadata.get("Name", "").lower() == "value-nlp"}
    )
    assert value_versions == ["0.3"], f"Unexpected value-nlp distributions: {value_versions}"
    forbidden = [p for p in sys.path if "multi-value" in p.lower() or "0.1.3" in p.lower()]
    assert not forbidden, f"Forbidden editable path is importable: {forbidden}"
    for pth in Path(sys.prefix, "lib", "python3.10", "site-packages").glob("*.pth"):
        content = pth.read_text(encoding="utf-8", errors="replace")
        assert "multi-value" not in content.lower() and "0.1.3" not in content.lower(), f"Forbidden .pth: {pth}"
    assert str(ENV_ROOT.resolve()) in str(Path(BD.__file__).resolve())

    package_hashes = imported_multivalue_hashes()
    return {
        "verified_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "spacy_version": spacy.__version__,
        "en_core_web_sm_distribution_version": importlib_metadata.version("en-core-web-sm"),
        "en_core_web_sm_loaded_version": en_core_web_sm.load().meta["version"],
        "value_nlp_distribution_version": importlib_metadata.version("value-nlp"),
        "value_nlp_versions_found": value_versions,
        "multivalue_base_dialect_path": str(Path(BD.__file__).resolve()),
        "multivalue_dialects_path": str(Path(Dialects.__file__).resolve()),
        "imported_multivalue_file_hashes": package_hashes,
        "environment_identifier": str(ENV_ROOT.resolve()),
        "environment_freeze_path": str(STEP25_ENV.relative_to(ROOT)),
        "environment_freeze_sha256": sha256_file(STEP25_ENV),
        "step25_hash_ledger_sha256": sha256_file(STEP25_HASHES),
        "transformation_script_sha256": sha256_file(Path(__file__).resolve()),
        "editable_0_1_3_importable": False,
    }


def load_verified_population() -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(SOURCE, keep_default_na=False)
    folds = pd.read_csv(FOLDS, keep_default_na=False)
    assert len(source) == 521 and len(folds) == 521
    assert source["id"].nunique() == 521
    assert folds["poem_id"].nunique() == 521 and folds["source_doc_id"].nunique() == 521
    assert not source["text"].duplicated().any()
    assert set(folds["sae_fold"].astype(int)) == {0, 1, 2, 3, 4}

    source = source.copy()
    source["poem_id"] = source["id"].astype(str)
    source["source_doc_id"] = "poem_sae_" + source["poem_id"]
    source["source_text_sha256"] = source["text"].astype(str).map(sha256_text)
    merged = folds.merge(
        source[["poem_id", "source_doc_id", "text", "source_text_sha256"]],
        on=["poem_id", "source_doc_id"], how="inner", validate="one_to_one",
    )
    assert len(merged) == 521
    assert (merged["n_words"].astype(int) == merged["text"].str.split().str.len()).all()
    assert (stable_folds(merged["source_doc_id"]).to_numpy() == merged["sae_fold"].astype(int).to_numpy()).all()

    cache_hashes: dict[str, str] = {}
    for cache_path in STEP1_CACHES:
        cached = pd.read_csv(cache_path, keep_default_na=False)
        cached = cached.loc[cached["variety"].eq("SAE"), ["source_doc_id", "text"]].copy()
        grouped = cached.groupby("source_doc_id", sort=False)["text"].agg(lambda s: list(dict.fromkeys(s.astype(str))))
        assert len(grouped) == 521 and grouped.map(len).eq(1).all()
        cached_map = grouped.map(lambda x: x[0])
        expected_map = merged.set_index("source_doc_id")["text"]
        assert set(cached_map.index) == set(expected_map.index)
        assert all(cached_map.loc[idx] == expected_map.loc[idx] for idx in expected_map.index)
        cache_hashes[str(cache_path.relative_to(ROOT))] = sha256_file(cache_path)

    scores = pd.read_csv(STEP1_SCORES, keep_default_na=False)
    scores = scores.loc[scores["variety"].eq("SAE")]
    assert scores["source_doc_id"].nunique() == 521
    assert set(scores["source_doc_id"]) == set(merged["source_doc_id"])
    score_folds = scores.groupby("source_doc_id")["evaluation_fold"].agg(lambda s: set(s.astype(int)))
    assert score_folds.map(len).eq(1).all()
    score_fold_map = score_folds.map(lambda s: next(iter(s)))
    expected_fold_map = merged.set_index("source_doc_id")["sae_fold"].astype(int)
    assert all(score_fold_map.loc[idx] == expected_fold_map.loc[idx] for idx in expected_fold_map.index)

    details = {
        "rows": 521,
        "unique_poem_ids": 521,
        "unique_source_doc_ids": 521,
        "duplicate_exact_texts": 0,
        "fold_counts": {str(k): int(v) for k, v in merged["sae_fold"].astype(int).value_counts().sort_index().items()},
        "source_dataset_path": str(SOURCE.relative_to(ROOT)),
        "source_dataset_sha256": sha256_file(SOURCE),
        "fold_file_sha256": sha256_file(FOLDS),
        "step1_score_file_sha256": sha256_file(STEP1_SCORES),
        "step1_cache_sha256": cache_hashes,
        "all_source_texts_exactly_equal_across_step1_caches": True,
        "all_step1_folds_exactly_reproduced": True,
    }
    return merged, details


# Stanza is used only for coreference in upstream 0.3.  Coreference was
# disabled in historical Arm C, so prevent constructor downloads/model loads.
BD.stanza.Pipeline = lambda *args, **kwargs: None


class AfricanAmericanVernacularNoCoref(Dialects.AfricanAmericanVernacular):
    def create_coref_cluster(self, string):
        return None


def build_dialect() -> AfricanAmericanVernacularNoCoref:
    dialect = AfricanAmericanVernacularNoCoref()
    assert dialect.coref is None
    assert "it_is_non_referential" not in dialect.morphosyntax_transforms
    return dialect


def simple_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): simple_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [simple_value(v) for v in value]
    return str(value)


def rule_ledger(rules: dict, include_values: bool = True) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    nested = bool(rules) and all(isinstance(v, dict) and not {"value", "type"}.intersection(v) for v in rules.values())
    if nested:
        iterable = []
        for feature, spans in rules.items():
            for span, item in spans.items():
                iterable.append((span, item, feature))
    else:
        iterable = [(span, item, item.get("type", "")) for span, item in rules.items()]
    for span, item, feature in iterable:
        rec = {
            "span": [int(span[0]), int(span[1])],
            "type": str(item.get("type", feature)),
        }
        if include_values:
            rec["value"] = str(item.get("value", ""))
            rec["token_alignment"] = simple_value(item.get("token_alignment", {}))
        ledger.append(rec)
    return sorted(ledger, key=lambda x: (x["span"][0], x["span"][1], x["type"], str(x.get("value", ""))))


def parser_ledger(dialect) -> list[dict[str, Any]]:
    return [
        {
            "i": int(token.i), "idx": int(token.idx), "text": token.text,
            "whitespace": token.whitespace_, "lemma": token.lemma_, "pos": token.pos_,
            "tag": token.tag_, "dep": token.dep_, "head_i": int(token.head.i),
            "morph": str(token.morph), "sent_start": bool(token.is_sent_start),
            "ent_iob": token.ent_iob_, "ent_type": token.ent_type_,
        }
        for token in dialect.doc
    ]


def capture_ledgers(dialect) -> dict[str, Any]:
    candidates = rule_ledger(dialect.rules, include_values=True)
    candidate_structure = rule_ledger(dialect.rules, include_values=False)
    executed = rule_ledger(dialect.executed_rules, include_values=True)
    executed_structure = rule_ledger(dialect.executed_rules, include_values=False)
    parser = parser_ledger(dialect)
    feature_counts = {
        str(k): int(v) for k, v in dialect.modification_counter.items()
        if k != "total" and int(v) != 0
    }
    return {
        "candidate_features_json": canonical_json(feature_counts),
        "candidate_rules_json": canonical_json(candidates),
        "candidate_ledger_sha256": sha256_text(canonical_json(candidates)),
        "candidate_structure_sha256": sha256_text(canonical_json(candidate_structure)),
        "executed_rules_json": canonical_json(executed),
        "executed_ledger_sha256": sha256_text(canonical_json(executed)),
        "executed_structure_sha256": sha256_text(canonical_json(executed_structure)),
        "executed_rule_count": len(executed),
        "parser_token_count": len(parser),
        "parser_ledger_sha256": sha256_text(canonical_json(parser)),
    }


def transform_row(dialect, row: pd.Series) -> dict[str, Any]:
    from src.analysis.grammatical_proxy_final import grammatical_aave_proxy_final
    from src.analysis.metrics import marker_density

    source_id = str(row["source_doc_id"])
    original = str(row["text"])
    seed = compute_seed(source_id)
    before = bool(dialect.aint)
    dialect.set_seed(seed)
    started = time.perf_counter()
    error_type = ""
    error_message = ""
    transformed = ""
    success = False
    try:
        transformed = dialect.transform(original)
        success = True
    except Exception as exc:  # historical broad per-row handling, deliberately preserved
        error_type = type(exc).__name__
        error_message = str(exc)
    elapsed = time.perf_counter() - started
    after = bool(dialect.aint)
    ledgers: dict[str, Any] = {
        "candidate_features_json": "", "candidate_rules_json": "",
        "candidate_ledger_sha256": "", "candidate_structure_sha256": "",
        "executed_rules_json": "", "executed_ledger_sha256": "",
        "executed_structure_sha256": "", "executed_rule_count": "",
        "parser_token_count": "", "parser_ledger_sha256": "",
    }
    if dialect.doc is not None:
        try:
            ledgers = capture_ledgers(dialect)
        except Exception as ledger_exc:
            error_message = (error_message + " | " if error_message else "") + f"ledger_capture={type(ledger_exc).__name__}: {ledger_exc}"
    candidate_count = int(dialect.modification_counter.get("total", 0)) if dialect.modification_counter else 0
    rec = {
        "canonical_batch_index": int(row["canonical_batch_index"]),
        "source_doc_id": source_id,
        "poem_id": str(row["poem_id"]),
        "sae_fold": int(row["sae_fold"]),
        "original_text": original,
        "transformed_text": transformed,
        "original_sha256": sha256_text(original),
        "transformed_sha256": sha256_text(transformed) if success else "",
        "seed": seed,
        "transformation_success": success,
        "error_type": error_type,
        "error_message": error_message,
        "candidate_rule_count": candidate_count,
        "original_word_count": len(original.split()),
        "transformed_word_count": len(transformed.split()) if success else "",
        "original_character_count": len(original),
        "transformed_character_count": len(transformed) if success else "",
        "original_orthographic_density": marker_density(original),
        "transformed_orthographic_density": marker_density(transformed) if success else "",
        "original_morphosyntactic_density": grammatical_aave_proxy_final(original),
        "transformed_morphosyntactic_density": grammatical_aave_proxy_final(transformed) if success else "",
        "aint_before": before,
        "aint_after": after,
        "elapsed_seconds": elapsed,
    }
    rec.update(ledgers)
    return rec


def first_difference(left: str, right: str) -> dict[str, Any]:
    if left == right:
        return {
            "tag": "", "canonical_start": "", "canonical_end": "",
            "fresh_start": "", "fresh_end": "", "canonical_text": "",
            "fresh_text": "", "canonical_context": "", "fresh_context": "",
        }
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    tag, i1, i2, j1, j2 = next(op for op in matcher.get_opcodes() if op[0] != "equal")
    return {
        "tag": tag, "canonical_start": i1, "canonical_end": i2,
        "fresh_start": j1, "fresh_end": j2, "canonical_text": left[i1:i2],
        "fresh_text": right[j1:j2],
        "canonical_context": left[max(0, i1 - 40):min(len(left), i2 + 40)],
        "fresh_context": right[max(0, j1 - 40):min(len(right), j2 + 40)],
    }


CONTRACTION_RE = re.compile(
    r"\b(?:ain['’]t|isn['’]t|aren['’]t|wasn['’]t|weren['’]t|hasn['’]t|haven['’]t|hadn['’]t|don['’]t|doesn['’]t|didn['’]t|can['’]t|couldn['’]t|won['’]t|wouldn['’]t|shouldn['’]t|mightn['’]t|mustn['’]t)\b",
    re.IGNORECASE,
)


def contraction_comparison(canonical: pd.Series, fresh: dict[str, Any]) -> tuple[bool, bool]:
    if not canonical["transformation_success"] or not fresh["transformation_success"]:
        return False, False
    c_text, f_text = str(canonical["transformed_text"]), str(fresh["transformed_text"])
    c_exec = json.loads(str(canonical["executed_rules_json"]) or "[]")
    f_exec = json.loads(str(fresh["executed_rules_json"]) or "[]")
    c_map = {(tuple(x["span"]), x["type"]): x.get("value", "") for x in c_exec}
    f_map = {(tuple(x["span"]), x["type"]): x.get("value", "") for x in f_exec}
    changed = [key for key in set(c_map) | set(f_map) if c_map.get(key) != f_map.get(key)]
    involves = bool(CONTRACTION_RE.search(c_text) or CONTRACTION_RE.search(f_text)) and any(
        "aint" in key[1] or key[1] == "negative_concord" for key in changed
    )
    normalized_equal = CONTRACTION_RE.sub("<NEG-CONTRACTION>", c_text) == CONTRACTION_RE.sub("<NEG-CONTRACTION>", f_text)
    allowed_types = all("aint" in key[1] or key[1] == "negative_concord" for key in changed)
    only = c_text != f_text and bool(changed) and normalized_equal and allowed_types
    return involves, only


def phase_prepare() -> None:
    if ORDER.exists() or PREFLIGHT.exists():
        raise FileExistsError("Preparation artifacts already exist; refusing to replace frozen order/preflight")
    population, input_details = load_verified_population()
    population = population.copy()
    population["ordering_sha256"] = population["source_doc_id"].astype(str).map(sha256_text)
    population = population.sort_values(["ordering_sha256", "source_doc_id"], kind="stable").reset_index(drop=True)
    population.insert(0, "canonical_batch_index", range(len(population)))
    order_frame = population[[
        "canonical_batch_index", "source_doc_id", "ordering_sha256", "poem_id", "sae_fold", "source_text_sha256"
    ]]
    write_csv_new(ORDER, order_frame)
    environment = verify_environment()
    preflight = {
        "created_at_utc": utc_now(),
        "phase": "pre-transformation freeze",
        "input_verification": input_details,
        "batch_order_path": str(ORDER.relative_to(ROOT)),
        "batch_order_sha256": sha256_file(ORDER),
        "canonical_batch_order_definition": "ascending full SHA-256 digest of source_doc_id",
        "environment": environment,
        "exact_command": COMMANDS["prepare"],
    }
    write_json_new(PREFLIGHT, preflight)
    print(f"PREPARED 521 rows; order_sha256={sha256_file(ORDER)}", flush=True)


def ordered_population() -> pd.DataFrame:
    population, _ = load_verified_population()
    order = pd.read_csv(ORDER, keep_default_na=False)
    assert len(order) == 521 and order["source_doc_id"].nunique() == 521
    assert order["canonical_batch_index"].astype(int).tolist() == list(range(521))
    assert (order["ordering_sha256"] == order["source_doc_id"].astype(str).map(sha256_text)).all()
    assert order["ordering_sha256"].tolist() == sorted(order["ordering_sha256"].tolist())
    joined = order.merge(
        population[["source_doc_id", "poem_id", "sae_fold", "text", "source_text_sha256"]],
        on=["source_doc_id", "poem_id", "sae_fold", "source_text_sha256"],
        validate="one_to_one", how="inner",
    )
    assert len(joined) == 521
    return joined.sort_values("canonical_batch_index").reset_index(drop=True)


def phase_generate() -> None:
    for path in [CANONICAL, FRESH]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite first frozen output: {path}")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert preflight["batch_order_sha256"] == sha256_file(ORDER)
    environment = verify_environment()
    assert environment["transformation_script_sha256"] == preflight["environment"]["transformation_script_sha256"]
    rows = ordered_population()

    canonical_records: list[dict[str, Any]] = []
    dialect = build_dialect()
    print("CANONICAL: one dialect instance, 521 poems", flush=True)
    for i, row in rows.iterrows():
        rec = transform_row(dialect, row)
        canonical_records.append(rec)
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(rows):
            print(f"CANONICAL {i + 1}/521 success={sum(bool(x['transformation_success']) for x in canonical_records)} aint={dialect.aint}", flush=True)
    canonical = pd.DataFrame(canonical_records)
    write_csv_new(CANONICAL, canonical)
    print(f"CANONICAL FROZEN sha256={sha256_file(CANONICAL)}", flush=True)

    fresh_records: list[dict[str, Any]] = []
    print("FRESH-INSTANCE DIAGNOSTIC: 521 new dialect instances", flush=True)
    for i, row in rows.iterrows():
        fresh_dialect = build_dialect()
        fresh = transform_row(fresh_dialect, row)
        del fresh_dialect
        canonical_row = canonical.iloc[i]
        exact_equal = bool(canonical_row["transformation_success"] and fresh["transformation_success"] and canonical_row["transformed_text"] == fresh["transformed_text"])
        diff = first_difference(str(canonical_row["transformed_text"]), str(fresh["transformed_text"])) if canonical_row["transformation_success"] and fresh["transformation_success"] else first_difference("", "")
        involves_contraction, contraction_only = contraction_comparison(canonical_row, fresh)
        comp = {
            "canonical_transformed_sha256": canonical_row["transformed_sha256"],
            "exact_string_equal_to_canonical": exact_equal,
            "first_diff_tag": diff["tag"],
            "first_diff_canonical_start": diff["canonical_start"],
            "first_diff_canonical_end": diff["canonical_end"],
            "first_diff_fresh_start": diff["fresh_start"],
            "first_diff_fresh_end": diff["fresh_end"],
            "first_diff_canonical_text": diff["canonical_text"],
            "first_diff_fresh_text": diff["fresh_text"],
            "first_diff_canonical_context": diff["canonical_context"],
            "first_diff_fresh_context": diff["fresh_context"],
            "character_count_difference_fresh_minus_canonical": (
                int(fresh["transformed_character_count"]) - int(canonical_row["transformed_character_count"])
                if fresh["transformation_success"] and canonical_row["transformation_success"] else ""
            ),
            "word_count_difference_fresh_minus_canonical": (
                int(fresh["transformed_word_count"]) - int(canonical_row["transformed_word_count"])
                if fresh["transformation_success"] and canonical_row["transformation_success"] else ""
            ),
            "orthographic_density_difference_fresh_minus_canonical": (
                float(fresh["transformed_orthographic_density"]) - float(canonical_row["transformed_orthographic_density"])
                if fresh["transformation_success"] and canonical_row["transformation_success"] else ""
            ),
            "morphosyntactic_density_difference_fresh_minus_canonical": (
                float(fresh["transformed_morphosyntactic_density"]) - float(canonical_row["transformed_morphosyntactic_density"])
                if fresh["transformation_success"] and canonical_row["transformation_success"] else ""
            ),
            "candidate_rule_count_difference_fresh_minus_canonical": (
                int(fresh["candidate_rule_count"]) - int(canonical_row["candidate_rule_count"])
            ),
            "candidate_rule_ledger_differs": fresh["candidate_ledger_sha256"] != canonical_row["candidate_ledger_sha256"],
            "candidate_rule_structure_differs": fresh["candidate_structure_sha256"] != canonical_row["candidate_structure_sha256"],
            "executed_rule_ledger_differs": fresh["executed_ledger_sha256"] != canonical_row["executed_ledger_sha256"],
            "executed_rule_structure_differs": fresh["executed_structure_sha256"] != canonical_row["executed_structure_sha256"],
            "difference_involves_contraction_or_aint": involves_contraction,
            "only_difference_is_contraction_realization": contraction_only,
        }
        fresh.update(comp)
        fresh_records.append(fresh)
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(rows):
            print(f"FRESH {i + 1}/521 text_differences={sum(not bool(x['exact_string_equal_to_canonical']) for x in fresh_records)}", flush=True)
        if (i + 1) % 25 == 0:
            gc.collect()
    write_csv_new(FRESH, pd.DataFrame(fresh_records))
    print(f"FRESH DIAGNOSTIC FROZEN sha256={sha256_file(FRESH)}", flush=True)


def phase_rerun() -> None:
    if RERUN.exists() or RERUN_CHECK.exists():
        raise FileExistsError("Refusing to overwrite reproducibility rerun artifacts")
    verify_environment()
    rows = ordered_population()
    canonical = pd.read_csv(CANONICAL, keep_default_na=False)
    assert len(canonical) == 521
    dialect = build_dialect()
    rerun_records: list[dict[str, Any]] = []
    print("RERUN: fresh process, one new dialect instance, 521 poems", flush=True)
    for i, row in rows.iterrows():
        rec = transform_row(dialect, row)
        first = canonical.iloc[i]
        rerun_records.append({
            "canonical_batch_index": int(row["canonical_batch_index"]),
            "source_doc_id": row["source_doc_id"],
            "first_run_success": bool(first["transformation_success"]),
            "rerun_success": bool(rec["transformation_success"]),
            "first_run_transformed_sha256": first["transformed_sha256"],
            "rerun_transformed_sha256": rec["transformed_sha256"],
            "output_hash_equal": bool(first["transformation_success"] and rec["transformation_success"] and first["transformed_sha256"] == rec["transformed_sha256"]),
            "first_run_candidate_ledger_sha256": first["candidate_ledger_sha256"],
            "rerun_candidate_ledger_sha256": rec["candidate_ledger_sha256"],
            "candidate_ledger_equal": first["candidate_ledger_sha256"] == rec["candidate_ledger_sha256"],
            "first_run_executed_ledger_sha256": first["executed_ledger_sha256"],
            "rerun_executed_ledger_sha256": rec["executed_ledger_sha256"],
            "executed_ledger_equal": first["executed_ledger_sha256"] == rec["executed_ledger_sha256"],
            "first_run_parser_ledger_sha256": first["parser_ledger_sha256"],
            "rerun_parser_ledger_sha256": rec["parser_ledger_sha256"],
            "parser_ledger_equal": first["parser_ledger_sha256"] == rec["parser_ledger_sha256"],
            "rerun_error_type": rec["error_type"],
            "rerun_error_message": rec["error_message"],
        })
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(rows):
            print(f"RERUN {i + 1}/521", flush=True)
    rerun = pd.DataFrame(rerun_records)
    write_csv_new(RERUN, rerun)
    successful = rerun.loc[rerun["first_run_success"].astype(bool)]
    mismatches = successful.loc[~successful["output_hash_equal"].astype(bool)]
    check = {
        "created_at_utc": utc_now(),
        "successful_first_run_rows": len(successful),
        "identical_output_hashes": int(successful["output_hash_equal"].sum()),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[["canonical_batch_index", "source_doc_id", "first_run_transformed_sha256", "rerun_transformed_sha256"]].to_dict("records"),
        "rerun_file_sha256": sha256_file(RERUN),
        "exact_command": COMMANDS["rerun"],
    }
    write_json_new(RERUN_CHECK, check)
    print(f"RERUN COMPLETE identical={check['identical_output_hashes']}/{check['successful_first_run_rows']}", flush=True)


def stat_summary(series: pd.Series) -> dict[str, float]:
    vals = [float(v) for v in series]
    return {
        "min": min(vals), "median": statistics.median(vals),
        "mean": statistics.mean(vals), "max": max(vals),
    }


def fmt_stats(values: dict[str, float], digits: int = 6) -> str:
    return ", ".join(f"{k}={v:.{digits}f}" for k, v in values.items())


def phase_finalize() -> None:
    for path in [MANIFEST, REPORT]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final artifact: {path}")
    environment = verify_environment()
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    rerun_check = json.loads(RERUN_CHECK.read_text(encoding="utf-8"))
    canonical = pd.read_csv(CANONICAL, keep_default_na=False)
    fresh = pd.read_csv(FRESH, keep_default_na=False)
    rerun = pd.read_csv(RERUN, keep_default_na=False)
    assert len(canonical) == len(fresh) == len(rerun) == 521
    assert environment["transformation_script_sha256"] == preflight["environment"]["transformation_script_sha256"]

    c_success = canonical["transformation_success"].astype(str).str.lower().eq("true")
    f_success = fresh["transformation_success"].astype(str).str.lower().eq("true")
    exact = fresh["exact_string_equal_to_canonical"].astype(str).str.lower().eq("true")
    text_diff = c_success & f_success & ~exact
    only_contraction = fresh["only_difference_is_contraction_realization"].astype(str).str.lower().eq("true")
    candidate_diff = fresh["candidate_rule_ledger_differs"].astype(str).str.lower().eq("true")
    executed_structure_diff = fresh["executed_rule_structure_differs"].astype(str).str.lower().eq("true")
    orth_diff = fresh["orthographic_density_difference_fresh_minus_canonical"].astype(float).abs() > 0
    morph_diff = fresh["morphosyntactic_density_difference_fresh_minus_canonical"].astype(float).abs() > 0

    successful = canonical.loc[c_success].copy()
    successful["word_change"] = successful["transformed_word_count"].astype(float) - successful["original_word_count"].astype(float)
    successful["char_change"] = successful["transformed_character_count"].astype(float) - successful["original_character_count"].astype(float)
    successful["orth_change"] = successful["transformed_orthographic_density"].astype(float) - successful["original_orthographic_density"].astype(float)
    successful["morph_change"] = successful["transformed_morphosyntactic_density"].astype(float) - successful["original_morphosyntactic_density"].astype(float)

    aint_before = canonical["aint_before"].astype(str).str.lower().eq("true")
    aint_after = canonical["aint_after"].astype(str).str.lower().eq("true")
    transitions = canonical.index[(~aint_before) & aint_after].tolist()
    first_transition = transitions[0] if transitions else None
    returns_false = bool(((aint_before) & (~aint_after)).any())

    failures = canonical.loc[~c_success]
    failure_by_fold = {str(k): int(v) for k, v in failures.groupby("sae_fold").size().items()}
    failure_characteristics = "No failures."
    if len(failures):
        failure_characteristics = (
            f"Failure original-word counts: {fmt_stats(stat_summary(failures['original_word_count'].astype(float)), 2)}; "
            f"sources: {dict(Counter(failures['source_doc_id'].astype(str).str.split('_').str[:3].str.join('_')))}."
        )

    abs_orth = fresh.loc[c_success & f_success, "orthographic_density_difference_fresh_minus_canonical"].astype(float).abs()
    abs_morph = fresh.loc[c_success & f_success, "morphosyntactic_density_difference_fresh_minus_canonical"].astype(float).abs()

    manifest = {
        "created_at_utc": utc_now(),
        "description": "new rule-transformed poetry arm generated with the reconstructed historical Arm C Multi-VALUE procedure",
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "source_dataset_sha256": sha256_file(SOURCE),
        "source_text_hash_count": 521,
        "batch_order_file": str(ORDER.relative_to(ROOT)),
        "batch_order_file_sha256": sha256_file(ORDER),
        "canonical_output_file": str(CANONICAL.relative_to(ROOT)),
        "canonical_output_file_sha256": sha256_file(CANONICAL),
        "fresh_instance_sensitivity_file": str(FRESH.relative_to(ROOT)),
        "fresh_instance_sensitivity_file_sha256": sha256_file(FRESH),
        "canonical_rerun_file": str(RERUN.relative_to(ROOT)),
        "canonical_rerun_file_sha256": sha256_file(RERUN),
        "rerun_verification_file": str(RERUN_CHECK.relative_to(ROOT)),
        "rerun_verification_file_sha256": sha256_file(RERUN_CHECK),
        "preflight_file": str(PREFLIGHT.relative_to(ROOT)),
        "preflight_file_sha256": sha256_file(PREFLIGHT),
        "environment_identifier": environment["environment_identifier"],
        "environment_freeze_sha256": environment["environment_freeze_sha256"],
        "transformation_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "transformation_script_sha256": environment["transformation_script_sha256"],
        "imported_multivalue_file_hashes": environment["imported_multivalue_file_hashes"],
        "runtime_versions": {
            "python": environment["python_version"], "spacy": environment["spacy_version"],
            "en_core_web_sm": environment["en_core_web_sm_loaded_version"],
            "value_nlp": environment["value_nlp_distribution_version"],
        },
        "commands": COMMANDS,
        "attempted": 521,
        "successful": int(c_success.sum()),
        "failed": int((~c_success).sum()),
        "fresh_instance_label": "fresh-instance state sensitivity diagnostic",
        "reproducibility": rerun_check,
    }
    write_json_new(MANIFEST, manifest)

    decision = (
        "PROCEED TO DETECTOR SCORING"
        if int(c_success.sum()) > 0 and rerun_check["mismatch_count"] == 0
        else "STOP BEFORE DETECTOR SCORING"
    )
    decision_reason = (
        f"The fixed canonical arm has {int(c_success.sum())} successful rows and the fresh-process rerun reproduced "
        f"{rerun_check['identical_output_hashes']}/{rerun_check['successful_first_run_rows']} successful output hashes."
        if decision.startswith("PROCEED") else
        f"Canonical successful rows={int(c_success.sum())}; reproducibility mismatches={rerun_check['mismatch_count']}."
    )

    fold_fail_table = "\n".join(
        f"| {fold} | {int((canonical['sae_fold'].astype(int) == fold).sum())} | {int(((canonical['sae_fold'].astype(int) == fold) & ~c_success).sum())} |"
        for fold in range(5)
    )
    lines = [
        "# RQ1 paired synthetic-poetry transformation (Step 3)",
        "",
        "No detector was run, GPTZero was not called, the manuscript was not edited, and the 521 source poems and Step 1 folds were not altered.",
        "",
        "## A. Input verification",
        "",
        "The exact Step 1 SAE population was recovered: **521 rows, 521 unique poem IDs, 521 unique source-document IDs, no duplicate text, and all five frozen folds**. Source text was compared byte-for-byte against every retained Step 1 local-detector cache (GPT2-XL, OPT-1.3B, and all pretrained-cache rows), and all 521 texts matched. The Step 1 fold assignments were independently recomputed and matched all 521 rows.",
        "",
        f"- Source dataset SHA-256: `{sha256_file(SOURCE)}`",
        f"- Frozen order file SHA-256: `{sha256_file(ORDER)}`",
        f"- Order: ascending full SHA-256 digest of `source_doc_id`, indexed 0–520.",
        "",
        "## B. Canonical generation",
        "",
        f"One `AfricanAmericanVernacularNoCoref` instance processed the full frozen batch. Attempted: **521**; successful: **{int(c_success.sum())}**; failed: **{int((~c_success).sum())}**.",
        "",
        "| fold | attempted | failed |",
        "|---:|---:|---:|",
        fold_fail_table,
        "",
        failure_characteristics,
        "",
        "The canonical arm is a new rule-transformed poetry arm generated with the reconstructed historical Arm C Multi-VALUE procedure. It is not authentic or naturally occurring AAVE, not a continuation of the historical stochastic sequence, and not a causal intervention covering every dimension of dialect.",
        "",
        "## C. Batch-state trace",
        "",
        f"`self.aint` first changed `False → True` at canonical batch index **{first_transition}**. It ever returned to false: **{'yes' if returns_false else 'no'}**. Poems processed with `self.aint=False` immediately before transformation: **{int((~aint_before).sum())}**; with `self.aint=True`: **{int(aint_before.sum())}**.",
        "",
        "Order is frozen because this upstream field is persistent instance state and is not cleared between documents; changing batch order could change contraction realization.",
        "",
        "## D. Fresh-instance sensitivity",
        "",
        "This arm is labeled **fresh-instance state sensitivity diagnostic** and is not a second primary benchmark or a historical Arm C equivalent.",
        "",
        f"- Text differs: **{int(text_diff.sum())}/521 ({100 * text_diff.sum() / 521:.2f}%)**.",
        f"- Only difference is contraction realization: **{int(only_contraction.sum())}**.",
        f"- Candidate-rule ledger differs: **{int(candidate_diff.sum())}** (candidate-count differs: **{int((fresh['candidate_rule_count_difference_fresh_minus_canonical'].astype(int) != 0).sum())}**).",
        f"- Executed-rule selection/structure differs: **{int(executed_structure_diff.sum())}**.",
        f"- Orthographic density changes: **{int(orth_diff.sum())}**; median absolute change `{abs_orth.median():.8f}`, maximum `{abs_orth.max():.8f}`.",
        f"- Morphosyntactic density changes: **{int(morph_diff.sum())}**; median absolute change `{abs_morph.median():.8f}`, maximum `{abs_morph.max():.8f}`.",
        "",
        "## E. Transformation magnitude",
        "",
        f"Among {len(successful)} successful canonical pairs:",
        "",
        f"- Word-count change (transformed minus SAE): {fmt_stats(stat_summary(successful['word_change']), 3)}.",
        f"- Character-count change: {fmt_stats(stat_summary(successful['char_change']), 3)}.",
        f"- Orthographic-density change: {fmt_stats(stat_summary(successful['orth_change']), 8)}.",
        f"- Morphosyntactic-density change: {fmt_stats(stat_summary(successful['morph_change']), 8)}.",
        f"- Candidate rules recorded per poem: {fmt_stats(stat_summary(successful['candidate_rule_count'].astype(float)), 3)}.",
        f"- Executed rule spans per poem: {fmt_stats(stat_summary(successful['executed_rule_count'].astype(float)), 3)}.",
        "",
        "No comparison to natural AAVE is made in this step.",
        "",
        "## F. Reproducibility",
        "",
        f"A fresh Python process constructed one new dialect instance and reran the same 521 rows in the same order with the same seeds. Successful-row output hashes were identical for **{rerun_check['identical_output_hashes']}/{rerun_check['successful_first_run_rows']}** rows; mismatches: **{rerun_check['mismatch_count']}**.",
        "",
        f"- Canonical output SHA-256: `{sha256_file(CANONICAL)}`",
        f"- Fresh-instance diagnostic SHA-256: `{sha256_file(FRESH)}`",
        f"- Provenance manifest SHA-256 is intentionally not self-recorded; manifest path: `{MANIFEST.relative_to(ROOT)}`.",
        "",
        "## G. Final decision",
        "",
        f"**{decision}**",
        "",
        decision_reason,
    ]
    tmp = REPORT.with_suffix(REPORT.suffix + ".partial")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, REPORT)
    print(f"FINALIZED decision={decision}; report_sha256={sha256_file(REPORT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "generate", "rerun", "finalize"])
    args = parser.parse_args()
    {"prepare": phase_prepare, "generate": phase_generate, "rerun": phase_rerun, "finalize": phase_finalize}[args.phase]()


if __name__ == "__main__":
    main()
