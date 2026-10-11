"""
SAE-side companion to source6_pd_poetry.py: loads the standard-English poems
that sit alongside the AAVE/Gullah dialect poems in the same four public-
domain sources, as generation seeds for the poetry cell's SAE side (which
readability_v2/v3 established is the binding constraint on N, not AAVE).

Runs the four source modules' load_sae() functions (dunbar, johnson1922,
henderson, hughes), applies the SAME contraction-rejoin normalization as the
AAVE loader, runs the marker-density heuristic as a NEGATIVE validation CHECK
(flagging any "SAE" poem that scores ABOVE the dialect threshold as a likely
misclassification -- printed for review, not silently dropped), and writes
ONE output file:
    data/processed/sae_poetry.csv

Does not read or write data/processed/aave_poetry.csv or unified.csv.

Usage:
    python -m src.loaders.source6_pd_poetry_sae
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.loaders import (
    source6_pd_poetry_dunbar as dunbar,
    source6_pd_poetry_henderson as henderson,
    source6_pd_poetry_hughes as hughes,
    source6_pd_poetry_johnson1922 as johnson1922,
)
from src.loaders.source6_pd_poetry import (
    COLUMNS,
    CONTRACTION_RULE_DESC,
    FLAG_THRESHOLD,
    _marker_density,
    _normalize_contractions,
    _slugify,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("pd_poetry_sae")

OUT_PATH = Path("data/processed/sae_poetry.csv")
SOURCE_DATASET = "pd_poetry"


def _build_rows(records: list[dict]) -> list[tuple[dict, int]]:
    rows = []
    for r in records:
        normalized_text, n_sub = _normalize_contractions(r["text"])
        row = {
            "id": f"sae_poetry_{r['source_subset']}_{_slugify(r['poet'])}_{_slugify(r['poem_title'])}",
            "text": normalized_text,
            "label": "human",
            "genre": "poetry",
            "variety": r["variety"],
            "source_dataset": SOURCE_DATASET,
            "source_subset": r["source_subset"],
            "n_words": len(normalized_text.split()),
            "needs_review": r["needs_review"],
            "needs_review_reason": r["needs_review_reason"],
            "poet": r["poet"],
            "poem_title": r["poem_title"],
            "volume_year": r["volume_year"],
            "year_basis": (
                "volume title page (1922); latest component copyright year 1913"
                if r["source_subset"] == "dunbar"
                else "volume title page (1904)"
                if r["source_subset"] == "henderson"
                else "volume title page (1922)"
                if r["source_subset"] == "johnson1922"
                else "source-bounded 1921–1928; per-poem/per-collection year unavailable"
            ),
            "orthography": r["orthography"],
            "normalization_applied": n_sub > 0,
        }
        rows.append((row, n_sub))
    return rows


def load_all() -> tuple[pd.DataFrame, dict]:
    all_records = []
    per_source_stats = {}

    log.info("=" * 70)
    log.info("LOADING dunbar SAE complement (PG #18338)")
    recs, stats = dunbar.load_sae()
    for r in recs:
        r["source_subset"] = dunbar.SOURCE_SUBSET
    all_records.extend(recs)
    per_source_stats["dunbar"] = stats
    log.info("  -> %d poems loaded", len(recs))

    log.info("=" * 70)
    log.info("LOADING johnson1922 SAE complement (PG #11986)")
    recs, stats = johnson1922.load_sae()
    for r in recs:
        r["source_subset"] = johnson1922.SOURCE_SUBSET
    all_records.extend(recs)
    per_source_stats["johnson1922"] = stats
    log.info("  -> %d poems loaded", len(recs))

    log.info("=" * 70)
    log.info("LOADING henderson SAE complement (PG #66997)")
    recs, stats = henderson.load_sae()
    for r in recs:
        r["source_subset"] = henderson.SOURCE_SUBSET
    all_records.extend(recs)
    per_source_stats["henderson"] = stats
    log.info("  -> %d poems loaded", len(recs))

    log.info("=" * 70)
    log.info("LOADING hughes SAE complement (Standard Ebooks)")
    recs, stats = hughes.load_sae()
    for r in recs:
        r["source_subset"] = hughes.SOURCE_SUBSET
    all_records.extend(recs)
    per_source_stats["hughes"] = stats
    log.info("  -> %d poems loaded", len(recs))

    rows_with_subs = _build_rows(all_records)
    df = pd.DataFrame([row for row, _ in rows_with_subs])[COLUMNS]

    contraction_stats = {
        "rule": CONTRACTION_RULE_DESC,
        "total_substitutions": sum(n for _, n in rows_with_subs),
        "per_source": {},
    }
    for source_subset in ["dunbar", "johnson1922", "henderson", "hughes"]:
        n = sum(n for row, n in rows_with_subs if row["source_subset"] == source_subset)
        contraction_stats["per_source"][source_subset] = n

    # NEGATIVE validation: an SAE poem scoring ABOVE the dialect threshold is
    # a likely misclassification (opposite direction from the AAVE loader's
    # check, same threshold/heuristic/word list for direct comparability).
    density_flags = []
    for row, _ in rows_with_subs:
        d = _marker_density(row["text"])
        if d > FLAG_THRESHOLD:
            density_flags.append(
                {
                    "source_subset": row["source_subset"],
                    "poet": row["poet"],
                    "poem_title": row["poem_title"],
                    "density": d,
                }
            )

    stats = {
        "per_source": per_source_stats,
        "contraction": contraction_stats,
        "density_flags": density_flags,
        "n_total": len(df),
    }
    return df, stats


def _apply_density_flags(df: pd.DataFrame, density_flags: list[dict]) -> pd.DataFrame:
    """Sets needs_review=True (appending to any existing reason) for every row
    the negative marker-density check flagged, so the flag is durable in the
    written file, not just a stdout note."""
    df = df.copy()
    flagged_keys = {(f["source_subset"], f["poet"], f["poem_title"]) for f in density_flags}
    density_by_key = {(f["source_subset"], f["poet"], f["poem_title"]): f["density"] for f in density_flags}
    for idx, row in df.iterrows():
        key = (row["source_subset"], row["poet"], row["poem_title"])
        if key in flagged_keys:
            d = density_by_key[key]
            reason = (
                f"NEGATIVE marker-density check: scored {d:.1%}, above the "
                f"{FLAG_THRESHOLD:.0%} threshold used to flag AAVE/dialect "
                f"content -- possible misclassification as SAE, needs a manual read"
            )
            if row["needs_review"] and row["needs_review_reason"]:
                df.at[idx, "needs_review_reason"] = row["needs_review_reason"] + " | " + reason
            else:
                df.at[idx, "needs_review_reason"] = reason
            df.at[idx, "needs_review"] = True
    return df


def _print_report(df: pd.DataFrame, stats: dict) -> None:
    print("\n" + "=" * 70)
    print("PER-SOURCE COVERAGE (SAE)")
    print("=" * 70)

    d = stats["per_source"]["dunbar"]
    print(f"\ndunbar: {d['n_loaded_sae']} loaded (task brief estimate: ~330)")
    print(f"  total poems parsed in volume: {d['n_total_poems_in_volume']} (task brief said 430 -- "
          f"this build's parse finds {d['n_total_poems_in_volume']}; the gap is 11 subsection-header "
          f"artifacts with no body text immediately following -- e.g. 'MISCELLANEOUS', 'HUMOUR AND "
          f"DIALECT', 'SONNET' -- correctly excluded by the empty-body filter, not real poems)")
    print(f"  dialect-selected (excluded from this load): {d['n_dialect_selected']}")
    sub = df[df.source_subset == "dunbar"]
    print(f"  tokens -- mean: {sub.n_words.mean():.0f}  min: {sub.n_words.min()}  max: {sub.n_words.max()}")

    j = stats["per_source"]["johnson1922"]
    print(f"\njohnson1922: {j['n_loaded_sae']} loaded (allowlist-author standard-English poems)")
    print(f"  per-author parsed counts: {j['n_authors_parsed']}")
    sub = df[df.source_subset == "johnson1922"]
    print(f"  tokens -- mean: {sub.n_words.mean():.0f}  min: {sub.n_words.min()}  max: {sub.n_words.max()}")

    h = stats["per_source"]["henderson"]
    print(f"\nhenderson: {h['n_loaded_sae']} loaded / {h['n_expected']} expected "
          f"(the 4 standard-English poems excluded from the AAVE loader)")
    sub = df[df.source_subset == "henderson"]
    print(f"  tokens -- mean: {sub.n_words.mean():.0f}  min: {sub.n_words.min()}  max: {sub.n_words.max()}")

    hu = stats["per_source"]["hughes"]
    print(f"\nhughes: {hu['n_loaded_sae']} loaded / ~205 expected "
          f"({hu['n_total_in_compilation']} total in compilation, {hu['n_excluded_from_aave_list']} "
          f"on the AAVE include-list and excluded here)")
    sub = df[df.source_subset == "hughes"]
    print(f"  tokens -- mean: {sub.n_words.mean():.0f}  min: {sub.n_words.min()}  max: {sub.n_words.max()}")

    print("\n" + "=" * 70)
    print("CONTRACTION NORMALIZATION")
    print("=" * 70)
    c = stats["contraction"]
    print(f"Rule: {c['rule']}")
    print(f"Total substitutions across corpus: {c['total_substitutions']}")
    for src, n in c["per_source"].items():
        fired = "fires" if n > 0 else "does not fire"
        print(f"  {src}: {n} substitutions ({fired})")

    print("\n" + "=" * 70)
    print(f"MARKER-DENSITY NEGATIVE VALIDATION (> {FLAG_THRESHOLD:.0%} density = possible misclassification)")
    print("=" * 70)
    flags = stats["density_flags"]
    if not flags:
        print("No SAE poems scored above the dialect threshold.")
    else:
        for f in flags:
            print(f"  [{f['source_subset']}] {f['poet']} -- \"{f['poem_title']}\": {f['density']:.1%} density -- FLAGGED for review")

    print("\n" + "=" * 70)
    print("COMBINED SAE TOTALS")
    print("=" * 70)
    print(f"Total rows written: {len(df)}")
    print(f"\nToken distribution (all SAE):")
    print(f"  mean: {df.n_words.mean():.0f}  median: {df.n_words.median():.0f}  "
          f"min: {df.n_words.min()}  max: {df.n_words.max()}")
    print(f"\nBy source:")
    print(df.groupby("source_subset")["n_words"].agg(["count", "mean", "min", "max"]).round(1).to_string())
    print(f"\nTotal needs_review=True: {int(df.needs_review.sum())} / {len(df)}")
    if df.needs_review.sum():
        print(df[df.needs_review][["source_subset", "poet", "poem_title", "needs_review_reason"]].to_string(index=False))


if __name__ == "__main__":
    df, stats = load_all()
    df = _apply_density_flags(df, stats["density_flags"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    _print_report(df, stats)
    print(f"\nWrote {len(df)} rows -> {OUT_PATH}")
