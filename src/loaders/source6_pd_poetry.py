"""
Combining loader for the public-domain African American dialect poetry cell
(generation seeds for the AAVE poetry cell).

Runs the four source modules (source6_pd_poetry_{dunbar,johnson1922,henderson,
hughes}), applies a uniform contraction-rejoin normalization, runs the
marker-density heuristic as a validation CHECK (never a selector -- selection in
every source module is an explicit title list), and writes ONE output file:
    data/processed/aave_poetry.csv

Usage:
    python -m src.loaders.source6_pd_poetry
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.loaders import (
    source6_pd_poetry_dunbar as dunbar,
    source6_pd_poetry_henderson as henderson,
    source6_pd_poetry_hughes as hughes,
    source6_pd_poetry_johnson1922 as johnson1922,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("pd_poetry")

OUT_PATH = Path("data/processed/aave_poetry.csv")
SOURCE_DATASET = "pd_poetry"

COLUMNS = [
    "id", "text", "label", "genre", "variety", "source_dataset", "source_subset",
    "n_words", "needs_review", "needs_review_reason",
    "poet", "poem_title", "volume_year", "year_basis", "orthography", "normalization_applied",
]

# Rejoin contractions this edition/these editions print with a spurious space:
# "would n't" -> "wouldn't", "I 's" -> "I's". Applied uniformly to all four
# sources; only Dunbar and (once) Johnson1922 actually contain the pattern, but
# the check runs everywhere so a silent hit elsewhere is not missed.
CONTRACTION_RE = re.compile(r"(?<=\w) (n't|'s|'d|'ll|'ve|'re|'m)\b")
CONTRACTION_RULE_DESC = (
    "re.sub(r\"(?<=\\w) (n't|'s|'d|'ll|'ve|'re|'m)\\b\", r\"\\1\", text) -- "
    "joins a preceding word character + space + contraction suffix into a single "
    "token (e.g. 'would n't' -> 'wouldn't', 'I 's' -> 'I's')"
)

# Marker-density validation heuristic -- a CHECK on the loaded corpus, never a
# selector. Any loaded poem scoring below FLAG_THRESHOLD is printed for manual
# eyeballing; it does not get dropped.
MARKER_RE = re.compile(
    r"\b(de|dat|dis|dem|gwine|wid|yo'|hyeah|ol'|an'|w'en|dey|sho'|chile|mah|dar|"
    r"daih|erbout|nuffin|fu'|fuh|da|bekaze|knowed|natchel)\b",
    re.IGNORECASE,
)
FLAG_THRESHOLD = 0.05


def _normalize_contractions(text: str) -> tuple[str, int]:
    new_text, n = CONTRACTION_RE.subn(r"\1", text)
    return new_text, n


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _marker_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    hits = len(MARKER_RE.findall(text))
    return hits / len(words)


def _build_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        normalized_text, n_sub = _normalize_contractions(r["text"])
        if r["source_subset"] == "dunbar":
            year_basis = "volume title page (1922); latest component copyright year 1913"
        elif r["source_subset"] == "henderson":
            year_basis = "volume title page (1904)"
        elif r["source_subset"] == "johnson1922":
            year_basis = "volume title page (1922)"
        elif r["volume_year"] == 1927:
            year_basis = "loader estimate tied to Fine Clothes to the Jew (1927); not source-verified"
        else:
            year_basis = "source-bounded 1921–1928; per-poem/per-collection year unavailable"
        row = {
            "id": f"pd_poetry_{r['source_subset']}_{_slugify(r['poet'])}_{_slugify(r['poem_title'])}",
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
            "year_basis": year_basis,
            "orthography": r["orthography"],
            "normalization_applied": n_sub > 0,
        }
        rows.append((row, n_sub))
    return rows


def load_all() -> tuple[pd.DataFrame, dict]:
    all_records = []
    per_source_stats = {}

    log.info("=" * 70)
    log.info("LOADING dunbar (PG #18338, 1922 volume; latest component copyright 1913)")
    recs, stats = dunbar.load()
    for r in recs:
        r["source_subset"] = dunbar.SOURCE_SUBSET
    all_records.extend(recs)
    per_source_stats["dunbar"] = stats
    log.info("  -> %d poems loaded", len(recs))

    log.info("=" * 70)
    log.info("LOADING johnson1922 (PG #11986, Book of American Negro Poetry)")
    recs, stats = johnson1922.load()
    all_records.extend(recs)
    per_source_stats["johnson1922"] = stats
    log.info("  -> %d poems loaded (%d AAVE, %d Gullah)", len(recs), stats["n_aave"], stats["n_gullah"])

    log.info("=" * 70)
    log.info("LOADING henderson (PG #66997, Plantation Echoes 1904)")
    recs, stats = henderson.load()
    for r in recs:
        r["source_subset"] = henderson.SOURCE_SUBSET
    all_records.extend(recs)
    per_source_stats["henderson"] = stats
    log.info("  -> %d poems loaded", len(recs))

    log.info("=" * 70)
    log.info("LOADING hughes (Standard Ebooks, Poetry compilation)")
    recs, stats = hughes.load()
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

    # Marker-density validation CHECK (not a selector).
    density_flags = []
    for row, _ in rows_with_subs:
        d = _marker_density(row["text"])
        if d < FLAG_THRESHOLD:
            density_flags.append(
                {
                    "source_subset": row["source_subset"],
                    "poet": row["poet"],
                    "poem_title": row["poem_title"],
                    "variety": row["variety"],
                    "density": d,
                    "already_flagged": row["needs_review"],
                }
            )

    stats = {
        "per_source": per_source_stats,
        "contraction": contraction_stats,
        "density_flags": density_flags,
        "n_total": len(df),
    }
    return df, stats


def _print_report(df: pd.DataFrame, stats: dict) -> None:
    print("\n" + "=" * 70)
    print("PER-SOURCE COVERAGE")
    print("=" * 70)

    d = stats["per_source"]["dunbar"]
    print(f"\ndunbar: {d['n_loaded']} loaded / 98 dialect titles on index "
          f"(expected yield in task brief: 95)")
    print(f"  dropped as confirmed not in this Gutenberg edition ({d['n_dropped_not_in_volume']}): "
          f"{d['dropped_titles']}")
    if d["n_unresolved"]:
        print(f"  UNRESOLVED (investigate): {d['unresolved_titles']}")
    sub = df[df.source_subset == "dunbar"]
    print(f"  tokens -- mean: {sub.n_words.mean():.0f}  min: {sub.n_words.min()}  max: {sub.n_words.max()}")
    print("  NOTE: actual yield is 93, not the 95 assumed in the original scoping "
          "session. Two titles beyond the documented 3 uncollected poems (Charity, "
          "Goin' Back, Sling Along) were found to be genuinely absent from this "
          "Gutenberg text during this build pass -- Lullaby \"Kiver up yo' haid, my "
          "little lady\" and Song \"De bee hit sip some honey f'om de matermony "
          "vine\" -- confirmed by direct search, not just a normalization gap. The "
          "original scoping report undercounted this by 2.")

    j = stats["per_source"]["johnson1922"]
    print(f"\njohnson1922: {j['n_loaded']} loaded ({j['n_aave']} AAVE + {j['n_gullah']} Gullah) "
          f"/ expected ~13 AAVE + 5 Gullah")
    if j["n_missing"]:
        print(f"  MISSING (investigate): {j['missing_titles']}")
    sub = df[df.source_subset == "johnson1922"]
    print(f"  tokens (all) -- mean: {sub.n_words.mean():.0f}  min: {sub.n_words.min()}  max: {sub.n_words.max()}")

    h = stats["per_source"]["henderson"]
    print(f"\nhenderson: {h['n_loaded']} loaded / 47 expected "
          f"({h['n_total_parsed']} total parsed, {h['n_excluded_standard_english']} standard-English excluded)")
    print(f"  excluded: {h['excluded_titles']}")
    sub = df[df.source_subset == "henderson"]
    print(f"  tokens -- mean: {sub.n_words.mean():.0f}  min: {sub.n_words.min()}  max: {sub.n_words.max()}")

    hu = stats["per_source"]["hughes"]
    print(f"\nhughes: {hu['n_loaded']} loaded / ~23 expected "
          f"({hu['n_total_in_compilation']} total poems/songs in the SE compilation)")
    print(f"  explicit include list used ({hu['n_include_list']} titles):")
    for t in hughes.HUGHES_INCLUDE_TITLES:
        print(f"    - {t}")
    print("  NOTE: 24 titles, not exactly 23 -- ranks 23/24 (Moan, Negro Dancer) tied "
          "at 8.1% marker density / 5 hits in 62 words; both included rather than an "
          "arbitrary tie-break (both confirmed heavy-dialect blues poems on manual read).")
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
    print("MARKER-DENSITY VALIDATION CHECK (informational only, not a selector)")
    print("=" * 70)
    print(f"Threshold for flagging: < {FLAG_THRESHOLD:.0%} marker density")
    flags = stats["density_flags"]
    if not flags:
        print("No poems fell below threshold.")
    else:
        for f in flags:
            tag = "(already needs_review from source module)" if f["already_flagged"] else "(NEWLY FLAGGED by this check)"
            print(f"  [{f['source_subset']}] {f['poet']} -- \"{f['poem_title']}\" "
                  f"({f['variety']}): {f['density']:.1%} density {tag}")

    print("\n" + "=" * 70)
    print("COMBINED TOTALS")
    print("=" * 70)
    aave = df[df.variety == "AAVE"]
    gullah = df[df.variety == "Gullah"]
    print(f"Total rows written: {len(df)}")
    print(f"  AAVE: {len(aave)}")
    print(f"  Gullah: {len(gullah)}")
    print(f"\nAAVE-only token distribution:")
    print(f"  mean: {aave.n_words.mean():.0f}  median: {aave.n_words.median():.0f}  "
          f"min: {aave.n_words.min()}  max: {aave.n_words.max()}")
    print(f"\nneeds_review breakdown:")
    print(df[df.needs_review][["source_subset", "poet", "poem_title", "needs_review_reason"]]
          .to_string(index=False))
    print(f"\nTotal needs_review=True: {int(df.needs_review.sum())} / {len(df)}")


if __name__ == "__main__":
    df, stats = load_all()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    _print_report(df, stats)
    print(f"\nWrote {len(df)} rows -> {OUT_PATH}")
