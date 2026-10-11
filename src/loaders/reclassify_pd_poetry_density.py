"""
Reclassifies the 27 PD poetry rows (22 Dunbar, 5 Hughes) that the negative
marker-density validation check (source6_pd_poetry_sae.py) found scoring
above the 5% dialect threshold despite being excluded from their sources'
original AAVE selection methods -- Dunbar's 22 are absent from the site's
98-title dialect index entirely (confirmed by direct search, not a matching
gap; includes "When Malindy Sings" and "When de Co'n Pone's Hot," Dunbar's
two most famous dialect poems); Hughes's 5 were near-misses just below the
original 24-title density-ranking cutoff. Manually confirmed as genuine
dialect content and moved from sae_poetry.csv to aave_poetry.csv here.

Also backfills a new `selection_source` column on aave_poetry.csv describing
how EVERY row (not just these 27) was originally selected:
  - dunbar:      "site_index" (93 original) / "density_reclassified" (22, here)
  - johnson1922: "manual_list" (explicit title list, JOHNSON_DIALECT_TITLES)
  - henderson:   "manual_list" (explicit exclusion list, STANDARD_ENGLISH_TITLES)
  - hughes:      "density_ranked" (24 original, ranked+natural-break selection)
                 / "density_reclassified" (5, here)
Only "site_index" and "density_reclassified" were specified in the task; the
other two values are this script's own choice, documented above, reflecting
each source's real original selection method faithfully.

Then re-runs the marker-density check on the REMAINING sae_poetry.csv rows
(557 - 27 = 530) at a LOWER 2% threshold -- printed only, nothing moved
automatically.

Reads and overwrites data/processed/aave_poetry.csv and
data/processed/sae_poetry.csv. Does NOT touch unified.csv.

Usage:
    python -m src.loaders.reclassify_pd_poetry_density
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.loaders.source6_pd_poetry import FLAG_THRESHOLD, _marker_density, _slugify

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("reclassify_pd_poetry_density")

AAVE_PATH = Path("data/processed/aave_poetry.csv")
SAE_PATH = Path("data/processed/sae_poetry.csv")

LOWER_THRESHOLD = 0.02

SELECTION_SOURCE_ORIGINAL = {
    "dunbar": "site_index",
    "johnson1922": "manual_list",
    "henderson": "manual_list",
    "hughes": "density_ranked",
}

HUGHES_RECLASSIFY_TITLES = {
    "Fortune Teller Blues", "Feet o’ Jesus", "Po’ Boy Blues",
    "Saturday Night", "Red Silk Stockings",
}


def main():
    aave = pd.read_csv(AAVE_PATH)
    sae = pd.read_csv(SAE_PATH)
    log.info("Loaded aave_poetry.csv (%d rows) and sae_poetry.csv (%d rows)", len(aave), len(sae))

    # ---------------- backfill selection_source on existing aave rows ----------------
    aave["selection_source"] = aave["source_subset"].map(SELECTION_SOURCE_ORIGINAL)
    assert aave["selection_source"].notna().all(), "every existing row must get a selection_source"

    # ---------------- identify the 27 rows to reclassify ----------------
    flagged_mask = sae["needs_review_reason"].fillna("").str.contains("NEGATIVE marker-density check")
    flagged = sae[flagged_mask].copy()
    n_dunbar_flagged = int((flagged["source_subset"] == "dunbar").sum())
    n_hughes_flagged = int((flagged["source_subset"] == "hughes").sum())
    log.info("Found %d density-flagged rows in sae_poetry.csv (%d dunbar, %d hughes)",
              len(flagged), n_dunbar_flagged, n_hughes_flagged)
    assert n_dunbar_flagged == 22, f"expected 22 flagged dunbar rows, got {n_dunbar_flagged}"
    assert n_hughes_flagged == 5, f"expected 5 flagged hughes rows, got {n_hughes_flagged}"
    hughes_flagged_titles = set(flagged.loc[flagged["source_subset"] == "hughes", "poem_title"])
    assert hughes_flagged_titles == HUGHES_RECLASSIFY_TITLES, (
        f"hughes flagged titles don't match expected set: {hughes_flagged_titles}"
    )
    assert set(flagged["source_subset"]) == {"dunbar", "hughes"}, "unexpected source in flagged set"

    # ---------------- transform + move ----------------
    moved_rows = []
    for _, row in flagged.iterrows():
        orthography = "heavy" if row["source_subset"] == "dunbar" else "light"
        density = _marker_density(row["text"])
        new_id = f"pd_poetry_{row['source_subset']}_{_slugify(row['poet'])}_{_slugify(row['poem_title'])}"
        reason = (
            f"Reclassified from sae_poetry.csv to aave_poetry.csv: marker-density scored "
            f"{density:.1%}, above the {FLAG_THRESHOLD:.0%} dialect threshold, despite "
            f"exclusion from the original AAVE selection method for this source "
            f"({'absent from the site dialect index' if row['source_subset'] == 'dunbar' else 'below the original density-ranking cutoff'}); "
            f"manually confirmed as genuine dialect content."
        )
        moved_rows.append({
            "id": new_id,
            "text": row["text"],
            "label": "human",
            "genre": "poetry",
            "variety": "AAVE",
            "source_dataset": "pd_poetry",
            "source_subset": row["source_subset"],
            "n_words": row["n_words"],
            "needs_review": True,
            "needs_review_reason": reason,
            "poet": row["poet"],
            "poem_title": row["poem_title"],
            "volume_year": row["volume_year"],
            "year_basis": row["year_basis"],
            "orthography": orthography,
            "normalization_applied": row["normalization_applied"],
            "selection_source": "density_reclassified",
        })

    moved_df = pd.DataFrame(moved_rows)[list(aave.columns)]
    new_aave = pd.concat([aave, moved_df], ignore_index=True)

    new_sae = sae[~flagged_mask].reset_index(drop=True)

    # ---------------- lower-threshold (2%) informational check on the remainder ----------------
    lower_flags = []
    for _, row in new_sae.iterrows():
        d = _marker_density(row["text"])
        if d > LOWER_THRESHOLD:
            text = str(row["text"])
            nonempty_lines = [l.strip() for l in text.split("\n") if l.strip()]
            first_two = nonempty_lines[:2]
            lower_flags.append({
                "source_subset": row["source_subset"],
                "poet": row["poet"],
                "poem_title": row["poem_title"],
                "density": d,
                "first_two_lines": first_two,
            })

    # ---------------- write ----------------
    new_aave.to_csv(AAVE_PATH, index=False)
    new_sae.to_csv(SAE_PATH, index=False)
    log.info("Wrote aave_poetry.csv (%d rows, +%d) and sae_poetry.csv (%d rows, -%d)",
              len(new_aave), len(moved_df), len(new_sae), len(flagged))

    # ---------------- report ----------------
    print("\n" + "=" * 70)
    print("RECLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"Moved {len(moved_df)} rows from sae_poetry.csv to aave_poetry.csv "
          f"({n_dunbar_flagged} dunbar, {n_hughes_flagged} hughes)")

    print("\n" + "=" * 70)
    print(f"LOWER-THRESHOLD (2%) DENSITY CHECK on remaining {len(new_sae)} sae_poetry.csv rows -- INFORMATIONAL ONLY")
    print("=" * 70)
    print(f"{len(lower_flags)} poems score above {LOWER_THRESHOLD:.0%} density (not moved -- for review)\n")
    for f in sorted(lower_flags, key=lambda x: -x["density"]):
        print(f"[{f['source_subset']}] {f['poet']} -- \"{f['poem_title']}\": {f['density']:.1%} density")
        for line in f["first_two_lines"]:
            print(f"    {line}")
        print()

    print("=" * 70)
    print("FINAL COUNTS")
    print("=" * 70)
    print(f"\naave_poetry.csv: {len(new_aave)} rows")
    print(new_aave.groupby(["source_subset", "variety"]).size().to_string())
    print(f"\n  token distribution by variety:")
    print(new_aave.groupby("variety")["n_words"].agg(["count", "mean", "median", "min", "max"]).round(1).to_string())
    print(f"\n  selection_source breakdown:")
    print(new_aave.groupby(["source_subset", "selection_source"]).size().to_string())

    print(f"\nsae_poetry.csv: {len(new_sae)} rows")
    print(new_sae.groupby("source_subset").size().to_string())
    print(f"\n  token distribution:")
    print(new_sae.groupby("source_subset")["n_words"].agg(["count", "mean", "median", "min", "max"]).round(1).to_string())
    print(f"\n  overall: mean={new_sae['n_words'].mean():.1f} median={new_sae['n_words'].median():.1f} "
          f"min={new_sae['n_words'].min()} max={new_sae['n_words'].max()}")

    return new_aave, new_sae, lower_flags


if __name__ == "__main__":
    main()
