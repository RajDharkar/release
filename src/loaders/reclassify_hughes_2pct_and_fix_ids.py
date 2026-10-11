"""
Two fixes to the poetry files, in order:

1. Moves the 9 Hughes poems flagged by the 2%-threshold follow-up check
   (reclassify_pd_poetry_density.py) from sae_poetry.csv to aave_poetry.csv:
   Misery, Lonesome Place, Listen Here Blues, Death of Do Dirty, The New
   Cabaret Girl, To Midnight Nan at Leroy's, Song for a Banjo Dance, Minnie
   Sings Her Blues, Hard Luck. Same pattern as the prior 22+5 reclassification:
   variety=AAVE, orthography=light, selection_source=density_reclassified,
   density recorded in needs_review_reason, pd_poetry_... id scheme.

2. Fixes the 15 colliding ids in sae_poetry.csv -- Dunbar and Hughes each
   wrote multiple DISTINCT poems under the same generic title ("II", "III",
   "A Song", "Dreams" / "Poem", "Song"), and the id scheme (poet+title only)
   collides on them. Fixed via first-line disambiguation: the same technique
   source6_pd_poetry_dunbar.py already uses for its 4 known site-index-alias
   duplicates (HYMN, LULLABY, SONG, PARTED) -- applied here as a general
   mechanism (slugify each colliding row's own first line and append it)
   rather than a hardcoded alias table, since these are pure within-corpus
   title collisions with no pre-existing external list to hardcode against.
   Only the colliding ids change; every already-unique id is left untouched.

Then asserts id uniqueness across BOTH files combined and prints the result.
If any collision remains after the fix, this stops WITHOUT writing rather
than papering over it.

Makes new backups (suffix _pre_hughes2pct_fix_backup) before writing;
the prior _pre_reclassify_backup files are left alone.

Reads and overwrites data/processed/aave_poetry.csv and
data/processed/sae_poetry.csv. Does NOT touch unified.csv.

Usage:
    python -m src.loaders.reclassify_hughes_2pct_and_fix_ids
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.loaders.source6_pd_poetry import _marker_density, _slugify

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("reclassify_hughes_2pct_and_fix_ids")

AAVE_PATH = Path("data/processed/aave_poetry.csv")
SAE_PATH = Path("data/processed/sae_poetry.csv")
LOWER_THRESHOLD = 0.02

HUGHES_2PCT_TITLES = {
    "Misery", "Lonesome Place", "Listen Here Blues", "Death of Do Dirty",
    "The New Cabaret Girl", "To Midnight Nan at Leroy’s", "Song for a Banjo Dance",
    "Minnie Sings Her Blues", "Hard Luck",
}


def _first_nonblank_line(text: str) -> str:
    return next((l.strip() for l in str(text).split("\n") if l.strip()), "")


def _first_line_slug(text: str, n_words: int = 5) -> str:
    words = _first_nonblank_line(text).split()[:n_words]
    return _slugify(" ".join(words))


def main():
    backup_aave = AAVE_PATH.parent / "aave_poetry_pre_hughes2pct_fix_backup.csv"
    backup_sae = SAE_PATH.parent / "sae_poetry_pre_hughes2pct_fix_backup.csv"
    shutil.copy(AAVE_PATH, backup_aave)
    shutil.copy(SAE_PATH, backup_sae)
    log.info("Backed up current files -> %s, %s (prior _pre_reclassify_backup files untouched)", backup_aave, backup_sae)

    aave = pd.read_csv(AAVE_PATH)
    sae = pd.read_csv(SAE_PATH)
    log.info("Loaded aave_poetry.csv (%d rows), sae_poetry.csv (%d rows)", len(aave), len(sae))

    # ================== Task 1: move the 9 Hughes rows ==================
    move_mask = (sae["source_subset"] == "hughes") & (sae["poem_title"].isin(HUGHES_2PCT_TITLES))
    to_move = sae[move_mask].copy()
    assert len(to_move) == 9, f"expected 9 hughes rows to move, found {len(to_move)}: {sorted(to_move['poem_title'])}"
    missing_titles = HUGHES_2PCT_TITLES - set(to_move["poem_title"])
    assert not missing_titles, f"titles not found in sae_poetry.csv: {missing_titles}"

    moved_rows = []
    for _, row in to_move.iterrows():
        density = _marker_density(row["text"])
        new_id = f"pd_poetry_{row['source_subset']}_{_slugify(row['poet'])}_{_slugify(row['poem_title'])}"
        reason = (
            f"Reclassified from sae_poetry.csv to aave_poetry.csv: marker-density scored "
            f"{density:.1%}, above the {LOWER_THRESHOLD:.0%} lower-threshold follow-up check; "
            f"manually confirmed as genuine dialect content."
        )
        moved_rows.append({
            "id": new_id, "text": row["text"], "label": "human", "genre": "poetry",
            "variety": "AAVE", "source_dataset": "pd_poetry", "source_subset": row["source_subset"],
            "n_words": row["n_words"], "needs_review": True, "needs_review_reason": reason,
            "poet": row["poet"], "poem_title": row["poem_title"],
            "volume_year": row["volume_year"], "year_basis": row["year_basis"],
            "orthography": "light", "normalization_applied": row["normalization_applied"],
            "selection_source": "density_reclassified",
        })
    moved_df = pd.DataFrame(moved_rows)[list(aave.columns)]
    aave2 = pd.concat([aave, moved_df], ignore_index=True)
    sae2 = sae[~move_mask].reset_index(drop=True)
    log.info("Moved %d hughes rows: aave_poetry.csv %d -> %d, sae_poetry.csv %d -> %d",
              len(moved_df), len(aave), len(aave2), len(sae), len(sae2))

    # ================== Task 2: fix colliding ids in sae2 ==================
    dup_mask = sae2["id"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_groups = sae2.loc[dup_mask, "id"].nunique()
    log.info("Found %d rows in %d colliding id groups before the fix", n_dup_rows, n_dup_groups)

    disambig_log = []
    new_ids = sae2["id"].tolist()
    for pos in range(len(sae2)):
        if dup_mask.iloc[pos]:
            row = sae2.iloc[pos]
            slug = _first_line_slug(row["text"])
            new_id = f"{row['id']}_{slug}"
            new_ids[pos] = new_id
            disambig_log.append({
                "old_id": row["id"], "new_id": new_id, "poet": row["poet"],
                "poem_title": row["poem_title"], "first_line": _first_nonblank_line(row["text"]),
            })
    sae2 = sae2.copy()
    sae2["id"] = new_ids

    # Safety net: if two colliding rows happen to share an identical first
    # line too (not expected, but checked rather than assumed), a numeric
    # suffix guarantees uniqueness rather than silently leaving a collision.
    seen = {}
    final_ids = []
    safety_net_used = []
    for id_ in sae2["id"]:
        seen[id_] = seen.get(id_, 0) + 1
        if seen[id_] > 1:
            final_ids.append(f"{id_}_{seen[id_]}")
            safety_net_used.append(id_)
        else:
            final_ids.append(id_)
    sae2["id"] = final_ids

    print("\n" + "=" * 70)
    print("ID DISAMBIGUATION (first-line method, sae_poetry.csv)")
    print("=" * 70)
    for d in disambig_log:
        print(f"  {d['poet']} -- \"{d['poem_title']}\": {d['old_id']}")
        print(f"    -> {d['new_id']}  (first line: {d['first_line'][:60]!r})")
    if safety_net_used:
        print(f"\n  NUMERIC SAFETY NET ENGAGED for {len(safety_net_used)} id(s) -- "
              f"first-line slug alone was not sufficient: {safety_net_used}")
    else:
        print(f"\n  First-line disambiguation alone resolved all {n_dup_groups} colliding groups "
              f"-- no numeric safety net needed.")

    # ================== cross-file uniqueness assertion ==================
    aave_dup = int(aave2["id"].duplicated().sum())
    sae_dup = int(sae2["id"].duplicated().sum())
    overlap = set(aave2["id"]) & set(sae2["id"])
    all_ids = pd.concat([aave2["id"], sae2["id"]])
    combined_dup = int(all_ids.duplicated().sum())

    print("\n" + "=" * 70)
    print("ID UNIQUENESS CHECK (across both files, combined)")
    print("=" * 70)
    print(f"  aave_poetry.csv: {len(aave2)} rows, {aave2['id'].nunique()} unique ids "
          f"-- {'OK' if aave_dup == 0 else f'FAIL ({aave_dup} internal duplicates)'}")
    print(f"  sae_poetry.csv:  {len(sae2)} rows, {sae2['id'].nunique()} unique ids "
          f"-- {'OK' if sae_dup == 0 else f'FAIL ({sae_dup} internal duplicates)'}")
    print(f"  cross-file overlap: {len(overlap)} id(s) shared between the two files "
          f"-- {'OK (none)' if not overlap else f'FAIL: {overlap}'}")
    print(f"  combined: {len(all_ids)} total rows, {all_ids.nunique()} unique ids "
          f"-- {'PASS' if combined_dup == 0 else f'FAIL ({combined_dup} collisions)'}")

    if aave_dup or sae_dup or overlap or combined_dup:
        print("\n*** COLLISION REMAINS -- STOPPING WITHOUT WRITING ANY FILE. ***")
        if overlap:
            print("Overlapping ids:", overlap)
        if aave_dup:
            print("aave_poetry.csv duplicated ids:", aave2[aave2['id'].duplicated(keep=False)]['id'].tolist())
        if sae_dup:
            print("sae_poetry.csv duplicated ids:", sae2[sae2['id'].duplicated(keep=False)]['id'].tolist())
        raise SystemExit(1)

    print("\n  ALL CHECKS PASSED.")

    # ================== write ==================
    aave2.to_csv(AAVE_PATH, index=False)
    sae2.to_csv(SAE_PATH, index=False)
    log.info("Wrote %s (%d rows) and %s (%d rows)", AAVE_PATH, len(aave2), SAE_PATH, len(sae2))

    # ================== final report ==================
    print("\n" + "=" * 70)
    print("FINAL COUNTS")
    print("=" * 70)
    print(f"\naave_poetry.csv: {len(aave2)} rows")
    print(aave2.groupby(["source_subset", "variety"]).size().to_string())
    print(f"\n  token distribution by variety:")
    print(aave2.groupby("variety")["n_words"].agg(["count", "mean", "median", "min", "max"]).round(1).to_string())
    print(f"\n  selection_source breakdown:")
    print(aave2.groupby(["source_subset", "selection_source"]).size().to_string())

    print(f"\nsae_poetry.csv: {len(sae2)} rows")
    print(sae2.groupby("source_subset").size().to_string())
    print(f"\n  token distribution:")
    print(sae2.groupby("source_subset")["n_words"].agg(["count", "mean", "median", "min", "max"]).round(1).to_string())
    print(f"\n  overall: mean={sae2['n_words'].mean():.1f} median={sae2['n_words'].median():.1f} "
          f"min={sae2['n_words'].min()} max={sae2['n_words'].max()}")

    return aave2, sae2


if __name__ == "__main__":
    main()
