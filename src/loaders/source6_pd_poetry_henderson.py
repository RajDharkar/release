"""
Loader for Elliott Blaine Henderson's Plantation Echoes (1904) --
Project Gutenberg #66997.

Raw file: data/raw/pd_poetry/henderson1904.txt

STRUCTURE
Poem titles are flush-left ALL-CAPS lines ending in a period. The volume is split
on BODY titles, not TOC titles: the table of contents spells several titles
differently from the poems themselves (e.g. TOC "LAMMENTATION" vs body
"LAMENTATION", TOC "KASE DE SUN AM SINKIN'" vs body "KAZE DE SUN AM SINKIN'"),
so keying selection off the TOC would silently fail to match on ~11 of 51 titles.
The TOC is used only to confirm the total poem count (51 vs 51 parsed), not for
title lookup.

Stripped: transcriber's content note (re: dialect/stereotype content), title page,
dedication, Burkham's introduction, table of contents.

EXCLUSION -- 4 of the 51 poems are standard English, not dialect, and are dropped:
  When the Moon Hangs Low, Soft Falls the Night, Memorial Day,
  The Gallant Blacks of San Juan Hill
Expected yield: 47.
"""
from __future__ import annotations

import re
from pathlib import Path

RAW_PATH = Path("data/raw/pd_poetry/henderson1904.txt")

POET = "Elliott Blaine Henderson"
VOLUME_YEAR = 1904
SOURCE_SUBSET = "henderson"
ORTHOGRAPHY = "heavy"

STANDARD_ENGLISH_TITLES = {
    "WHEN THE MOON HANGS LOW",
    "SOFT FALLS THE NIGHT",
    "MEMORIAL DAY",
    "THE GALLANT BLACKS OF SAN JUAN HILL",
}

TITLE_RE = re.compile(r"^[A-Z][A-Z0-9 .,;:'’!?—\-()\"“”]*$")
BODY_START_MARKER = "WHEN THE MOON HANGS LOW."


def _parse_poems(text: str) -> list[dict]:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    s = text.index(start_marker)
    s = text.index("\n", s)
    e = text.index(end_marker)
    lines = text[s:e].split("\n")

    body_start = next(i for i, l in enumerate(lines) if l.strip() == BODY_START_MARKER)

    poems: list[dict] = []
    cur_title, cur_lines = None, []
    for l in lines[body_start:]:
        st = l.rstrip()
        sp = st.strip()
        if sp and not st.startswith(" ") and TITLE_RE.match(sp) and len(sp) > 2:
            if cur_title is not None:
                poems.append({"title": cur_title.rstrip("."), "lines": cur_lines})
            cur_title, cur_lines = sp, []
        else:
            cur_lines.append(l)
    if cur_title is not None:
        poems.append({"title": cur_title.rstrip("."), "lines": cur_lines})
    return poems


def load(raw_path: Path = RAW_PATH) -> tuple[list[dict], dict]:
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    poems = _parse_poems(raw_text)

    records = []
    excluded = []
    for p in poems:
        title_key = p["title"].upper()
        body = "\n".join(p["lines"]).strip()
        if not body:
            continue
        if title_key in STANDARD_ENGLISH_TITLES:
            excluded.append(p["title"])
            continue
        records.append(
            {
                "poem_title": p["title"],
                "text": body,
                "poet": POET,
                "volume_year": VOLUME_YEAR,
                "source_subset": SOURCE_SUBSET,
                "orthography": ORTHOGRAPHY,
                "variety": "AAVE",
                "needs_review": False,
                "needs_review_reason": None,
            }
        )

    stats = {
        "n_total_parsed": len(poems),
        "n_loaded": len(records),
        "n_excluded_standard_english": len(excluded),
        "excluded_titles": excluded,
    }
    return records, stats


def load_sae(raw_path: Path = RAW_PATH) -> tuple[list[dict], dict]:
    """The 4 standard-English poems load() excludes: When the Moon Hangs Low,
    Soft Falls the Night, Memorial Day, The Gallant Blacks of San Juan Hill."""
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    poems = _parse_poems(raw_text)

    records = []
    for p in poems:
        if p["title"].upper() not in STANDARD_ENGLISH_TITLES:
            continue
        body = "\n".join(p["lines"]).strip()
        if not body:
            continue
        records.append(
            {
                "poem_title": p["title"],
                "text": body,
                "poet": POET,
                "volume_year": VOLUME_YEAR,
                "source_subset": SOURCE_SUBSET,
                "orthography": "standard",
                "variety": "SAE",
                "needs_review": False,
                "needs_review_reason": None,
            }
        )

    stats = {"n_loaded_sae": len(records), "n_expected": len(STANDARD_ENGLISH_TITLES)}
    return records, stats
