"""
Loader for The Book of American Negro Poetry (1922, ed. James Weldon Johnson) --
Project Gutenberg #11986.

Raw file: data/raw/pd_poetry/johnson1922.txt

STRUCTURE
The anthology proper starts at the line "THE BOOK OF AMERICAN NEGRO POETRY" (the
title repeated as a section head right before Dunbar's first poem). Everything
before that -- including Johnson's ~40-page preface -- is front matter and must NOT
be parsed as content: the preface itself quotes several poems in full (including at
least one Campbell dialect poem), so a naive title search over the whole file would
double-count or wrongly attribute preface quotations.

Within the anthology body, author bylines are 4-space-indented mixed-case lines
matching a TOC author name; poem titles are flush-left ALL-CAPS lines. Two quirks
handled here:
  - Titles beginning with an apostrophe ('Weh Down Souf, 'Ittle Touzle Head) parse
    fine under the same flush-left-caps regex used for Dunbar (it allows a leading
    apostrophe/quote).
  - Footnote markers ("A NEGRO LOVE SONG[1]") are stripped from titles.
  - Braithwaite's "Sandy Star and Willie Gee" has numbered roman-numeral sub-poems
    (I. Sculptured Worship, etc.) -- irrelevant here since Braithwaite is out of
    scope for this loader (not in the author allowlist below).

Back matter (Plácido appendix, biographical index, index of titles) is excluded by
construction: the poem parser stops consuming once no author byline or poem title
is found before EOF, and the appendix's non-caps prose body simply accumulates as
"body" of the last real poem parsed unless bounded. To avoid that, parsing is
truncated at the first back-matter marker line (APPENDIX / BIOGRAPHICAL INDEX OF
AUTHORS / INDEX OF TITLES).

AUTHOR ALLOWLIST (never subject/topic tags): Campbell, Corrothers, Davis, Holloway,
Rogers, Dandridge. Dunbar's 8 poems in this anthology are explicitly excluded --
they duplicate his Complete Poems (source6_pd_poetry_dunbar.py) and would double the
poem's weight in the corpus if kept in both places.

SELECTION -- explicit manual title list (JOHNSON_DIALECT_TITLES), NOT the
marker-density heuristic. The heuristic was used only during scoping to narrow down
candidates and is re-run by the combining loader as a validation check afterward.
Two poems needed a manual read of the full text because their marker-density score
(using a narrow de/dat/gwine/wid/yo'/hyeah word list) came back low despite being
unambiguously heavy dialect -- they use dialect spellings the narrow heuristic
doesn't recognize (da, fuh, erbout, mah, bekaze, natchel, ...):
  - Rogers, "Why Adam Sinned" and "The Rain Song" (3.2-3.6% by the narrow heuristic)
  - Dandridge, "'Ittle Touzle Head"
All three are included in JOHNSON_DIALECT_TITLES and flagged needs_review=True with
this reason so the discrepancy is visible downstream.

Campbell's 5 dialect poems (all but "Compensation", which is standard English) are
Gullah, not AAVE -- kept in this loader's output but tagged variety="Gullah" per
the exclusion rule from the scoping report, so they are never silently folded into
the AAVE pool.
"""
from __future__ import annotations

import re
from pathlib import Path

RAW_PATH = Path("data/raw/pd_poetry/johnson1922.txt")

VOLUME_YEAR = 1922
SOURCE_SUBSET = "johnson1922"
ORTHOGRAPHY = "heavy"
ANTHOLOGY_START_MARKER = "THE BOOK OF AMERICAN NEGRO POETRY"
BACK_MATTER_MARKERS = ("APPENDIX", "BIOGRAPHICAL INDEX OF AUTHORS", "INDEX OF TITLES")

TOC_AUTHORS = {
    "JAMES EDWIN CAMPBELL": "James Edwin Campbell",
    "JAMES D. CORROTHERS": "James D. Corrothers",
    "DANIEL WEBSTER DAVIS": "Daniel Webster Davis",
    "JOHN WESLEY HOLLOWAY": "John Wesley Holloway",
    "ALEX ROGERS": "Alex Rogers",
    "RAY G. DANDRIDGE": "Ray G. Dandridge",
}

# TOC start/end line offsets (relative to the anthology-wide line split used in
# _parse_toc_authors) bounding the "AUTHOR\n  Title\n  Title\n..." TOC block.
TOC_SLICE = slice(24, 214)

# Explicit manual title list: author (TOC caps key) -> [(title, variety, needs_review_reason)]
JOHNSON_DIALECT_TITLES: dict[str, list[tuple[str, str, str | None]]] = {
    "JAMES EDWIN CAMPBELL": [
        ("Negro Serenade", "Gullah", None),
        ("De Cunjah Man", "Gullah", None),
        ("Uncle Eph's Banjo Song", "Gullah", None),
        ("Ol' Doc' Hyar", "Gullah", None),
        ("When Ol' Sis' Judy Pray", "Gullah", None),
        # "Compensation" excluded: standard English, not dialect.
    ],
    "JAMES D. CORROTHERS": [
        ("An Indignation Dinner", "AAVE", None),
    ],
    "DANIEL WEBSTER DAVIS": [
        ("'Weh Down Souf", "AAVE", None),
        ("Hog Meat", "AAVE", None),
    ],
    "JOHN WESLEY HOLLOWAY": [
        ("Miss Melerlee", "AAVE", None),
        ("Calling the Doctor", "AAVE", None),
        ("The Corn Song", "AAVE", None),
        ("Black Mammies", "AAVE", None),
    ],
    "ALEX ROGERS": [
        ("Why Adam Sinned", "AAVE",
         "marker-density heuristic scored 3.6% (below the ~7%+ typical of this "
         "corpus) because it uses dialect spellings (da, fuh, knowed) outside the "
         "heuristic's narrow word list; manual read confirms heavy AAVE dialect"),
        ("The Rain Song", "AAVE",
         "marker-density heuristic scored 3.2% for the same reason as 'Why Adam "
         "Sinned'; manual read confirms heavy AAVE dialect"),
    ],
    "RAY G. DANDRIDGE": [
        # "Time to Die" removed: manual review confirmed standard English, not dialect.
        ("Sprin' Fevah", "AAVE", None),
        ("De Drum Majah", "AAVE", None),
        ("'Ittle Touzle Head", "AAVE",
         "marker-density heuristic scored near zero because the poem uses dialect "
         "spellings (erbout, mah, bekaze, natchel) outside the heuristic's narrow "
         "word list; manual read confirms heavy AAVE dialect"),
        # "Zalka Peetruza" excluded: standard English, not dialect.
    ],
}

TITLE_RE = re.compile(r"^[A-Z\"'][A-Z0-9 .,;:'\"!?()&\-\[\]]*$")
FOOTNOTE_RE = re.compile(r"\[\d+\]")


def _squash(name: str) -> str:
    """Alphanumeric-only, lowercased. Body bylines format initials tighter than
    the TOC listing does (e.g. body 'W.E. Burghardt Du Bois' vs TOC 'W. E.
    BURGHARDT DU BOIS'); squashing both to bare letters makes them comparable."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _norm(title: str) -> str:
    t = title.lower().strip("'").strip()
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _parse_toc_authors(lines: list[str]) -> set[str]:
    """All ~31 author names (TOC caps form) in the anthology's table of contents,
    used only to recognize EVERY author-byline boundary in the body -- not just
    our 6-author allowlist -- so a poem's body never bleeds into the next
    (out-of-scope) author's section."""
    authors = set()
    cur = None
    for l in lines[TOC_SLICE]:
        st = l.rstrip()
        if not st.strip():
            continue
        if not st.startswith(" "):
            cur = st.strip()
            if cur != "PREFACE":
                authors.add(cur)
    return authors


def _parse_anthology(text: str) -> dict[str, dict[str, tuple[str, str]]]:
    """Returns {author_toc_key: {normalized_title: (original_title, body_text)}}
    for the target authors only, parsed strictly from the anthology body (after
    the preface, before the back matter). The original-case title is kept
    alongside the normalized key (not just discarded) so load_sae() can recover
    a properly-cased poem_title for titles that aren't on the manual dialect
    list and therefore have no explicit-list casing to fall back on."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    s = text.index(start_marker)
    s = text.index("\n", s)
    e = text.index(end_marker)
    full = text[s:e]
    all_lines = full.split("\n")
    all_authors = _parse_toc_authors(all_lines)
    all_authors_squashed = {_squash(a): a for a in all_authors}

    anth_start = full.index(ANTHOLOGY_START_MARKER)
    # skip past the marker line itself
    anth_start = full.index("\n", anth_start)
    back_positions = [full.index(m) for m in BACK_MATTER_MARKERS if m in full]
    anth_end = min(back_positions) if back_positions else len(full)

    lines = full[anth_start:anth_end].split("\n")

    target_authors_squashed = {_squash(v): k for k, v in TOC_AUTHORS.items()}
    result: dict[str, dict[str, str]] = {k: {} for k in TOC_AUTHORS}

    cur_author = None  # TOC_AUTHORS key, or None if current section is out of scope
    cur_title = None
    cur_lines: list[str] = []

    def flush():
        if cur_author in result and cur_title and any(x.strip() for x in cur_lines):
            body = "\n".join(cur_lines).strip()
            result[cur_author][_norm(cur_title)] = (cur_title, body)

    for l in lines:
        st = l.rstrip()
        sp = st.strip()
        if st.startswith("    ") and _squash(sp) in all_authors_squashed:
            # Author-byline boundary -- recognized for ALL anthology authors, not
            # just our target 6, so a target author's last poem never swallows
            # the next (out-of-scope) author's byline as body text.
            flush()
            cur_author = target_authors_squashed.get(_squash(sp))  # None if out of scope
            cur_title, cur_lines = None, []
        elif sp and not st.startswith(" ") and TITLE_RE.match(sp) and len(sp) > 1:
            flush()
            cur_title = FOOTNOTE_RE.sub("", sp).strip()
            cur_lines = []
        else:
            cur_lines.append(l)
    flush()

    return result


def load(raw_path: Path = RAW_PATH) -> tuple[list[dict], dict]:
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_anthology(raw_text)

    records = []
    missing = []
    for author_key, title_specs in JOHNSON_DIALECT_TITLES.items():
        author_name = TOC_AUTHORS[author_key]
        bodies = parsed.get(author_key, {})
        for title, variety, reason in title_specs:
            entry = bodies.get(_norm(title))
            if entry is None:
                missing.append(f"{author_name}: {title}")
                continue
            _orig_title, body = entry
            records.append(
                {
                    "poem_title": title,
                    "text": body,
                    "poet": author_name,
                    "volume_year": VOLUME_YEAR,
                    "source_subset": SOURCE_SUBSET,
                    "orthography": ORTHOGRAPHY,
                    "variety": variety,
                    "needs_review": reason is not None,
                    "needs_review_reason": reason,
                }
            )

    n_aave = sum(1 for r in records if r["variety"] == "AAVE")
    n_gullah = sum(1 for r in records if r["variety"] == "Gullah")

    stats = {
        "n_loaded": len(records),
        "n_aave": n_aave,
        "n_gullah": n_gullah,
        "n_missing": len(missing),
        "missing_titles": missing,
        "n_authors_parsed": {k: len(v) for k, v in parsed.items()},
    }
    return records, stats


def load_sae(raw_path: Path = RAW_PATH) -> tuple[list[dict], dict]:
    """SAE complement: for each of the 6 allowlisted authors, every parsed poem
    NOT on JOHNSON_DIALECT_TITLES -- e.g. Campbell's "Compensation" (his one
    standard-English poem, explicitly noted as excluded from the Gullah list
    in this module's docstring), Dandridge's "Time to Die" and "Zalka
    Peetruza" (both explicitly commented above as "standard English, not
    dialect"). Stays within the same 6-author allowlist as load() -- does not
    pull in any of the anthology's other ~25 authors, who were never in scope
    for this loader."""
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_anthology(raw_text)

    selected_norm = {
        author_key: {_norm(t) for t, _, _ in specs}
        for author_key, specs in JOHNSON_DIALECT_TITLES.items()
    }

    records = []
    for author_key, bodies in parsed.items():
        author_name = TOC_AUTHORS[author_key]
        selected = selected_norm.get(author_key, set())
        for norm_title, (orig_title, body) in bodies.items():
            if norm_title in selected or not body.strip():
                continue
            records.append(
                {
                    "poem_title": orig_title,
                    "text": body,
                    "poet": author_name,
                    "volume_year": VOLUME_YEAR,
                    "source_subset": SOURCE_SUBSET,
                    "orthography": "standard",
                    "variety": "SAE",
                    "needs_review": False,
                    "needs_review_reason": None,
                }
            )

    stats = {
        "n_authors_parsed": {k: len(v) for k, v in parsed.items()},
        "n_loaded_sae": len(records),
    }
    return records, stats
