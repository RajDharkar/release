"""
Loader for the 1922 edition of Paul Laurence Dunbar's Complete Poems -- Project Gutenberg #18338.

Raw file: data/raw/pd_poetry/dunbar_complete.txt (full PG plain-text release)
Dialect title index: data/raw/pd_poetry/dunbar_dialect_index.html
  (paullaurencedunbar.org/2022/plangbd -- the site's own "Black dialect" tag list,
  98 titles, used as the selection filter instead of a marker-density heuristic)

STRUCTURE
Poem titles in the body are flush-left ALL-CAPS lines; bodies are 2-space indented.
The five "LYRICS OF ..." section headers (LOWLY LIFE / THE HEARTHSIDE / LOVE AND
LAUGHTER / LOVE AND SORROW / SUNSHINE AND SHADOW) are also flush-left caps and must
be excluded from the poem list.

Stripped: PG boilerplate, the alphabetical title-and-page-number index (precedes the
poems), W. D. Howells's introduction (poems begin at the first "LYRICS OF LOWLY LIFE"
heading AFTER the introduction, not the one in the index/dedication front matter),
per-section dedication pages, transcriber's notes (trailing).

SELECTION -- 98 titles on the dialect index, resolved as follows:
  - 84 match directly after normalizing case/punctuation/trailing-article form.
  - 9 titles need an explicit alias (spelling variants the 2022 site normalized
    differently than the component text, quoted-title punctuation, or first-line
    disambiguation for the 4 title collisions in the volume: HYMN x2, LULLABY x2,
    SONG x2, PARTED x2). See DUNBAR_ALIASES.
  - 5 titles are NOT recoverable from this Gutenberg text at all -- confirmed absent
    by direct search, not just a matching failure. Three are documented uncollected
    poems (Charity, Goin' Back, Sling Along; "Goin' Back" in particular is a
    *different, standard-English* poem about an old man at a train station, not the
    dialect poem the site describes). Two more surfaced only during this build pass
    and were NOT caught by the original scoping session (which assumed a 3-title drop
    and a 95-poem yield): Lullaby "Kiver up yo' haid, my little lady" and Song "De bee
    hit sip some honey f'om de matermony vine" -- both fetched from the site and
    searched for verbatim and by fragment in this text, with no match. This session's
    real yield is 93, not the previously assumed 95; see DUNBAR_DROP_NOT_IN_VOLUME.

A CABIN TALE requires a structural merge, not just a title alias: in the print
volume this poem's title line is immediately followed by a subtitle line ("THE YOUNG
MASTER ASKS FOR A STORY") before the body starts, and that subtitle line itself
satisfies the flush-left-caps title regex, so the naive parser splits it into a
titleless empty poem + a second "poem" titled after the subtitle. Handled by merging
the subtitle poem's body back onto "A CABIN TALE" (see _merge_cabin_tale).

NORMALIZATION
This edition prints contractions spaced ("I 's goin'", "would n't") -- see
DUNBAR_CONTRACTION_RE in this module; the rejoin rule and substitution count are
reported by the combining loader (source6_pd_poetry.py), not here.

[Illustration: ...] lines (referring to engravings/musical-score plates in the
original book) appear inline in a few poem bodies (all within "How Lucy Backslid")
and are stripped as non-text content.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

RAW_PATH = Path("data/raw/pd_poetry/dunbar_complete.txt")
INDEX_PATH = Path("data/raw/pd_poetry/dunbar_dialect_index.html")

POET = "Paul Laurence Dunbar"
VOLUME_YEAR = 1922  # title-page year; latest component copyright year is 1913
SOURCE_SUBSET = "dunbar"
ORTHOGRAPHY = "heavy"

SECTION_HEADERS = {
    "LYRICS OF LOWLY LIFE",
    "LYRICS OF THE HEARTHSIDE",
    "LYRICS OF LOVE AND LAUGHTER",
    "LYRICS OF LOVE AND SORROW",
    "LYRICS OF SUNSHINE AND SHADOW",
}

TITLE_RE = re.compile(r"^[A-Z\"'][A-Z0-9 .,;:'\"!?()\[\]&\-]*$")
ILLUSTRATION_RE = re.compile(r"^\[Illustration.*\]$", re.MULTILINE)

# Site titles that need an explicit alias to a (body_title, first_line_substring)
# pair. first_line_substring is None when the body title is unique (no collision);
# it disambiguates the 4 duplicate-title groups otherwise.
DUNBAR_ALIASES: dict[str, tuple[str, str | None]] = {
    "Cabin Tale: The Young Master Asks for a Story, A": ("A CABIN TALE", None),  # merged body
    "Chris'mus Is a-Comin'": ("CHRISMUS IS A-COMIN'", None),
    "Fishin'": ("FISHING", None),
    "Howdy, Honey, Howdy!": ('"HOWDY, HONEY, HOWDY!"', None),
    "Long To'ds Night": ("'LONG TO'DS NIGHT", None),
    "Hymn \"O li'l' lamb out in de col'\"": ("HYMN", "li'l' lamb"),
    "Lullaby \"Bedtime's come fu' little boys\"": ("LULLABY", "Bedtime"),
    "Parted \"De breeze is blowin' 'cross de bay\"": ("PARTED", "breeze is blowin'"),
    'Song "Wintah, summah, snow er shine"': ("SONG", "Wintah, summah"),
}

# Confirmed absent from this Gutenberg text by direct search (not a matching gap).
DUNBAR_DROP_NOT_IN_VOLUME = {
    "Charity",
    "Goin' Back",
    "Sling Along",
    "Lullaby \"Kiver up yo' haid, my little lady\"",
    "Song \"De bee hit sip some honey f'om de matermony vine\"",
}


def _norm(title: str) -> str:
    t = title.lower()
    m = re.match(r"^(.*), (the|a|an|de)$", t)
    if m:
        t = m.group(2) + " " + m.group(1)
    t = t.split(":")[0]
    t = re.sub(r"[^a-z' ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _load_index_titles(index_path: Path = INDEX_PATH) -> list[str]:
    raw = index_path.read_text(encoding="utf-8")
    return [
        html.unescape(t.strip())
        for _, t in re.findall(r'<a href="(/2022/p\d+)"[^>]*>([^<]+)</a>', raw)
    ]


def _parse_poems(text: str) -> list[dict]:
    """Return ordered list of {title, lines} for every flush-left-caps segment
    in the body (poems, plus a handful of subtitle/section artifacts that are
    simply not selected downstream)."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    s = text.index(start_marker)
    s = text.index("\n", s)
    e = text.index(end_marker)
    lines = text[s:e].split("\n")

    intro_idx = [i for i, l in enumerate(lines) if l.strip().startswith("INTRODUCTION TO LYRICS")]
    lyrics_idx = [i for i, l in enumerate(lines) if l.strip() == "LYRICS OF LOWLY LIFE"]
    poem_start = [i for i in lyrics_idx if i > intro_idx[-1]][0]

    poems: list[dict] = []
    cur_title, cur_lines = None, []
    for l in lines[poem_start:]:
        st = l.rstrip()
        sp = st.strip()
        if sp and not st.startswith(" ") and TITLE_RE.match(sp) and len(sp) > 1:
            if sp in SECTION_HEADERS:
                continue
            if cur_title is not None:
                poems.append({"title": cur_title, "lines": cur_lines})
            cur_title, cur_lines = sp, []
        else:
            cur_lines.append(l)
    if cur_title is not None:
        poems.append({"title": cur_title, "lines": cur_lines})
    return poems


def _merge_cabin_tale(poems: list[dict]) -> list[dict]:
    """A CABIN TALE's title line is immediately followed by a subtitle line that
    the naive parser reads as a second poem title; merge that second entry's body
    back onto A CABIN TALE and drop the orphan entry."""
    out = []
    i = 0
    while i < len(poems):
        p = poems[i]
        if p["title"] == "A CABIN TALE" and not any(x.strip() for x in p["lines"]):
            if i + 1 < len(poems):
                nxt = poems[i + 1]
                merged_lines = [f"  {nxt['title']}", ""] + nxt["lines"]
                out.append({"title": "A CABIN TALE", "lines": merged_lines})
                i += 2
                continue
        out.append(p)
        i += 1
    return out


def _clean_body(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = ILLUSTRATION_RE.sub("", text)
    text = "\n".join(l for l in text.split("\n"))
    return text.strip()


def _resolve_dialect_poems(all_poems: list[dict], index_titles: list[str]) -> tuple[dict, list[str], list[str]]:
    """Returns (chosen: {id(poem): poem}, dropped, unresolved) -- the exact same
    resolution logic load() uses, factored out so load_sae() can compute the
    complement (every parsed poem NOT chosen here) against the SAME parse pass,
    matched by object identity (safe even across the 4 duplicate-title groups,
    since `chosen` is keyed on the specific poem object selected, not its title
    string)."""
    by_title: dict[str, list[dict]] = {}
    for p in all_poems:
        by_title.setdefault(p["title"], []).append(p)

    norm_lookup: dict[str, dict] = {}
    for p in all_poems:
        norm_lookup.setdefault(_norm(p["title"]), p)

    chosen: dict[int, dict] = {}
    dropped = []
    unresolved = []
    for site_title in index_titles:
        if site_title in DUNBAR_DROP_NOT_IN_VOLUME:
            dropped.append(site_title)
            continue

        if site_title in DUNBAR_ALIASES:
            body_title, disambig = DUNBAR_ALIASES[site_title]
            candidates = by_title.get(body_title, [])
            poem = None
            if disambig is None:
                poem = candidates[0] if candidates else None
            else:
                for c in candidates:
                    first_line = next((x.strip() for x in c["lines"] if x.strip()), "")
                    if disambig.lower() in first_line.lower():
                        poem = c
                        break
            if poem is None:
                unresolved.append(site_title)
                continue
        else:
            n = _norm(site_title)
            poem = norm_lookup.get(n)
            if poem is None:
                unresolved.append(site_title)
                continue

        body = _clean_body(poem["lines"])
        if not body:
            unresolved.append(site_title)
            continue

        chosen[id(poem)] = poem

    return chosen, dropped, unresolved


def load(raw_path: Path = RAW_PATH, index_path: Path = INDEX_PATH) -> tuple[list[dict], dict]:
    """Returns (records, stats). Each record has keys: poem_title, text (raw body,
    pre-normalization), needs_review, needs_review_reason."""
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    all_poems = _parse_poems(raw_text)
    all_poems = _merge_cabin_tale(all_poems)
    index_titles = _load_index_titles(index_path)

    chosen, dropped, unresolved = _resolve_dialect_poems(all_poems, index_titles)

    # chosen preserves index_titles' iteration order (insertion order == the
    # order site_title was resolved above), so this matches the pre-refactor
    # output order exactly.
    records = []
    for poem in chosen.values():
        body = _clean_body(poem["lines"])
        records.append(
            {
                "poem_title": poem["title"],
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
        "n_index_titles": len(index_titles),
        "n_loaded": len(records),
        "n_dropped_not_in_volume": len(dropped),
        "dropped_titles": dropped,
        "n_unresolved": len(unresolved),
        "unresolved_titles": unresolved,
        "n_total_poems_in_volume": len(all_poems),
    }
    return records, stats


def load_sae(raw_path: Path = RAW_PATH, index_path: Path = INDEX_PATH) -> tuple[list[dict], dict]:
    """The SAE complement: every parsed poem in the volume NOT selected as one
    of the 98 dialect-index titles (~330 of the volume's 430 poems). Same
    parsing, same object-identity exclusion set as load()'s AAVE selection --
    computed from a fresh parse, not by mutating/reusing load()'s output."""
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    all_poems = _parse_poems(raw_text)
    all_poems = _merge_cabin_tale(all_poems)
    index_titles = _load_index_titles(index_path)

    chosen, _dropped, _unresolved = _resolve_dialect_poems(all_poems, index_titles)
    chosen_ids = set(chosen.keys())

    records = []
    for poem in all_poems:
        if id(poem) in chosen_ids:
            continue
        body = _clean_body(poem["lines"])
        if not body:
            continue
        records.append(
            {
                "poem_title": poem["title"],
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

    stats = {
        "n_total_poems_in_volume": len(all_poems),
        "n_dialect_selected": len(chosen),
        "n_loaded_sae": len(records),
    }
    return records, stats
