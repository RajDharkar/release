"""
Loader for Langston Hughes's public-domain poems -- Standard Ebooks "Poetry"
compilation (single-page XHTML), the US-PD subset of his work.

Raw file: data/raw/pd_poetry/hughes_se.html

STRUCTURE
Each poem/song is its own <article>/<section epub:type="...z3998:poem"> or
"...z3998:song"> element with an <h2>/<h3> title -- the cleanest boundary marker
of the four sources, no ambiguity. Stripped: titlepage, imprint, colophon,
uncopyright sections (all separately tagged, non-bodymatter epub:types).

License: the Standard Ebooks "Uncopyright" page states the compiled writing is
believed to be US public domain. The colophon states the compilation covers poems
"published between 1921 and 1928" as an aggregate range for the whole ebook -- it
does NOT carry per-poem original-collection or original-publication-year metadata
in this file (checked: no "Fine Clothes" or collection-name string appears anywhere
in the source HTML). See VOLUME_YEAR_NOTE below.

ORTHOGRAPHY / REGISTER
Of Standard Ebooks' 229 poems/songs, only a minority carry heavy orthographic
dialect marking; the rest are standard-orthography free verse (vernacular in
register/theme, not spelling). Per the scoping report, only the strongly marked
subset is loaded here -- orthography="light" relative to the 19th-century sources,
reflecting that even the included poems use lighter, more consistent dialect
spelling than Dunbar/Henderson/Campbell.

SELECTION -- explicit title list (HUGHES_INCLUDE_TITLES), not a runtime heuristic.
The list was derived by scoring every poem's density of a narrow dialect-marker
set (de/dat/dis/dem/gwine/wid/yo'/ma/i'se/ain't/'bout/'cause/gonna/lawd/hunh/sho/
fo'/mo'/no'th/husban' + word-final -in' endings) and taking the poems above a
natural break in the ranking. The top 22 poems by this score have no close
runners-up; the 23rd/24th slot is a genuine tie (Moan and Negro Dancer, both 8.1%
density, 5 marker hits in 62 words) -- both are included rather than forcing an
arbitrary alphabetical tie-break, so the final list has 24 titles, not exactly 23.
Both tied poems were read in full and confirmed as heavy-dialect blues pieces
consistent with the rest of the list (see source6_pd_poetry.py's printed
derivation for the full ranked table). All 24 are thematically and stylistically
of a piece with Fine Clothes to the Jew (1927) -- gambling, hard-luck romance,
migration, and blues-gospel register -- consistent with the task's expectation of
"~23, mostly the blues poems from Fine Clothes to the Jew."
"""
from __future__ import annotations

import html
import re
from pathlib import Path

RAW_PATH = Path("data/raw/pd_poetry/hughes_se.html")

POET = "Langston Hughes"
SOURCE_SUBSET = "hughes"
ORTHOGRAPHY = "light"

# Best estimate, not verified per-poem in the source file: Standard Ebooks' colophon
# states only an aggregate 1921-1928 range for the whole 229-poem compilation and
# carries no per-poem original-collection metadata. All 24 titles below are, by
# general bibliographic record, blues-form dialect poems consistent with Fine
# Clothes to the Jew (1927) -- used as the best-estimate year -- but this is a
# judgment call, not a verified fact from the fetched source, so every row from
# this loader is flagged needs_review=True for a bibliography spot-check.
VOLUME_YEAR_ESTIMATE = 1927
VOLUME_YEAR_NOTE = (
    "volume_year is a best estimate (Fine Clothes to the Jew, 1927) based on general "
    "bibliographic knowledge, not per-poem metadata verified in the Standard "
    "Ebooks source file (whose colophon states only an aggregate 1921-1928 range "
    "for the full compiled ebook)"
)

# Explicit selection list, derived as described in the module docstring.
HUGHES_INCLUDE_TITLES = [
    "Hey!",
    "Ma Lord",
    "Homesick Blues",
    "Fire",
    "Angels Wings",
    "Hey! Hey!",
    "Red Roses",
    "A Ruined Gal",
    "Wide River",
    "Judgment Day",
    "Workin’ Man",
    "Midwinter Blues",
    "Suicide",
    "Young Gal’s Blues",
    "Cora",
    "Hard Daddy",
    "Baby",
    "Gal’s Cry for a Dying Lover",
    "Evil Woman",
    "Prayer Meeting",
    "Bound No’th Blues",
    "Crap Game",
    "Moan",
    "Negro Dancer",
]


def _norm(title: str) -> str:
    t = title.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _parse_poems(raw: str) -> list[dict]:
    parts = re.split(r'<(?:article|section)[^>]*epub:type="bodymatter[^"]*"[^>]*>', raw)[1:]
    poems = []
    for p in parts:
        tm = re.search(r"<h[23][^>]*>(.*?)</h[23]>", p, re.S)
        if not tm:
            continue
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", tm.group(1)))).strip()
        text = html.unescape(re.sub(r"</p>|<br/?>", "\n", p))
        text = re.sub(r"<[^>]+>", "", text)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # first surviving line is the title itself; drop it
        body = "\n".join(lines[1:]).strip()
        poems.append({"title": title, "text": body})
    return poems


def load(raw_path: Path = RAW_PATH) -> tuple[list[dict], dict]:
    raw = raw_path.read_text(encoding="utf-8", errors="replace")
    all_poems = _parse_poems(raw)
    by_norm = {_norm(p["title"]): p for p in all_poems}

    records = []
    missing = []
    for title in HUGHES_INCLUDE_TITLES:
        poem = by_norm.get(_norm(title))
        if poem is None:
            missing.append(title)
            continue
        records.append(
            {
                "poem_title": poem["title"],
                "text": poem["text"],
                "poet": POET,
                "volume_year": VOLUME_YEAR_ESTIMATE,
                "source_subset": SOURCE_SUBSET,
                "orthography": ORTHOGRAPHY,
                "variety": "AAVE",
                "needs_review": True,
                "needs_review_reason": VOLUME_YEAR_NOTE,
            }
        )

    stats = {
        "n_total_in_compilation": len(all_poems),
        "n_include_list": len(HUGHES_INCLUDE_TITLES),
        "n_loaded": len(records),
        "n_missing": len(missing),
        "missing_titles": missing,
    }
    return records, stats


SAE_VOLUME_YEAR_NOTE = (
    "volume_year unknown -- this poem was not part of the ~24-title heavy-dialect "
    "selection (whose Fine Clothes to the Jew, 1927 estimate was justified by "
    "thematic consistency with that collection), so no comparable per-poem "
    "year estimate was researched for it. The Standard Ebooks colophon only "
    "gives an aggregate 1921-1928 range for the whole compiled ebook."
)


def load_sae(raw_path: Path = RAW_PATH) -> tuple[list[dict], dict]:
    """The ~205 standard-orthography poems NOT on HUGHES_INCLUDE_TITLES."""
    raw = raw_path.read_text(encoding="utf-8", errors="replace")
    all_poems = _parse_poems(raw)
    included_norm = {_norm(t) for t in HUGHES_INCLUDE_TITLES}

    records = []
    for p in all_poems:
        if _norm(p["title"]) in included_norm or not p["text"].strip():
            continue
        records.append(
            {
                "poem_title": p["title"],
                "text": p["text"],
                "poet": POET,
                "volume_year": None,
                "source_subset": SOURCE_SUBSET,
                "orthography": "standard",
                "variety": "SAE",
                "needs_review": True,
                "needs_review_reason": SAE_VOLUME_YEAR_NOTE,
            }
        )

    stats = {
        "n_total_in_compilation": len(all_poems),
        "n_excluded_from_aave_list": len(included_norm),
        "n_loaded_sae": len(records),
    }
    return records, stats
