"""
Prompt templates for the MGT generation harness (Part 3).

No template ever names or requests AAVE, dialect, or vernacular -- both
line 2 (SAE-HWT source) and line 4 (DIA-HWT source) use the SAME
templates; the only difference is which row supplies the topic/source
text. DIA-MGT is produced later by running Multi-VALUE over this
harness's SAE-MGT output (arm D) -- never by asking a model to write in
dialect directly (see docs/decisions.md).

Every render is traceable: TEMPLATE_VERSION is bumped whenever wording
changes, and template_id encodes (genre, mode).
"""
from __future__ import annotations

TEMPLATE_VERSION = "v3"  # v3 (poetry pilot): mode_b now explicitly instructs
# converting non-standard spellings/grammar/vocabulary into standard forms
# (not just "preserve register," which v2 already tried but which the v2
# live retest showed was still insufficient on the hardest, most heavily
# dialect-marked / explicit sources -- see reports/generation_scaffold_report.md,
# TEMPLATE v2 RETEST). mode_a's wording is also tightened to match the
# same terser, noun-based phrasing style, applied generally across genres
# via _GENRE_NOUN (this pilot only exercises the poetry case).
#
# v2: mode_b rewritten to explicitly require standard written English --
# v1 only said "preserve tone/register," which the live smoke test
# confirmed let models reproduce dialect-spelled source text (including
# slurs) near-verbatim under a "rewrite" framing.

GENRES = ("poetry", "short_fiction", "oratory", "lyrics")
MODES = ("a", "b")  # a = topic (new piece), b = rewrite (in-context)

_GENRE_LABEL = {
    "poetry": "a poem",
    "short_fiction": "a short piece of fiction",
    "oratory": "a short speech",
    "lyrics": "a song's lyrics",
}

# Singular noun form for mode_b's "the following {genre}" / "the rewritten
# {genre}" phrasing -- distinct from _GENRE_LABEL's "a poem"/"a short piece
# of fiction" article form, which doesn't read naturally in that slot.
_GENRE_NOUN = {
    "poetry": "poem",
    "short_fiction": "short story",
    "oratory": "speech",
    "lyrics": "song lyrics",
}


def template_id(genre: str, mode: str) -> str:
    if genre not in GENRES:
        raise ValueError(f"Unknown genre: {genre!r}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode!r}")
    return f"{genre}_{mode}_{TEMPLATE_VERSION}"


def render_mode_a(genre: str, topic: str, target_length: int | None) -> str:
    """mode_a ('topic'): the source text itself must NOT appear here --
    only a topic/theme string derived from it (see topics.py). Poetry
    wording matches the pilot spec exactly ("Write a poem on the
    following topic: {topic}. Write only the poem, no title or
    commentary."); generalized to other genres via _GENRE_NOUN so the
    module keeps one shared template rather than forking a poetry-only
    variant."""
    noun = _GENRE_NOUN.get(genre, genre)
    length_clause = (
        f"Aim for approximately {target_length} words."
        if target_length else
        "Keep it a natural, self-contained length for the form."
    )
    return (
        f"Write a {noun} on the following topic: {topic}. "
        f"Write only the {noun}, no title or commentary. {length_clause}"
    )


def render_mode_b(genre: str, source_text: str, target_length: int | None) -> str:
    """mode_b ('rewrite'): the source text IS included, in-context. v3
    conversion wording (poetry pilot) -- explicitly instructs converting
    non-standard spellings/grammar/vocabulary into standard forms, not
    just "preserve register" (v2), which the v2 live retest showed was
    still insufficient on the hardest sources. One shared instruction
    across all genres and both lines, as established in v2."""
    noun = _GENRE_NOUN.get(genre, genre)
    length_clause = (
        f"Keep the rewrite to approximately {target_length} words."
        if target_length else
        "Keep the rewrite close to the original's length."
    )
    return (
        f"Rewrite the following {noun} in standard written English. "
        f"Convert any non-standard spellings, grammar, and vocabulary "
        f"into standard forms, while keeping the meaning, imagery, and "
        f"emotional weight intact. Output only the rewritten {noun}.\n\n"
        f"--- SOURCE TEXT ---\n{source_text}\n--- END SOURCE TEXT ---\n\n"
        f"{length_clause}"
    )


def render(genre: str, mode: str, *, topic: str | None = None,
           source_text: str | None = None, target_length: int | None = None) -> str:
    tid = template_id(genre, mode)
    if mode == "a":
        if topic is None:
            raise ValueError("mode_a requires `topic`")
        prompt = render_mode_a(genre, topic, target_length)
    else:
        if source_text is None:
            raise ValueError("mode_b requires `source_text`")
        prompt = render_mode_b(genre, source_text, target_length)
    return prompt, tid


# --------------------------------------------------------------------------
# direct_aave (Direct-AAVE generation arm) -- a THIRD mode, structurally
# separate from mode_a/mode_b's shared a/b v3 lineage above. This is the
# first template in the project to explicitly ask a model to write in
# AAVE (mode_a/mode_b never name a dialect at all -- see this module's own
# module docstring). Approved wording, per-genre, after an explicit
# design-review round (see reports/direct_aave_arm.md):
#   - POETRY: dialect requested throughout (vocabulary + grammar only --
#     "spelling" was deliberately dropped from the approved wording so
#     orthographic density is a model decision, not a prompt instruction;
#     see the lyrics note below for why this asymmetry is intentional).
#   - FICTION: SAE narration, AAVE dialogue -- structurally encodes the
#     dialogue-concentration finding rather than leaving it to chance.
#     Deliberately carries NO "don't exaggerate" brake: this arm exists to
#     MEASURE whether models produce authentic or caricatured dialect, and
#     instructing against exaggeration would suppress the exact failure
#     mode under test.
#   - LYRICS: AAVE grammar/vocabulary requested, but standard spelling is
#     explicitly required ("do not respell phonetically"). This is the
#     ONLY one of the three prompts that constrains orthography by
#     instruction -- disclosed intentionally: it means any orthographic-
#     density gap observed between poetry and lyrics output is partly a
#     property of the prompts themselves, not purely of model behavior.
# No feature checklist is named in any of the three (no "habitual be",
# "copula deletion", etc.) -- naming specific eWAVE features risks the
# same mechanical over-injection artifact found in Arm C (Multi-VALUE
# double modals at ~1,850x the authentic rate).
DIRECT_AAVE_TEMPLATE_VERSION = "v1"

_DIRECT_AAVE_NOUN = {"poetry": "poem", "fiction": "short story", "lyrics": "song lyrics"}

_DIRECT_AAVE_PROMPTS = {
    "poetry": (
        "Write a poem on the following topic: {topic}. Write the poem in African "
        "American Vernacular English (AAVE) -- let the dialect's vocabulary and "
        "grammar carry through the whole poem, not just isolated lines. Write only "
        "the poem, no title or commentary. {length_clause}"
    ),
    "fiction": (
        "Write a short piece of fiction on the following topic: {topic}. Write the "
        "narration in standard written English. Give at least one character spoken "
        "dialogue in African American Vernacular English (AAVE), so the dialect "
        "appears in what that character says, not in the narration around them. "
        "Write only the story, no title or commentary. {length_clause}"
    ),
    "lyrics": (
        "Write song lyrics on the following topic: {topic}. Write the lyrics in "
        "African American Vernacular English (AAVE) -- carry the dialect's grammar, "
        "word choice, and phrasing throughout -- but write everything in standard "
        "English spelling; do not respell words phonetically to represent "
        "pronunciation. Write only the lyrics, no title or commentary. {length_clause}"
    ),
}

DIRECT_AAVE_GENRES = tuple(_DIRECT_AAVE_PROMPTS)


def direct_aave_template_id(genre: str) -> str:
    if genre not in _DIRECT_AAVE_PROMPTS:
        raise ValueError(f"Unknown direct_aave genre: {genre!r}")
    return f"{genre}_direct_aave_{DIRECT_AAVE_TEMPLATE_VERSION}"


def render_direct_aave(genre: str, topic: str, target_length: int | None) -> tuple[str, str]:
    if genre not in _DIRECT_AAVE_PROMPTS:
        raise ValueError(f"Unknown direct_aave genre: {genre!r}")
    noun = _DIRECT_AAVE_NOUN[genre]
    length_clause = (
        f"Aim for approximately {target_length} words."
        if target_length else
        f"Keep it a natural, self-contained length for the {noun}."
    )
    prompt = _DIRECT_AAVE_PROMPTS[genre].format(topic=topic, length_clause=length_clause)
    return prompt, direct_aave_template_id(genre)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import pandas as pd
    from src.generate.topics import extract_topic

    REPO_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED = REPO_ROOT / "data/processed"

    # One real source row per genre, drawn from whatever's available
    # without touching unified.csv/arm_c_mv_perturbation.csv.
    examples = {}
    poetry_df = pd.read_csv(PROCESSED / "aave_poetry.csv")
    examples["poetry"] = poetry_df.iloc[0]["text"]

    book_df = pd.read_csv(PROCESSED / "aave_book.csv", low_memory=False, nrows=50)
    examples["short_fiction"] = book_df.iloc[0]["text"]

    lyric_df = pd.read_csv(PROCESSED / "aave_lyric.csv", low_memory=False, nrows=50)
    examples["lyrics"] = lyric_df.iloc[0]["text"]

    oratory_path = PROCESSED / "sae_oratory.csv"
    if oratory_path.exists():
        oratory_df = pd.read_csv(oratory_path, low_memory=False, nrows=50)
        examples["oratory"] = oratory_df[oratory_df["needs_review"] == False].iloc[0]["text"]  # noqa: E712
    else:
        examples["oratory"] = None

    for genre in GENRES:
        source_text = examples.get(genre)
        print("=" * 70)
        print(f"GENRE: {genre}")
        print("=" * 70)
        if source_text is None:
            print("  (no source row available -- skipped)")
            continue

        topic = extract_topic(source_text)
        prompt_a, tid_a = render(genre, "a", topic=topic, target_length=None)
        print(f"\n--- mode_a [{tid_a}] ---")
        print(prompt_a)

        prompt_b, tid_b = render(genre, "b", source_text=source_text, target_length=None)
        print(f"\n--- mode_b [{tid_b}] ---")
        print(prompt_b[:600] + ("..." if len(prompt_b) > 600 else ""))
        print()
