"""
Analysis metrics for MGT/HGT generation outputs (Part 1 of the metrics +
cost + partial-rewrite task).

Four independent measurements, deliberately NOT combined into one score:
  1. Readability (Flesch Reading Ease / Flesch-Kincaid Grade), plus a
     syllable_confidence flag -- dialect spellings break syllable/word
     counting, and that distortion is itself a result to report, not an
     artifact to normalize away. Scores are never adjusted for dialect.
  2. difflib word-level similarity (rewrite-vs-copy signal).
  3. The project's existing marker-density heuristic (imported, not
     reimplemented -- see src/loaders/source6_pd_poetry.py).
  4. A lexical flag scan for historical epithets that survive "standard
     English" conversion in standard spelling (marker density is blind to
     these, since they aren't dialect-spelled) -- reports matches only,
     never removes or alters text.
"""
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

log = logging.getLogger("analysis.metrics")

try:
    import textstat
    _HAVE_TEXTSTAT = True
except ImportError:  # pragma: no cover -- environment without textstat
    _HAVE_TEXTSTAT = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.loaders.source6_pd_poetry import _marker_density, MARKER_RE  # noqa: E402

marker_density = _marker_density  # public re-export; "keep the existing heuristic"

# --------------------------------------------------------------------------
# 1. Readability + syllable_confidence
# --------------------------------------------------------------------------

# Standard English dictionary, tried in order. All of these are plain
# newline-delimited wordlists (one word per line, no apostrophes) on a
# typical macOS/Linux box. If none is found, dictionary coverage degrades
# to the small built-in allowlist below and a warning is logged once --
# syllable_confidence becomes less reliable, not silently "fixed".
_DICT_PATHS = [
    Path("/usr/share/dict/words"),
    Path("/usr/share/dict/american-english"),
    Path("/usr/share/dict/british-english"),
    Path("/usr/local/share/dict/words"),
]

# Common contractions: not in /usr/share/dict/words (which has no
# apostrophe entries at all), but ordinary in clean Standard English --
# without this allowlist, EVERY text using "don't"/"I'm" would register
# spurious OOV hits unrelated to dialect. Dialect-specific contractions
# (gwine, 'bout, yo', w'en, ...) are deliberately NOT here -- failing
# lookup on those is the intended signal.
_CONTRACTION_ALLOWLIST = {
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "can't", "couldn't", "won't", "wouldn't",
    "shouldn't", "mightn't", "mustn't", "ain't", "let's", "that's",
    "there's", "here's", "what's", "who's", "it's", "he's", "she's",
    "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd",
    "we're", "we've", "we'll", "we'd", "they're", "they've", "they'll",
    "they'd",
}

# Matches the curly right-single-quote (U+2019) as an apostrophe too --
# LLM output routinely uses "wasn't" with a smart quote, and without this
# the regex splits it into two spurious tokens ("wasn" + "t") at the
# unmatched character, double-counting a single ordinary contraction as
# two OOV hits.
_WORD_RE = re.compile(r"[A-Za-z'’]+")
_OOV_THRESHOLD = 0.10  # >10% of tokens not in dictionary => syllable_confidence "low"

_dictionary: set[str] | None = None
_dictionary_warned = False


def _load_dictionary() -> set[str]:
    global _dictionary, _dictionary_warned
    if _dictionary is not None:
        return _dictionary
    for path in _DICT_PATHS:
        if path.exists():
            with open(path, encoding="utf-8", errors="ignore") as f:
                _dictionary = {line.strip().lower() for line in f if line.strip()}
            log.info("syllable_confidence dictionary: %s (%d words)", path, len(_dictionary))
            return _dictionary
    if not _dictionary_warned:
        log.warning(
            "No system dictionary found at any of %s -- falling back to a "
            "small built-in wordlist. syllable_confidence will be LESS "
            "reliable (more false 'low' flags on ordinary English) until "
            "this is fixed in the environment.", _DICT_PATHS
        )
        _dictionary_warned = True
    # Minimal fallback so the module still runs somewhere without a system
    # dictionary; deliberately small and not meant to be authoritative.
    _dictionary = set(_CONTRACTION_ALLOWLIST) | {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "is",
        "was", "were", "are", "be", "been", "for", "with", "as", "at", "by",
        "it", "he", "she", "they", "you", "we", "i", "this", "that", "not",
    }
    return _dictionary


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


# Common inflectional suffixes not reliably present as their own entries
# in a root-word system dictionary (/usr/share/dict/words lists "climb"
# but not "climbed"/"climbing"/"climbs"). Without stripping these first,
# ordinary past-tense/plural Standard English registers as OOV at a rate
# that swamps the genuine dialect-spelling signal this flag is meant to
# catch -- confirmed empirically: on a clean, fully SAE-converted mode_a
# poem with marker_density=0.000, pct_oov was still 0.12 before this fix,
# entirely from words like "hummed", "climbed", "listened". This is a
# lightweight-stemmer accuracy fix to the OOV *lookup*, not a correction
# of the Flesch score itself -- the readability numbers are untouched.
_SUFFIXES = ("'s", "es", "ed", "ing", "er", "est", "ly", "s")


def _is_known_word(token: str) -> bool:
    low = token.lower().replace("’", "'")  # normalize smart quote before any lookup
    if low in _CONTRACTION_ALLOWLIST:
        return True
    dictionary = _load_dictionary()
    if low in dictionary:
        return True
    stripped = low.strip("'")
    if stripped in dictionary:
        return True
    for suffix in _SUFFIXES:
        if stripped.endswith(suffix) and len(stripped) > len(suffix) + 1:
            stem = stripped[: -len(suffix)]
            if stem in dictionary:
                return True
            # handle consonant-doubling ("stopped" -> "stop") and
            # e-insertion ("hummed" -> "hum") for -ed/-ing
            if suffix in ("ed", "ing"):
                if len(stem) > 1 and stem[-1] == stem[-2] and stem[:-1] in dictionary:
                    return True
                if (stem + "e") in dictionary:
                    return True
    return False


def pct_oov_tokens(text: str) -> float:
    """Fraction (0-1) of whitespace/word tokens not found in the standard
    English dictionary (after allowing common contractions). Dialect
    spellings, slang, and proper nouns all count as OOV here -- no attempt
    is made to distinguish them, per the task's "do NOT normalize" rule."""
    tokens = _tokenize_words(text)
    if not tokens:
        return 0.0
    n_oov = sum(1 for t in tokens if not _is_known_word(t))
    return n_oov / len(tokens)


def _syllables_in_word(word: str) -> int:
    """Crude vowel-group heuristic, used only in the no-textstat fallback."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def _fallback_flesch(text: str) -> tuple[float, float]:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = _tokenize_words(text)
    n_sentences = max(len(sentences), 1)
    n_words = max(len(words), 1)
    n_syllables = sum(_syllables_in_word(w) for w in words) or 1
    fre = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
    fkgl = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
    return fre, fkgl


_SENT_TERMINATOR_RE = re.compile(r"[.!?]+")
MIN_SENTENCE_TERMINATORS = 2  # below this, punctuation-based sentence splitting is unreliable


def _count_sentence_terminators(text: str) -> int:
    return len(_SENT_TERMINATOR_RE.findall(text))


def _syllable_count(text: str) -> int:
    if _HAVE_TEXTSTAT:
        return textstat.syllable_count(text) or 0
    return sum(_syllables_in_word(w) for w in _tokenize_words(text))


def _line_break_flesch(text: str) -> tuple[float, float]:
    """Flesch formulas with sentence count taken from non-blank LINES
    instead of punctuation. Fixes a real breakdown found in song lyrics:
    verse/hook formatting relies on line breaks, not periods, so a
    punctuation-based sentence splitter (textstat's default, or the
    regex fallback above) treats a whole unpunctuated song as one
    giant run-on sentence, driving words-per-sentence to the hundreds
    and the formulas to implausible extremes (observed: FRE=-793.6,
    FKGL=342 on one 879-word, zero-punctuation song). Word and syllable
    counts are unchanged -- only the sentence-count denominator differs
    from the punctuation-based path."""
    lines = [l for l in text.splitlines() if l.strip()]
    n_sentences = max(len(lines), 1)
    words = _tokenize_words(text)
    n_words = max(len(words), 1)
    n_syllables = _syllable_count(text) or 1
    fre = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
    fkgl = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
    return fre, fkgl


def readability(text: str) -> dict:
    """Flesch Reading Ease + Flesch-Kincaid Grade Level, computed on the
    text exactly as given -- NEVER corrected or normalized for dialect
    spelling. A low score, a negative grade level, or a wildly implausible
    number on dialect-spelled text is the expected/reportable distortion,
    not a bug to hide.

    Sentence segmentation: if the text has fewer than
    MIN_SENTENCE_TERMINATORS (2) punctuation-based sentence breaks, the
    punctuation-based sentence count is unreliable (whole-text-as-one-
    sentence) rather than dialect-distorted, so this is a genuine
    measurement bug, not the phenomenon under study -- FRE/FKGL fall back
    to line-break segmentation instead (see _line_break_flesch). This
    fallback is NOT dialect-aware and does not touch dialect-spelled text
    that has normal punctuation; it only engages for the punctuation-free
    case (e.g. song lyrics).

    syllable_confidence is a separate, orthogonal flag: "low" when >10% of
    the text's tokens aren't recognized by a standard English dictionary
    (a proxy for "the underlying syllable count is probably unreliable
    because the tokenizer/syllable-splitter doesn't know these spellings")."""
    text = str(text) if text is not None else ""
    if not text.strip():
        return {
            "flesch_reading_ease": None, "flesch_kincaid_grade": None,
            "pct_oov_tokens": None, "syllable_confidence": None,
            "sentence_segmentation": None,
        }
    if _count_sentence_terminators(text) < MIN_SENTENCE_TERMINATORS:
        fre, fkgl = _line_break_flesch(text)
        segmentation = "line_break"
    elif _HAVE_TEXTSTAT:
        fre = textstat.flesch_reading_ease(text)
        fkgl = textstat.flesch_kincaid_grade(text)
        segmentation = "punctuation"
    else:
        fre, fkgl = _fallback_flesch(text)
        segmentation = "punctuation"
    oov = pct_oov_tokens(text)
    return {
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fkgl,
        "pct_oov_tokens": oov,
        "syllable_confidence": "low" if oov > _OOV_THRESHOLD else "ok",
        "sentence_segmentation": segmentation,
    }


# --------------------------------------------------------------------------
# 2. difflib word-level similarity
# --------------------------------------------------------------------------

def diff_summary(source_text: str, output_text: str) -> dict:
    """SequenceMatcher over whitespace-split word lists. n_replaced_words
    counts source-side words consumed by a 'replace' opcode; n_inserted /
    n_deleted are output-only / source-only words. pct_changed is the
    share of SOURCE words that were replaced, inserted-around, or deleted
    -- 0 means an exact copy, 100 means no source word survived in place."""
    src_words = str(source_text).split() if source_text else []
    out_words = str(output_text).split() if output_text else []
    sm = difflib.SequenceMatcher(None, src_words, out_words, autojunk=False)
    opcodes = sm.get_opcodes()

    n_equal = n_replaced = n_inserted = n_deleted = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            n_equal += (i2 - i1)
        elif tag == "replace":
            n_replaced += (i2 - i1)
        elif tag == "insert":
            n_inserted += (j2 - j1)
        elif tag == "delete":
            n_deleted += (i2 - i1)

    n_src = max(len(src_words), 1)
    pct_changed = 100.0 * (n_replaced + n_inserted + n_deleted) / n_src

    return {
        "ratio": sm.ratio(),
        "n_equal_words": n_equal,
        "n_replaced_words": n_replaced,
        "n_inserted": n_inserted,
        "n_deleted": n_deleted,
        "pct_changed": pct_changed,
    }


# --------------------------------------------------------------------------
# 4. Lexical flag scan (historical epithets, standard-spelled)
# --------------------------------------------------------------------------

# Seeded with terms actually found surviving "standard English" conversion
# in this project's poetry-generation runs so far (see
# reports/generation_scaffold_report.md: TEMPLATE v2 RETEST and the v3
# poetry pilot) -- NOT an exhaustive list. Configurable: pass your own
# term_list to lexical_flag_scan() to extend or replace it.
DEFAULT_LEXICAL_FLAGS = [
    "pickaninny", "pickaninnies",
    "niggah", "niggahs",
    "coon",
    "darky", "darkey", "darkies",
]


def lexical_flag_scan(text: str, row_id: str, term_list: list[str] | None = None) -> list[dict]:
    """Case-insensitive whole-word scan. Returns matches only (row_id,
    term, count, first_char_offset) -- never modifies or removes text."""
    terms = term_list if term_list is not None else DEFAULT_LEXICAL_FLAGS
    text = str(text) if text is not None else ""
    matches = []
    for term in terms:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        hits = list(pattern.finditer(text))
        if hits:
            matches.append({
                "row_id": row_id,
                "term": term,
                "count": len(hits),
                "first_char_offset": hits[0].start(),
            })
    return matches


# --------------------------------------------------------------------------
# Convenience: all metrics for one generated row
# --------------------------------------------------------------------------

def compute_row_metrics(row_id: str, mode: str, output_text: str, source_text: str | None = None) -> dict:
    """mode 'b' (and 'partial') rows get a source-vs-output diff summary;
    mode 'a' rows don't (the topic-only output has no content
    correspondence to the source poem, so a word-diff against it isn't
    meaningful -- see the poetry pilot report's mode_a section)."""
    out = readability(output_text)
    result = {
        "row_id": row_id,
        "marker_density_out": marker_density(str(output_text or "")),
        "flesch_reading_ease_out": out["flesch_reading_ease"],
        "flesch_kincaid_grade_out": out["flesch_kincaid_grade"],
        "pct_oov_tokens_out": out["pct_oov_tokens"],
        "syllable_confidence_out": out["syllable_confidence"],
        "lexical_flags_out": lexical_flag_scan(output_text, row_id),
    }
    if source_text is not None:
        src = readability(source_text)
        result.update({
            "marker_density_src": marker_density(str(source_text or "")),
            "flesch_reading_ease_src": src["flesch_reading_ease"],
            "flesch_kincaid_grade_src": src["flesch_kincaid_grade"],
            "pct_oov_tokens_src": src["pct_oov_tokens"],
            "syllable_confidence_src": src["syllable_confidence"],
            "lexical_flags_src": lexical_flag_scan(source_text, row_id),
        })
        if mode in ("b", "partial"):
            result.update(diff_summary(source_text, output_text))
    return result
