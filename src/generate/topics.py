"""
Topic/theme extraction for mode_a prompts (Part 2 of the generation
scaffold).

extract_topic_llm() is the SAFE, LIVE-GENERATION-APPROVED extractor: it
prompts a resolved model to name a passage's subject in a short neutral
standard-English phrase, explicitly forbidding reproduction of dialect
spelling/slang/profanity from the source, and verifies the model's own
output against this module's extractive-path filters (retry-then-flag on
a dirty result, never a silent pass-through). Use this for any live
mode_a call.

The original purely-extractive functions (extract_topic,
extract_theme_phrase / extract_theme_phrase_with_flag) remain for
non-live, offline preview use (extract_topics_for_dataframe, the
__main__ demo block below, run_dryrun.py's DryRunClient-only preview) --
extract_theme_phrase_with_flag() already has adequate filtering
(_clean_theme_lemma: profanity blocklist + dialect-marker check +
dictionary check) and is unaffected by the bug below.

DISABLED: extract_topic_standard_english() and
extract_topic_for_live_mode_a() are confirmed broken and now raise
RuntimeError on call rather than silently leaking. Diagnosis (see
docs/decisions.md): extract_topic_for_live_mode_a() regenerated via
extract_topic_standard_english() only when it detected a token outside
the macOS system dictionary -- but that dictionary (Webster's Second
International, 1934) is old and unusually permissive, so AAVE eye-dialect
spellings (de, dey, den, thoo, boun, bes, lak, chile, bah, ...) pass as
legitimate archaic/alternate-sense headwords, AND profanity/slurs (hell,
damn, nigger, ...) are themselves ordinary dictionary words that never
even trigger the "regenerate" check. Confirmed empirically: 8/20 (40%)
dialect-leak rate on a dialect-heavy sample, plus 2/6 direct profanity
leaks with zero fallback trigger, via the live path exactly as
run_live_smoke.py called it.

Determinism (extractive path only): the ranking is a pure function of the
text (frequency count, then alphabetical tie-break) -- no randomness is
involved, so no seed is needed for extraction itself. compute_seed() is
provided anyway for any downstream caller that does need a reproducible,
order-independent per-row seed (same convention as Arm C: sha256(row_id),
never builtin hash()), and extract_topic_llm() uses it implicitly via
row_id-derived request labeling.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import spacy

if TYPE_CHECKING:
    from .config import ModelConfig
    from .harness import GenerationClient

log = logging.getLogger("generate.topics")

_NLP = None  # lazy-loaded singleton; avoid reloading the model per call

MAX_TERMS = 5
MAX_TOKENS_PER_TERM = 3  # cap phrase length so a term can't itself become a long verbatim span
LEAKAGE_NGRAM = 4

STOP_EXTRA = {"thing", "things", "way", "lot"}  # generic nouns that make poor topics


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner"])
    return _NLP


def compute_seed(row_id: str) -> int:
    return int(hashlib.sha256(str(row_id).encode()).hexdigest()[:8], 16) or 1


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_leakage(source_text: str, topic: str, n: int = LEAKAGE_NGRAM) -> bool:
    """True if any n-gram of `topic` appears verbatim in `source_text`."""
    src_ngrams = _word_ngrams(source_text, n)
    if not src_ngrams:
        return False
    topic_ngrams = _word_ngrams(topic, n)
    return bool(src_ngrams & topic_ngrams)


def extract_topic(text: str, max_terms: int = MAX_TERMS) -> str:
    """Extractive topic string: top noun chunks by frequency, each capped
    to MAX_TOKENS_PER_TERM tokens, joined into a short comma-separated
    string. Deterministic: ties broken alphabetically, not by encounter
    order, so re-running on the same text always yields the same string
    regardless of any prior process state."""
    doc = _nlp()(str(text))
    counts: Counter[str] = Counter()
    for chunk in doc.noun_chunks:
        tokens = [t.lemma_.lower() for t in chunk if not t.is_stop and t.is_alpha]
        tokens = tokens[:MAX_TOKENS_PER_TERM]
        if not tokens:
            continue
        phrase = " ".join(tokens)
        if phrase in STOP_EXTRA or len(phrase) < 3:
            continue
        counts[phrase] += 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    terms = [phrase for phrase, _ in ranked[:max_terms]]

    if not terms:
        # Fallback: most frequent content words (adjective/noun/proper noun).
        word_counts: Counter[str] = Counter(
            t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and t.pos_ in ("NOUN", "PROPN", "ADJ")
        )
        ranked_words = sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        terms = [w for w, _ in ranked_words[:max_terms]]

    topic = ", ".join(terms) if terms else "(no salient terms extracted)"

    if check_leakage(text, topic):
        log.warning("LEAKAGE: topic contains a %d-gram verbatim from source text: %r", LEAKAGE_NGRAM, topic)

    return topic


_ENGLISH_WORDS: set[str] | None = None
DICT_PATH = Path("/usr/share/dict/words")


def _english_words() -> set[str]:
    """macOS's standard system dictionary (235,976 words) -- used as a
    plain "is this a standard-English word" check for the JOB 3 step 7
    dialect-token filter. No new dependency: this ships with the OS."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        if DICT_PATH.exists():
            _ENGLISH_WORDS = {
                w.strip().lower() for w in DICT_PATH.read_text(errors="ignore").splitlines() if w.strip()
            }
        else:
            log.warning("System dictionary not found at %s -- dialect-token filter disabled (treats everything as standard).", DICT_PATH)
            _ENGLISH_WORDS = set()
    return _ENGLISH_WORDS


def _is_dictionary_word(word: str) -> bool:
    return word.lower() in _english_words()


def topic_nondictionary_tokens(topic: str) -> list[str]:
    """Words in `topic` that are not in the standard-English dictionary --
    a proxy for dialect-spelled tokens (e.g. `dat`, `de`, `gwine`, `ouah`)
    leaking into a mode_a prompt from an AAVE-spelled source."""
    words = re.findall(r"[a-zA-Z']+", topic)
    return [w for w in words if w and not _is_dictionary_word(w)]


_STANDARD_ENGLISH_FALLBACK_DISABLED_MSG = (
    "extract_topic_standard_english() is DISABLED -- confirmed broken (see "
    "docs/decisions.md and the topics.py module docstring). Its only filter "
    "is macOS system-dictionary membership, which is NOT sufficient: many "
    "AAVE eye-dialect spellings (de, dey, den, thoo, boun, bes, lak, chile, "
    "bah, ...) are legitimate archaic/alternate-sense headwords in that "
    "dictionary and pass straight through, and profanity/slurs (hell, damn, "
    "nigger, ...) are also ordinary dictionary words with zero blocklist "
    "check on this path. Confirmed empirically: 8/20 (40%) leak rate on a "
    "dialect-heavy sample via extract_topic_for_live_mode_a(), which calls "
    "this function as its fallback. Use extract_topic_llm() instead."
)


def extract_topic_standard_english(text: str, max_terms: int = MAX_TERMS) -> str:
    """DISABLED. Kept only so the function signature/name remains
    discoverable in a diff/blame; calling it is a hard error. See
    _STANDARD_ENGLISH_FALLBACK_DISABLED_MSG for why, and extract_topic_llm()
    for the replacement."""
    raise RuntimeError(_STANDARD_ENGLISH_FALLBACK_DISABLED_MSG)


# Explicit blocklist for the poetry-pilot theme extractor: some slurs and
# profanity ARE valid entries in a standard system dictionary (curse words
# are still "words"), so the dictionary-membership check alone does not
# reject them -- this list catches what that check structurally cannot.
PROFANITY_SLUR_BLOCKLIST = {
    "nigger", "nigga", "niggah", "coon", "darkey", "darky", "pickaninny",
    "fuck", "fucker", "fucking", "motherfucker", "shit", "bitch", "cunt",
    "ass", "asshole", "damn", "goddamn", "hell", "whore", "slut", "cock",
    "dick", "pussy",
}


# The standard-English dictionary check alone is NOT sufficient: macOS's
# system word list (web2, Webster's Second International, 1934) is old
# and unusually comprehensive, and includes several AAVE eye-dialect
# spellings as legitimate archaic/rare headwords -- "dey" (a historical
# Ottoman governor's title), "dar", "mah", "roun", "thoo" all pass
# `_is_dictionary_word` cleanly. Caught by manually eyeballing the first
# 30-sample printout (exactly what that step is for), not by the
# automated flag count, which was a false 0%. Fixed by additionally
# excluding this project's own established dialect-marker list
# (source6_pd_poetry.MARKER_RE) plus two more terms ("thoo", "roun")
# observed slipping through that aren't in that list.
# Apostrophe-stripping (below) can turn a genuine contraction marker into
# an ordinary standard-English word that happens to share the stripped
# spelling -- confirmed empirically via extract_topic_llm()'s own
# verification layer: "an'" (dialect contraction of "and") strips to "an",
# the standard indefinite article, causing every ordinary use of "an" to
# false-flag as a dialect marker (LLM topic "A man discovering an injured
# woman" was wrongly flagged over the word "an"). "chile" is excluded
# pre-emptively for the same reason (collides with the country name Chile
# and the pepper), even though not observed to false-fire in this
# session's verification sample -- both are foreseeable, not hypothetical.
_MARKER_STRIP_COLLISIONS = {"an", "chile"}


def _dialect_marker_words() -> set[str]:
    from src.loaders.source6_pd_poetry import MARKER_RE
    # MARKER_RE's pattern is `\b(term|term|...)\b`; pull the alternatives
    # back out rather than re-typing the list a third place in the repo.
    inner = MARKER_RE.pattern
    alts = inner[inner.index("(") + 1: inner.rindex(")")].split("|")
    stripped = {a.strip("'") for a in alts}
    return stripped - _MARKER_STRIP_COLLISIONS


_EXTRA_DIALECT_WORDS = {
    "thoo", "roun", "huh", "den", "boun", "dah", "git", "lak", "col", "bah",
    "min", "bout", "fum", "ole", "bes", "flo", "bus", "stan", "sah", "cum",
    "spec", "bile", "sot", "dun", "wes", "cep",
}
# All of the above were found by directly scanning this pilot's own
# topics.csv output for short residual tokens after the first two filter
# passes -- some (huh, den, bile) are themselves valid standard-English
# words in a different sense (interjection "huh?", noun "den" as in
# animal lair, verb "bile" is not standard actually but "boil" is) that
# happen to double as common AAVE eye-dialect spellings of a different
# word ("her", "then", "boil") in this corpus's specific usage. Pure
# lexical filtering cannot fully disambiguate that without a semantic/
# contextual model, which is out of scope for a purely-extractive, no-LLM
# extractor -- this list closes the gap empirically rather than perfectly.
_DIALECT_MARKER_WORDS: set[str] | None = None


def _is_dialect_marker(word: str) -> bool:
    global _DIALECT_MARKER_WORDS
    if _DIALECT_MARKER_WORDS is None:
        _DIALECT_MARKER_WORDS = _dialect_marker_words() | _EXTRA_DIALECT_WORDS
    return word.lower().strip("'") in _DIALECT_MARKER_WORDS


def _clean_theme_lemma(token) -> str | None:
    """Return a lowercase lemma usable in a theme phrase, or None to
    reject it. Rejects stopwords, non-alphabetic tokens, proper nouns
    (character/place-name fragments), profanity/slurs (explicit
    blocklist), this project's known AAVE eye-dialect markers, any token
    still not a standard dictionary word, and anything under 3
    characters."""
    if token.is_stop or not token.is_alpha:
        return None
    if token.pos_ == "PROPN":
        return None
    lemma = token.lemma_.lower()
    if lemma in PROFANITY_SLUR_BLOCKLIST:
        return None
    if _is_dialect_marker(lemma):
        return None
    if not _is_dictionary_word(lemma):
        return None
    if len(lemma) < 3:
        return None
    return lemma


def extract_theme_phrase(text: str, max_terms: int = 4) -> str:
    """Short, readable, standard-English theme phrase for mode_a prompts.
    Unlike extract_topic()/extract_topic_standard_english(), which return
    a flat comma-joined keyword list, this joins the top filtered terms
    as a natural short phrase ("X, Y, and Z") and additionally rejects
    profanity/slurs and proper-noun fragments -- see _clean_theme_lemma().

    Still purely extractive (no LLM call): frequency-ranked, alphabetical
    tie-break, deterministic. Given the "no LLM calls" constraint, this
    cannot produce a grammatically generated sentence -- "readable theme
    phrase" here means a short, natural-sounding coordination of 1-4
    cleaned content words/short noun-chunk fragments, not free-form NLG.
    """
    doc = _nlp()(str(text))
    counts: Counter[str] = Counter()

    for chunk in doc.noun_chunks:
        words = [w for w in (_clean_theme_lemma(t) for t in chunk) if w]
        words = words[:2]  # keep each term to 1-2 words so the phrase stays short
        if not words:
            continue
        phrase = " ".join(words)
        if phrase in STOP_EXTRA or len(phrase) < 3:
            continue
        counts[phrase] += 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    terms = [phrase for phrase, _ in ranked[:max_terms]]

    if not terms:
        word_counts: Counter[str] = Counter()
        for t in doc:
            if t.pos_ in ("NOUN", "ADJ"):
                w = _clean_theme_lemma(t)
                if w:
                    word_counts[w] += 1
        ranked_words = sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        terms = [w for w, _ in ranked_words[:max_terms]]

    if not terms:
        return "(no salient standard-English theme extracted)"
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


def extract_theme_phrase_with_flag(text: str, max_terms: int = 4) -> tuple[str, bool, list[str]]:
    """Wraps extract_theme_phrase() with the pilot's flagging logic.
    Flags a row if: the extractor fell through to the empty/garbage
    fallback, the 4-gram leakage check trips, or (defensive re-check,
    should be structurally impossible given the filtering above) the
    resulting phrase still contains a non-dictionary or blocklisted
    word. Returns (topic, flagged, reasons)."""
    topic = extract_theme_phrase(text, max_terms=max_terms)
    reasons = []

    if topic == "(no salient standard-English theme extracted)":
        reasons.append("empty_extraction")
    if check_leakage(text, topic):
        reasons.append("4gram_leakage")
    residual_bad = [
        w for w in re.findall(r"[a-zA-Z']+", topic)
        if w.lower() in PROFANITY_SLUR_BLOCKLIST or _is_dialect_marker(w) or not _is_dictionary_word(w)
    ]
    if residual_bad:
        reasons.append(f"residual_bad_tokens:{residual_bad}")

    return topic, bool(reasons), reasons


_LIVE_MODE_A_DISABLED_MSG = (
    "extract_topic_for_live_mode_a() is DISABLED -- confirmed broken (see "
    "docs/decisions.md and the topics.py module docstring). It regenerated "
    "via extract_topic_standard_english() when it detected a non-dictionary "
    "token, but that fallback's only filter is dictionary membership, which "
    "misses eye-dialect words that happen to be legitimate archaic dictionary "
    "entries, AND it never regenerated at all for profanity/slurs that are "
    "themselves ordinary dictionary words (e.g. 'nigger', 'hell', 'damn'), "
    "since those never register as 'non-dictionary' in the first place. "
    "Confirmed empirically: 8/20 (40%) dialect-leak rate on a dialect-heavy "
    "sample, plus 2/6 direct profanity leaks with zero fallback trigger. Use "
    "extract_topic_llm() instead for any live mode_a generation."
)


def extract_topic_for_live_mode_a(text: str) -> tuple[str, bool, list[str]]:
    """DISABLED. Kept only so the function name/signature remains
    discoverable in a diff/blame; calling it is a hard error. See
    _LIVE_MODE_A_DISABLED_MSG for why, and extract_topic_llm() for the
    replacement actually safe for live generation."""
    raise RuntimeError(_LIVE_MODE_A_DISABLED_MSG)


TOPIC_EXTRACTION_PROMPT = (
    "Read the following passage and identify its general subject in 3-6 words, "
    "as a short neutral topic phrase (not a sentence, not a summary, not a rewrite).\n\n"
    "Rules:\n"
    "- Write the topic phrase in plain standard English only.\n"
    "- Do not copy any dialect-spelled words, slang, or nonstandard spellings from "
    "the passage, even if they appear in it -- always use standard spelling.\n"
    "- Do not include profanity, slurs, or offensive language, even if present in "
    "the passage -- describe the subject in neutral, general terms instead.\n"
    "- Do not quote or copy phrases verbatim from the passage.\n"
    "- Output ONLY the topic phrase, nothing else -- no preamble, no quotation marks, "
    "no explanation.\n\n"
    "--- PASSAGE ---\n{text}\n--- END PASSAGE ---\n\n"
    "Topic phrase:"
)

TOPIC_RETRY_SUFFIX = (
    "\n\n(Your previous answer contained a nonstandard-dialect word, slang, or "
    "profane/offensive content. Try again: use ONLY plain standard English words, "
    "and do not include any offensive language.)"
)

MAX_TOPIC_LLM_RETRIES = 1  # one retry on a dirty first attempt, then flag rather than pass through


def _topic_leak_check(text: str, topic: str) -> dict:
    """Post-hoc verification of an LLM-produced topic, reusing this
    module's existing extractive-path filters as a check layer rather than
    trusting the model's instruction-following blindly."""
    words = re.findall(r"[a-zA-Z']+", topic)
    bad_dialect = [w for w in words if _is_dialect_marker(w)]
    bad_profanity = [w for w in words if w.lower() in PROFANITY_SLUR_BLOCKLIST]
    leaked_ngram = check_leakage(text, topic)
    return {
        "bad_dialect_tokens": bad_dialect,
        "bad_profanity_tokens": bad_profanity,
        "ngram_leakage": leaked_ngram,
        "clean": not (bad_dialect or bad_profanity or leaked_ngram),
    }


def extract_topic_llm(
    text: str,
    client: "GenerationClient",
    model_cfg: "ModelConfig",
    *,
    row_id: str,
    max_retries: int = MAX_TOPIC_LLM_RETRIES,
) -> tuple[str, dict]:
    """LLM-based topic extraction -- the safe replacement for the disabled
    extract_topic_for_live_mode_a(). Prompts a resolved model to name the
    passage's general subject in a short neutral phrase, explicitly
    forbidding reproduction of dialect spelling, slang, or profanity from
    the source. Deterministic-friendly: caller should pass a
    low-temperature model_cfg; row_id is threaded through to the client for
    per-row provenance the same way every other harness caller does.

    An LLM instruction is not a guarantee, so the output is checked with
    this module's own extractive-path filters (_topic_leak_check) rather
    than trusted blindly. RETRY-THEN-FLAG, not pass-through: if the first
    attempt is dirty, one retry is sent with an explicit correction
    reminder appended to the prompt; if the retry is ALSO dirty, the topic
    is still returned (a caller may want it for inspection/logging) but
    diagnostics["flagged"] is True and diagnostics["clean"] is False --
    callers MUST check diagnostics["flagged"] before using the topic in a
    live generation prompt, since a flagged topic is not safe to use as-is.
    """
    diagnostics: dict = {}
    topic = ""
    for attempt in range(max_retries + 1):
        prompt = TOPIC_EXTRACTION_PROMPT.format(text=text)
        if attempt > 0:
            prompt += TOPIC_RETRY_SUFFIX
        result = client.generate(prompt, model_cfg, row_id=f"{row_id}_attempt{attempt}", mode="topic_llm")
        topic = result.text.strip().strip('"').strip("'")

        check = _topic_leak_check(text, topic)
        diagnostics = {
            "finish_reason": result.finish_reason,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "attempts": attempt + 1,
            **check,
        }
        if check["clean"]:
            diagnostics["flagged"] = False
            return topic, diagnostics

    # Exhausted retries and still dirty -- flag hard rather than silently
    # returning a leaked topic as if it were normal, usable output.
    diagnostics["flagged"] = True
    log.warning(
        "TOPIC EXTRACTION FLAGGED after %d attempt(s), row_id=%s: topic=%r diagnostics=%s",
        diagnostics["attempts"], row_id, topic, diagnostics,
    )
    return topic, diagnostics


def extract_topics_for_dataframe(df, text_col: str = "text", id_col: str = "id") -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        topic = extract_topic(text)
        leaked = check_leakage(text, topic)
        rows.append({"id": row[id_col], "topic": topic, "leakage_flag": leaked})
    return rows


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    REPO_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED = REPO_ROOT / "data/processed"
    INTERIM = REPO_ROOT / "data/interim"
    INTERIM.mkdir(parents=True, exist_ok=True)
    OUT_PATH = INTERIM / "topics_preview.csv"

    SAMPLE_SEED = 4242
    SAMPLE_N = 200

    all_rows: list[dict] = []
    blocked: list[str] = []

    # AAVE poetry seed pool -- fully available, no blockers.
    poetry_df = pd.read_csv(PROCESSED / "aave_poetry.csv")
    log.info("aave_poetry.csv: %d rows (spec said 176; actual pool is %d)", len(poetry_df), len(poetry_df))
    for rec in extract_topics_for_dataframe(poetry_df):
        rec["source"] = "aave_poetry"
        all_rows.append(rec)

    # 200-row samples from each SAE source (LiteraryTaste, HAP-E via
    # unified.csv; MillerCenter via sae_oratory.csv). unified.csv is
    # currently missing from data/processed/ (see prior audit turn) --
    # LiteraryTaste and HAP-E are blocked; MillerCenter proceeds.
    unified_path = PROCESSED / "unified.csv"
    oratory_path = PROCESSED / "sae_oratory.csv"

    if unified_path.exists():
        unified = pd.read_csv(unified_path, low_memory=False)
        lt = unified[(unified["source_dataset"] == "mj-storytelling/LiteraryTaste") & (unified["label"] == "human")]
        hape = unified[
            (unified["source_dataset"] == "browndw/human-ai-parallel-corpus")
            & (unified["label"] == "human")
            & (~unified["source_subset"].fillna("").str.contains("chunk_1"))
        ]
        for name, pool in [("LiteraryTaste", lt), ("HAP-E", hape)]:
            sample = pool.sample(n=min(SAMPLE_N, len(pool)), random_state=SAMPLE_SEED)
            for rec in extract_topics_for_dataframe(sample):
                rec["source"] = name
                all_rows.append(rec)
    else:
        blocked.extend(["LiteraryTaste", "HAP-E"])
        log.warning("BLOCKED: data/processed/unified.csv is missing -- "
                    "LiteraryTaste and HAP-E 200-row topic samples skipped.")

    if oratory_path.exists():
        oratory = pd.read_csv(oratory_path, low_memory=False)
        mc = oratory[oratory["needs_review"] == False]  # noqa: E712
        sample = mc.sample(n=min(SAMPLE_N, len(mc)), random_state=SAMPLE_SEED)
        for rec in extract_topics_for_dataframe(sample):
            rec["source"] = "MillerCenter"
            all_rows.append(rec)
    else:
        blocked.append("MillerCenter")
        log.warning("BLOCKED: data/processed/sae_oratory.csv is missing -- MillerCenter sample skipped.")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUT_PATH, index=False)
    log.info("Wrote %d topic rows -> %s", len(out_df), OUT_PATH)
    if blocked:
        log.warning("BLOCKED sources (missing input file): %s", blocked)

    n_leak = int(out_df["leakage_flag"].sum())
    log.info("Leakage check: %d / %d rows flagged (4-gram overlap between topic and source)", n_leak, len(out_df))

    print("\n-- 15 sample topic lines --")
    for _, r in out_df.sample(n=min(15, len(out_df)), random_state=SAMPLE_SEED).iterrows():
        print(f"  [{r['source']}] {r['id']}: {r['topic']}")
