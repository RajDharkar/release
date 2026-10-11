"""
PROPOSED expansion of the grammatical-AAVE surface-pattern proxy -- NOT WIRED
IN. This module is a review artifact, produced by an audit that grounded the
current 5-feature proxy (src/generate/run_fiction_modeb_run2.py) against the
reference inventory of AAVE-attested morphosyntactic features: eWAVE (the
electronic World Atlas of Varieties of English), specifically the "Urban
African American Vernacular English" (UAAVE) variety, cross-referenced with
which of those features Multi-VALUE implements as a generation rule (see
data/raw/multi-value/resources/ewave.csv + attestation_vectors.csv +
feature_id_to_function_name.json, and reports/grammatical_proxy_ewave_coverage.csv
for the full audit table).

Finding: the current proxy's 5 surface categories map to only 7 of 141
eWAVE-UAAVE-attested features (5.0% coverage) -- see the coverage report. This
module adds detectors for 8 additional features, chosen by a STEP-3 ranking on
(a) surface/regex tractability and (b) measured corpus frequency in
data/interim/aave_book_seeds.csv (human AAVE fiction) and
data/interim/arm_d_aave_mgt_run1.csv (rule-injected AAVE-MGT) -- not by
attestation strength alone, since several attestation=1.0 (pervasive) eWAVE
features (yall/youse, finna/gon, double modals) turned out to be nearly absent
from THIS specific historical-literary seed corpus and were deprioritized as a
result (see module docstring notes on each skipped feature below).

DOES NOT MODIFY OR RE-EXPORT the existing proxy's behavior: the 5 original
features are imported UNCHANGED from run_fiction_modeb_run2.py (not
reimplemented, so there is no possibility of silent drift), and this module is
never imported by any scoring script. Nothing here has been used to re-score
run2.csv, Arm D, or any other batch.
"""
from __future__ import annotations

import re

# The 5 corrected features, imported unchanged -- see that module for their
# eWAVE mapping (habitual_be->90, zero_copula->174, ain't->155/156/157,
# negative_concord->154, completive_done->104, partial on 105/106).
from src.generate.run_fiction_modeb_run2 import (
    _habitual_be_count,
    _zero_copula_count,
    _AINT_RE,
    _multiple_negation_count,
    _done_completive_count,
)

# --------------------------------------------------------------------------
# NEW DETECTORS -- 8 features, ranked in STEP 3 of the audit by tractability
# x measured corpus frequency (counts below are from that audit, seed corpus
# n=1216 passages / Arm D corpus n=659 outputs; see reports/ for the full
# ranked table). Each is a REGEX SURFACE SCANNER with the same "not a parser"
# caveat as the original 5 -- false-positive/negative risk is stated per
# feature, not hidden.
# --------------------------------------------------------------------------

# --- 1. eWAVE 170 "Invariant present tense forms due to zero marking for
# the third person singular" (attestation 1.0, MV: uninflect). Highest
# attestation tier; measured 76 hits / 66 of 1216 seed docs.
# Pattern: 3rd-singular pronoun immediately followed by a bare common verb
# (base form, no -s) from a small closed list -- avoids the much harder
# general problem of knowing whether ANY bare verb after "he/she/it" is an
# error (that needs real subject-verb-agreement parsing).
# FALSE-NEGATIVE risk: only catches the ~15 verbs on the list, misses all
# others. FALSE-POSITIVE risk: low -- the list is deliberately restricted to
# common lexical verbs, excluding modals/existing irregulars.
_UNINFLECT_VERBS = (
    "go|want|look|seem|make|take|come|know|think|say|need|like|love|hate|got|do|have|"
    "walk|talk|work|live|stay|run|feel|see|hear|tell|ask|call|try|start|stop|keep|"
    "hold|show|turn|move|play|help|mean|carry|bring|sit|stand"
)
_UNINFLECT_RE = re.compile(rf"\b(?:he|she|it)\s+(?:{_UNINFLECT_VERBS})\b", re.IGNORECASE)


def _uninflect_count(text: str) -> int:
    return len(_UNINFLECT_RE.findall(text))


# --- 2. eWAVE 176/177/178 "Deletion of copula be: before NPs / AdjPs /
# locatives" (all attestation 1.0, MV: drop_copula_be_NP/AP/locative). This
# is the single most explicitly named gap from the hand-check ("predicate-
# adjective zero-copula like 'she nice' is not attempted"). Measured 117
# combined hits across ~115 of 1216 seed docs (18 AdjP + 62 NP + 37 locative).
# Three separate sub-patterns, same spirit as the existing (corrected)
# progressive zero-copula detector: pronoun directly followed by a predicate
# with no intervening copula.
#   AdjP: pronoun + adjective from a small closed list (avoids the
#     false-positive risk of an open adjective class -- "she nice" is
#     unambiguous, an open-class attempt would over-fire on ordinary
#     adjective-final clauses).
#   NP: pronoun + "a"/"an" + noun (e.g. "he a teacher"). FALSE-POSITIVE risk:
#     LOW-MODERATE -- "a"/"an" essentially never directly follows a subject
#     pronoun in standard English, so this is fairly clean, but does not
#     catch bare-noun predicates without an article ("he doctor now").
#   Locative: pronoun + a small closed set of locative prepositions + "the"
#     (e.g. "they in the house"). FALSE-POSITIVE risk: LOW, but narrow --
#     misses locatives without "the" ("she in trouble").
_DROP_COPULA_AP_ADJ = (
    "nice|tired|hungry|happy|sad|mad|cold|hot|sick|scared|ready|busy|crazy|fine|"
    "okay|ok|good|bad|wrong|right|late|early|strong|weak|tall|short|young|old"
)
_DROP_COPULA_AP_RE = re.compile(rf"\b(?:I|you|he|she|it|we|they)\s+(?:{_DROP_COPULA_AP_ADJ})\b", re.IGNORECASE)
_DROP_COPULA_NP_RE = re.compile(r"\b(?:he|she|it|I|you|we|they)\s+(?:a|an)\s+\w+", re.IGNORECASE)
_DROP_COPULA_LOC_RE = re.compile(r"\b(?:he|she|it|I|you|we|they)\s+(?:in|on|at|over|under|by|near)\s+the\b", re.IGNORECASE)


def _drop_copula_count(text: str) -> int:
    return (
        len(_DROP_COPULA_AP_RE.findall(text))
        + len(_DROP_COPULA_NP_RE.findall(text))
        + len(_DROP_COPULA_LOC_RE.findall(text))
    )


# --- 3. eWAVE 128 "Levelling of past tense/past participle verb forms:
# regularization of irregular verb paradigms" (attestation 0.6, MV:
# regularized_past_tense) -- one of the hand-check's 3 named misses. Measured
# 85 hits / 71 of 1216 seed docs. Closed list of common "wrongly regularized"
# irregular-verb forms (knowed, throwed, seed [=saw], growed, etc.) --
# genuinely tractable ONLY as a closed list, since detecting this generally
# requires knowing a verb is irregular AND that -ed was applied anyway (that
# needs a lexicon + morphological analysis, not a general regex).
# FALSE-POSITIVE risk: near-zero -- every form on the list is a genuine
# nonstandard regularization, not a real English word with another meaning.
# FALSE-NEGATIVE risk: HIGH -- only the ~60 forms listed are caught; any
# other irregular verb regularized in a way not on this list is missed.
_REGULARIZED_PAST_RE = re.compile(
    r"\b(?:knowed|throwed|seed|growed|blowed|drawed|catched|teached|bringed|buyed|"
    r"thinked|standed|runned|singed|ringed|drinked|sweared|tored|worned|choosed|"
    r"freezed|wroted|comed|beated|breaked|builded|feeled|finded|flied|forgetted|"
    r"getted|goed|hurted|keeped|leaved|losed|maked|meaned|payed|readed|rided|"
    r"rised|selled|sended|shooted|showed|sleeped|speaked|spended|stealed|"
    r"swimmed|taked|telled|winned)\b",
    re.IGNORECASE,
)


def _regularized_past_count(text: str) -> int:
    return len(_REGULARIZED_PAST_RE.findall(text))


# --- 4. eWAVE 180 "Was/were generalization" (attestation 0.6, MV: were_was)
# -- the second of the hand-check's 3 named misses ("the thing were").
# Measured 71 hits / 67 of 1216 seed docs. Pattern: singular subject + "were"
# OR plural subject + "was" -- a coarse subject-number mismatch, using only
# pronouns/demonstratives as the subject (not general NPs, which would need
# real number agreement to check reliably).
# FALSE-POSITIVE risk: LOW for the singular+"were" side (subjunctive "if I
# were" is the main confound -- genuinely ambiguous with this construction
# without deeper context, since eWAVE's own "was for conditional were" is a
# DIFFERENT feature (147) describing the opposite direction). FALSE-POSITIVE
# risk MODERATE for plural+"was" ("you was" is clean, but dialectal "youse
# was"/"we was" could also be colloquial non-AAVE registers).
_WAS_WERE_LEVELING_RE = re.compile(
    r"\b(?:it|he|she|this|that|i)\s+were\b|\b(?:we|you|they|these|those)\s+was\b",
    re.IGNORECASE,
)


def _was_were_leveling_count(text: str) -> int:
    return len(_WAS_WERE_LEVELING_RE.findall(text))


# --- 5. eWAVE 11 "Regularized reflexives paradigm" (attestation 1.0, MV:
# regularized_reflexives) -- hisself/theirself/theirselves instead of
# himself/themselves. Measured only 4 hits / 4 of 1216 seed docs (low
# frequency in THIS corpus), but included anyway: closed word list, near-zero
# false-positive risk, and cheap to score regardless of base rate.
_REGULARIZED_REFLEXIVE_RE = re.compile(r"\b(?:hisself|theirself|theirselves|ourself|meself)\b", re.IGNORECASE)


def _regularized_reflexive_count(text: str) -> int:
    return len(_REGULARIZED_REFLEXIVE_RE.findall(text))


# --- 6. eWAVE 158 "Invariant don't for all persons in the present tense"
# (attestation 1.0, MV: dont) -- "he/she/it don't" instead of "doesn't".
# Measured only 8 hits / 8 of 1216 seed docs, but included: highest
# attestation tier, clean closed pattern, near-zero false-positive risk (a
# 3rd-singular pronoun is never followed by "don't" in standard English).
_INVARIANT_DONT_RE = re.compile(r"\b(?:he|she|it)\s+don'?t\b", re.IGNORECASE)


def _invariant_dont_count(text: str) -> int:
    return len(_INVARIANT_DONT_RE.findall(text))


# --- 7. eWAVE 121 "Double modals" (attestation 0.6, MV: double_modals) --
# "might could", "might would", etc. -- the THIRD of the hand-check's 3
# named misses. Measured 0 hits in the 1216-passage seed corpus (this
# specific historical/literary fiction register apparently doesn't use them,
# or at least not in a form this pattern catches) but 579 hits / 194 of 659
# Arm D outputs (Multi-VALUE generates them constantly). Included despite
# zero seed-corpus support because: (a) it is one of the explicitly named
# gaps this audit was commissioned to check, (b) it will matter for scoring
# Arm D output even if it can't discriminate within the human seed corpus,
# and (c) it may occur in other, non-literary AAVE corpora even where it
# doesn't here -- zero support in ONE corpus is not evidence the feature
# proxy is worthless, just that this corpus's register doesn't exhibit it.
_DOUBLE_MODAL_RE = re.compile(
    r"\b(?:might|may|must|could|would|should|used to)\s+(?:could|would|should|can|will|oughta|ought to)\b",
    re.IGNORECASE,
)


def _double_modal_count(text: str) -> int:
    return len(_DOUBLE_MODAL_RE.findall(text))


# --- 8. eWAVE 68 "Them instead of demonstrative those" (attestation 1.0,
# MV: those_them) -- "them books" instead of "those books". Measured the
# LARGEST raw count of any candidate (527 hits / 369 of 1216 seed docs), but
# flagged with the audit's biggest known false-positive risk: the naive
# pattern "them" + next word cannot distinguish the demonstrative-determiner
# use ("them books" = "those books") from ordinary standard object-pronoun
# use immediately followed by an unrelated noun phrase starting a new
# constituent (e.g. ditransitive "gave them books" = "gave [to] them
# [some] books", where "them" is the indirect-object pronoun, not a
# determiner on "books"). This proposal does NOT attempt that
# disambiguation (it would need real dependency parsing to know whether
# "them" and the following noun are in the same NP) -- it is flagged here
# as the clearest case in this whole shortlist where the tractable regex
# and the correct answer diverge, and is the one candidate I would NOT trust
# without a stronger caveat or a verb-preceding exclusion list (give/tell/
# show/send/bring + them + NP as a ditransitive-frame exclusion, untested
# here).
_THEM_DEMONSTRATIVE_RE = re.compile(r"\bthem\s+\w+", re.IGNORECASE)


def _them_demonstrative_count(text: str) -> int:
    return len(_THEM_DEMONSTRATIVE_RE.findall(text))


# --------------------------------------------------------------------------
# Deliberately NOT added, despite being tractable in principle -- corpus
# frequency in THIS project's specific data did not support inclusion (see
# STEP 3 of the audit for the counts):
#   - eWAVE 34 "yall/youse" (2nd person plural other than 'you'): 8 hits /
#     1 of 1216 seed docs even after broadening the pattern to include
#     "youse" (which does occur once, heavily, in one passage). Near-absent
#     from this historical-literary corpus.
#   - eWAVE 114/126 "gon/gonna/finna/fixin' to" (future markers): 0 hits in
#     the seed corpus. Same likely explanation -- these read as more
#     contemporary AAVE forms than this ~1900-1930s literary fiction uses.
#   - eWAVE 7 "me instead of I in coordinate subjects" ("me and him went"):
#     3 hits / 2 docs. Tractable but too rare here to be worth the
#     complexity of a subject-position heuristic.
# All three remain real, well-attested eWAVE/UAAVE features -- they are
# absent from this proposal only because this specific corpus doesn't
# exhibit them, not because they are illegitimate. A different corpus
# (contemporary AAVE, e.g.) would likely support adding all three.
# --------------------------------------------------------------------------


def grammatical_aave_proxy_v3_breakdown(text: str) -> dict:
    """Per-feature hit counts -- for review, not scoring. Keys named by
    eWAVE feature id(s) they operationalize."""
    text = str(text or "")
    return {
        "habitual_be_90": _habitual_be_count(text),
        "zero_copula_progressive_174": _zero_copula_count(text),
        "aint_155_156_157": len(_AINT_RE.findall(text)),
        "negative_concord_154": _multiple_negation_count(text),
        "completive_done_104": _done_completive_count(text),
        "uninflect_3sg_170": _uninflect_count(text),
        "drop_copula_NP_AP_loc_176_177_178": _drop_copula_count(text),
        "regularized_past_128": _regularized_past_count(text),
        "was_were_leveling_180": _was_were_leveling_count(text),
        "regularized_reflexive_11": _regularized_reflexive_count(text),
        "invariant_dont_158": _invariant_dont_count(text),
        "double_modal_121": _double_modal_count(text),
        "them_demonstrative_68": _them_demonstrative_count(text),
    }


def grammatical_aave_proxy_v3(text: str) -> float:
    """PROPOSED replacement density: total hits (5 original + 8 new
    features) / word count, same units as the original grammatical_aave_proxy
    and as marker_density. NOT wired into any scoring script -- exists so a
    reviewer can compare v2 (5 features) vs. this v3 draft (13 features) on
    the same text before deciding whether/how to adopt it."""
    text = str(text or "")
    words = text.split()
    if not words:
        return 0.0
    hits = sum(grammatical_aave_proxy_v3_breakdown(text).values())
    return hits / len(words)
