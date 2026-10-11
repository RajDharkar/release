"""
FINAL adopted grammatical-AAVE proxy -- the only proxy version wired into
scoring. Built from the hand-check verdicts on src/analysis/grammatical_proxy_v3.py
(a proposal module, still unwired, left as-is). This module makes NO regex-logic
changes anywhere: the 5 v2 features are imported byte-unchanged from
src/generate/run_fiction_modeb_run2.py, and the 3 new adopted features are
either imported unchanged (regularized_reflexive, invariant_dont) or built from
a verbatim sub-expression / a closed-wordlist deletion of v3's own patterns
(regularized_past, double_modal, was_were_leveling) -- never new regex.

ADOPTED (8 features, scored):
  1. habitual_be              -- eWAVE 90  (v2, unchanged)
  2. zero_copula (progressive)-- eWAVE 174 (v2, unchanged)
  3. ain't                    -- eWAVE 155/156/157 (v2, unchanged, undifferentiated)
  4. negative_concord         -- eWAVE 154 (v2, unchanged)
  5. completive_done          -- eWAVE 104 (v2, unchanged; 105/106 partial only)
  6. regularized_reflexive    -- eWAVE 11  (v3, unchanged, hand-check: 0% suspect)
  7. invariant_dont           -- eWAVE 158 (v3, unchanged, hand-check: 0% suspect)
  8. regularized_past         -- eWAVE 128 (v3 wordlist MINUS "showed")
  9. double_modal             -- eWAVE 121 (v3 modal-pair list MINUS *-plus-"will" pairs)
 10. was_were_leveling        -- eWAVE 180, PLURAL-WAS HALF ONLY (verbatim
                                  sub-expression from v3's existing alternation;
                                  the singular-"were" half is dropped entirely,
                                  not merely deprioritized)

HELD, NOT SCORED (defined nowhere in this module -- inert by omission, not by
a broken/neutered copy): uninflect_3sg (170), drop_copula_AP/NP/loc
(176/177/178). Their logic in grammatical_proxy_v3.py is untouched and this
module does not import or call them. them-demonstrative (68) was never
adopted and is likewise absent.

Coverage: 10 of 141 eWAVE-attested UAAVE features (7.1%) -- see
reports/proxy_final_coverage.md for the full accounting, including why each
HOLD feature was held and which features are permanently out of surface-regex
scope regardless of adoption decisions.
"""
from __future__ import annotations

import re

# The 5 v2 features -- imported unchanged, not reimplemented.
from src.generate.run_fiction_modeb_run2 import (
    _habitual_be_count,
    _zero_copula_count,
    _AINT_RE,
    _multiple_negation_count,
    _done_completive_count,
)

# The 2 clean-as-is v3 additions -- imported unchanged.
from src.analysis.grammatical_proxy_v3 import (
    _regularized_reflexive_count,
    _invariant_dont_count,
)

# --------------------------------------------------------------------------
# eWAVE 128 "Levelling of past tense/past participle verb forms:
# regularization of irregular verb paradigms" -- WORDLIST EDIT #1.
# BEFORE (src/analysis/grammatical_proxy_v3.py._REGULARIZED_PAST_RE),
# verbatim, for the record:
#   knowed|throwed|seed|growed|blowed|drawed|catched|teached|bringed|buyed|
#   thinked|standed|runned|singed|ringed|drinked|sweared|tored|worned|choosed|
#   freezed|wroted|comed|beated|breaked|builded|feeled|finded|flied|forgetted|
#   getted|goed|hurted|keeped|leaved|losed|maked|meaned|payed|readed|rided|
#   rised|selled|sended|shooted|showed|sleeped|speaked|spended|stealed|
#   swimmed|taked|telled|winned
# AFTER (this module) -- "showed" removed. The hand-check found "showed" is
# the standard regular past tense of "show" (only the participle "shown" is
# irregular), so it was a pure wordlist error, not an ambiguous edge case --
# it fired on 4 of the 6 SEED examples pulled, all of them genuine Standard
# English ("the marks on his face showed that...", etc.), and 0 of 6 Arm D
# examples. Every other entry on the list was hand-check-confirmed clean in
# Arm D (bringed, runned, throwed, seed[verb], ringed, singed all judged
# PLAUSIBLE) and is kept verbatim.
# --------------------------------------------------------------------------
_REGULARIZED_PAST_FINAL_RE = re.compile(
    r"\b(?:knowed|throwed|seed|growed|blowed|drawed|catched|teached|bringed|buyed|"
    r"thinked|standed|runned|singed|ringed|drinked|sweared|tored|worned|choosed|"
    r"freezed|wroted|comed|beated|breaked|builded|feeled|finded|flied|forgetted|"
    r"getted|goed|hurted|keeped|leaved|losed|maked|meaned|payed|readed|rided|"
    r"rised|selled|sended|shooted|sleeped|speaked|spended|stealed|"
    r"swimmed|taked|telled|winned)\b",
    re.IGNORECASE,
)


def _regularized_past_final_count(text: str) -> int:
    return len(_REGULARIZED_PAST_FINAL_RE.findall(text))


# --------------------------------------------------------------------------
# eWAVE 121 "Double modals" -- WORDLIST EDIT #2 (a modal-PAIR deletion, same
# spirit as the wordlist edit above -- not a logic/structure change).
# BEFORE (src/analysis/grammatical_proxy_v3.py._DOUBLE_MODAL_RE), verbatim:
#   first slot:  might|may|must|could|would|should|used to
#   second slot: could|would|should|can|will|oughta|ought to
# AFTER (this module) -- "will" removed from the second slot. The hand-check
# stress-tested "might will" specifically and found 99 occurrences in Arm D
# (~17% of all 579 double_modal hits pre-deletion), reading in every sampled
# case as Multi-VALUE's own future-marker substitution artifact colliding
# with a modal, not a canonical AAVE double-modal combination -- "might will"
# does not appear among the standard double-modal pairs in the dialectology
# literature (might could / might should / might would / might can / used to
# could, all of which the hand-check found clean and which are kept
# verbatim). Removing "will" removes only that one contaminated pairing.
# --------------------------------------------------------------------------
_DOUBLE_MODAL_FINAL_RE = re.compile(
    r"\b(?:might|may|must|could|would|should|used to)\s+(?:could|would|should|can|oughta|ought to)\b",
    re.IGNORECASE,
)


def _double_modal_final_count(text: str) -> int:
    return len(_DOUBLE_MODAL_FINAL_RE.findall(text))


# --------------------------------------------------------------------------
# eWAVE 180 "Was/were generalization" -- PLURAL-WAS HALF ONLY.
# v3's _WAS_WERE_LEVELING_RE was one regex with two alternatives joined by
# "|": `\b(?:it|he|she|this|that|i)\s+were\b` (singular+were) OR
# `\b(?:we|you|they|these|those)\s+was\b` (plural+was). The hand-check found
# the plural+was half 6/6 clean in BOTH corpora, and the singular+were half
# 6/6 SUSPECT (subjunctive mood "if he were"/"as if it were", the fixed idiom
# "as it were", compound-subject and prepositional-object blindness -- none
# fixable by regex, per the session's own instruction to HOLD rather than
# force a fix). This is the verbatim second alternative of v3's existing
# expression, copied unchanged -- not new logic, and the singular+were half
# is not merely deprioritized, it is entirely absent from this module.
# --------------------------------------------------------------------------
_WAS_LEVELING_PLURAL_ONLY_RE = re.compile(
    r"\b(?:we|you|they|these|those)\s+was\b",
    re.IGNORECASE,
)


def _was_leveling_plural_only_count(text: str) -> int:
    return len(_WAS_LEVELING_PLURAL_ONLY_RE.findall(text))


def grammatical_aave_proxy_final_breakdown(text: str) -> dict:
    """Per-feature hit counts for the 10 ADOPTED features only. Keys named
    by eWAVE feature id(s). HOLD features (170/176/177/178) and
    them-demonstrative (68) are deliberately absent -- they contribute
    nothing to this breakdown or to the combined score below."""
    text = str(text or "")
    return {
        "habitual_be_90": _habitual_be_count(text),
        "zero_copula_progressive_174": _zero_copula_count(text),
        "aint_155_156_157": len(_AINT_RE.findall(text)),
        "negative_concord_154": _multiple_negation_count(text),
        "completive_done_104": _done_completive_count(text),
        "regularized_reflexive_11": _regularized_reflexive_count(text),
        "invariant_dont_158": _invariant_dont_count(text),
        "regularized_past_128": _regularized_past_final_count(text),
        "double_modal_121": _double_modal_final_count(text),
        "was_leveling_plural_180": _was_leveling_plural_only_count(text),
    }


def grammatical_aave_proxy_final(text: str) -> float:
    """FINAL scored density: total hits (10 adopted features) / word count,
    same units as v2's grammatical_aave_proxy and marker_density. This is
    the version to wire into any future scoring script."""
    text = str(text or "")
    words = text.split()
    if not words:
        return 0.0
    hits = sum(grammatical_aave_proxy_final_breakdown(text).values())
    return hits / len(words)


if __name__ == "__main__":
    print("Scored features (10):")
    for k in grammatical_aave_proxy_final_breakdown(""):
        print(" ", k)
    print()
    print("WORDLIST EDIT #1 -- regularized_past (eWAVE 128):")
    print("  BEFORE: ...payed|readed|rided|rised|selled|sended|shooted|showed|sleeped|speaked...")
    print("  AFTER : ...payed|readed|rided|rised|selled|sended|shooted|sleeped|speaked...  ('showed' removed)")
    print()
    print("WORDLIST EDIT #2 -- double_modal (eWAVE 121):")
    print("  BEFORE second slot: could|would|should|can|will|oughta|ought to")
    print("  AFTER  second slot: could|would|should|can|oughta|ought to  ('will' removed)")
