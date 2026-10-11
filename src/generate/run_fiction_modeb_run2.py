"""
AAVE fiction seed pool -> SAE MGT via mode_b rewrite (run2).

Supersedes run1 (src/generate/run_fiction_modeb_run1.py), which halted at
its Gate 1 pilot: Kimi was dropped for burning its full 1200-token ceiling
on hidden reasoning and returning empty completions (5/5), and its refusal
detector false-positived on in-story dialogue ("That I cannot tell, Miss
Etheridge" is not a refusal) -- fixed in run1.py directly (FIX 1) and
reused here via import.

Design changes from run1, per the 2026-07-25/26 decisions.md entries and
this run's brief:
  - Three models only: Qwen3.5-9B, Llama-4-Scout, DeepSeek-V3.2. Kimi is
    dropped, not re-added; no 4th model is added.
  - N=699 (233/model, divides evenly), not 700/175.
  - Similarity ("contamination") is RECORDED, never a halt condition. The
    authenticity gap is expected to be non-uniform -- a mostly-narration
    seed has little dialect to flatten, so a near-identical SAE rewrite of
    it is a legitimate research finding (dialect concentrates in dialogue,
    not narration), not contamination.
  - Two density measures are scored on every seed AND every output, so the
    gap can be reported as a function of density:
      1. orthographic marker density (src.analysis.metrics.marker_density,
         imported, not reimplemented).
      2. a coarse grammatical-AAVE surface-pattern proxy (this module),
         since the orthographic measure is blind to grammatically-AAVE
         text in standard spelling (e.g. aave_lyric.csv scored 0.074 on it
         despite being real AAVE).
  - Halting is reserved for genuine mechanical breakage only: empty
    completions, refusals (post-FIX-1), thinking-leak, provider errors, or
    LiveBudget exhaustion. Similarity and density values never halt a run.

Read-only on data/processed/, data/interim/aave_book_seeds.csv, templates.py,
and any loader. Writes only data/interim/aave_fiction_mgt_modeb_run2.csv.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generate import harness as H
from src.generate.config import ModelConfig
from src.generate.templates import render, TEMPLATE_VERSION
from src.generate.run_fiction_modeb_run1 import (
    REFUSAL_PATTERNS, _inside_quotes, THINK_LEAK_RE, has_thinking_leak,
)
from src.analysis.metrics import marker_density

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("generate.run_fiction_modeb_run2")

SEEDS_PATH = H.INTERIM / "aave_book_seeds.csv"
OUT_PATH = H.INTERIM / "aave_fiction_mgt_modeb_run2.csv"

N_TOTAL = 699
N_PER_MODEL = 233
GATE1_N_PER_MODEL = 5
GATE1_TOTAL = GATE1_N_PER_MODEL * 3  # 15
CHECKPOINT_EVERY = 50
MODE = "b"
GENRE = "short_fiction"

MODEL_ORDER = ["qwen3", "llama", "deepseek"]
MODEL_IDS = {
    "qwen3": "Qwen/Qwen3.5-9B",
    "llama": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
}
# Unchanged from run1 (HARD RULE: do not change the token ceiling). Qwen3
# MUST get /no_think appended every call -- it is a hybrid reasoner with
# thinking on by default. Llama and DeepSeek answer cleanly with no
# reasoning by default, so disable_thinking is a no-op for them.
MODEL_MAX_TOKENS = {"qwen3": 800, "llama": 800, "deepseek": 800}
MODEL_DISABLE_THINKING = {"qwen3": True, "llama": False, "deepseek": False}

BUDGET_MAX_REQUESTS = 1000
BUDGET_MAX_TOKENS = 1_000_000
ROLLING_FAIL_RATE_HALT = 0.20  # empty+refusal+error rate; genuine breakage only

OUTPUT_COLUMNS = [
    "generation_id", "source_passage_id", "source_book_id",
    "model_short_name", "model_full_id", "mode", "template_version",
    "prompt", "completion", "tokens_in", "tokens_out", "finish_reason",
    "seed_orthographic_density", "seed_grammatical_proxy",
    "output_orthographic_density", "output_grammatical_proxy",
    "seed_vs_output_similarity", "refusal_bool", "error",
    "timestamp", "git_commit", "git_dirty",
]


# --------------------------------------------------------------------------
# Refusal (post-FIX-1, imported from run1) vs. emptiness -- kept as two
# separate signals here since this run's schema wants a standalone
# refusal_bool column distinct from an empty completion.
# --------------------------------------------------------------------------

def is_empty(text: str) -> bool:
    return not text or not str(text).strip()


def is_refusal(text: str) -> bool:
    """Same quote-context discriminator as run1's fixed is_refusal_or_empty
    (a match inside an odd count of preceding quote marks is dialogue, not
    the model's own voice), but returns False on empty text -- emptiness is
    tracked as its own signal here, not folded into "refusal"."""
    if is_empty(text):
        return False
    for p in REFUSAL_PATTERNS:
        for m in re.finditer(p, text, re.IGNORECASE):
            if not _inside_quotes(text, m.start()):
                return True
    return False


def similarity(seed_text: str, output_text: str) -> float:
    return difflib.SequenceMatcher(None, seed_text, output_text or "", autojunk=False).ratio()


# --------------------------------------------------------------------------
# Coarse grammatical-AAVE surface-pattern proxy.
#
# This is a REGEX SURFACE SCANNER, not a parser -- it will have both false
# positives (e.g. a gerund used as a participial modifier, "She, laughing,
# turned away," can still match the zero-copula pattern) and false
# negatives (predicate-adjective zero-copula like "she nice" is not
# attempted at all, since a closed adjective list would be too fragile to
# be worth the false-positive risk). It exists because the orthographic
# marker-density measure is blind to grammatically-AAVE text in standard
# spelling (aave_lyric.csv scored 0.074 on it despite being real AAVE).
#
# Five categories, exact patterns:
#   1. invariant/habitual "be" + V-ing (e.g. "he be working"), EXCLUDING
#      modal/infinitive-preceded "be" (e.g. "will be working", "to be
#      working" -- ordinary SAE future/infinitive progressive, not
#      habitual aspect).
#   2. zero-copula (progressive only): a subject pronoun directly followed
#      by a V-ing word with no intervening copula (e.g. "she talking").
#      "she is talking" does not match, since "is" sits between the
#      pronoun and the gerund.
#      Two guards added after a hand-check found false fires: (a) the
#      matched "V-ing" word can never be "ain'"/"ain't" -- the character
#      class ['g] can match the apostrophe inside "ain't" (e.g. "we ain'"
#      satisfies \w+in['g]), which is the ain't feature's job, not
#      zero-copula's; (b) a pronoun immediately preceded by am/is/are/
#      was/were is excluded -- that is an aux-inversion question ("What
#      are you thinking") where the copula is real, just positioned
#      before the pronoun instead of between it and the gerund.
#   3. "ain't" (all forms: ain't / aint).
#   4. multiple negation / negative concord: a sentence containing BOTH a
#      negator (ain't/isn't/wasn't/weren't/don't/didn't/doesn't/won't/
#      can't/couldn't/never/not) AND a negative indefinite (nothing/
#      nobody/no one/nowhere/none/neither/no), e.g. "he don't know nothing
#      about it".
#      Sentence splitter also breaks on closing-quote-then-newline and on
#      blank-line paragraph breaks -- a hand-check found two cases where
#      two separate dialogue turns/paragraphs, each with its own
#      independent standard negation, were glued into one pseudo-sentence
#      by the old punctuation-only splitter and falsely scored as concord.
#   5. completive "done" + past participle (e.g. "she done gone", "I done
#      told you") -- a regular -ed/-en participle after "done", plus a
#      fixed list of common irregular participles.
# --------------------------------------------------------------------------

_MODALS_BEFORE_BE = {"will", "would", "shall", "should", "can", "could", "may", "might", "must", "to"}
_HABITUAL_BE_RE = re.compile(r"\bbe\s+\w+in['g]?\b", re.IGNORECASE)

_ZERO_COPULA_RE = re.compile(r"\b(?:I|you|he|she|it|we|they)\s+(\w+in['g])\b", re.IGNORECASE)
# BUG 1: this group can match inside "ain't" (see comment above) -- exclude it.
_ZERO_COPULA_EXCLUDE_VERBS = {"ain'", "ain't"}
# BUG 2: a pronoun preceded by a real copula is an inverted question, not zero-copula.
_ZERO_COPULA_PRECEDING_AUX = {"am", "is", "are", "was", "were"}

_AINT_RE = re.compile(r"\bain'?t\b", re.IGNORECASE)

_NEGATOR_RE = re.compile(
    r"\b(?:ain'?t|isn'?t|wasn'?t|weren'?t|don'?t|didn'?t|doesn'?t|won'?t|can'?t|couldn'?t|never|not)\b",
    re.IGNORECASE,
)
_NEG_INDEF_RE = re.compile(r"\b(?:nothing|nobody|no ?one|nowhere|none|neither|no)\b", re.IGNORECASE)
# BUG 3: also break on terminal-punctuation-then-closing-quote-then-whitespace
# (e.g. '."\n\n"'), and on any blank-line paragraph break, regardless of what
# punctuation precedes it -- the old pattern only broke on bare [.!?] + space,
# so consecutive quoted dialogue turns separated by a blank line were never
# split into separate sentences.
_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[.!?][\"'‘’“”])\s+"
    r"|\n\s*\n+"
)

_DONE_REGULAR_RE = re.compile(r"\bdone\s+\w+(?:ed|en)\b", re.IGNORECASE)
_DONE_IRREGULAR_RE = re.compile(
    r"\bdone\s+(?:gone|been|seen|said|told|took|did|come|ran|broke|wrote|ate|drove|forgot|"
    r"threw|drew|had|got|made|left|found|kept|held|met|paid|sold|sent|spent|stood|won|"
    r"heard|felt|lost|built|caught|fought|brought|bought|taught|thought|sat|woke|rose|"
    r"fell|grew|flew|blew|knew|swore|wore|tore|bore|chose|froze)\b",
    re.IGNORECASE,
)


def _habitual_be_count(text: str) -> int:
    count = 0
    for m in _HABITUAL_BE_RE.finditer(text):
        preceding = text[: m.start()].rstrip()
        prev_words = preceding.split()
        prev = re.sub(r"[^a-z]", "", prev_words[-1].lower()) if prev_words else ""
        if prev not in _MODALS_BEFORE_BE:
            count += 1
    return count


def _zero_copula_count(text: str) -> int:
    count = 0
    for m in _ZERO_COPULA_RE.finditer(text):
        verb = m.group(1).lower()
        if verb in _ZERO_COPULA_EXCLUDE_VERBS:  # BUG 1
            continue
        preceding = text[: m.start()].rstrip()
        prev_words = preceding.split()
        prev = re.sub(r"[^a-z]", "", prev_words[-1].lower()) if prev_words else ""
        if prev in _ZERO_COPULA_PRECEDING_AUX:  # BUG 2
            continue
        count += 1
    return count


def _multiple_negation_count(text: str) -> int:
    count = 0
    for sent in _SENT_SPLIT_RE.split(text):
        if _NEGATOR_RE.search(sent) and _NEG_INDEF_RE.search(sent):
            count += 1
    return count


def _done_completive_count(text: str) -> int:
    return sum(1 for _ in _DONE_REGULAR_RE.finditer(text)) + sum(1 for _ in _DONE_IRREGULAR_RE.finditer(text))


def grammatical_aave_proxy(text: str) -> float:
    """Combined density: total pattern hits / word count, same units
    (hits per word) as marker_density. 0.0 for empty text."""
    text = str(text or "")
    words = text.split()
    if not words:
        return 0.0
    hits = (
        _habitual_be_count(text)
        + _zero_copula_count(text)
        + len(_AINT_RE.findall(text))
        + _multiple_negation_count(text)
        + _done_completive_count(text)
    )
    return hits / len(words)


# --------------------------------------------------------------------------
# Selection + generation
# --------------------------------------------------------------------------

def build_assignment(seeds_df: pd.DataFrame) -> pd.DataFrame:
    selected = H.select_rows(seeds_df, N_TOTAL, id_col="passage_id").reset_index(drop=True)
    assert len(selected) == N_TOTAL, f"select_rows returned {len(selected)}, expected {N_TOTAL}"
    selected["model_short_name"] = [MODEL_ORDER[i % 3] for i in range(len(selected))]
    counts = selected["model_short_name"].value_counts().to_dict()
    for m in MODEL_ORDER:
        assert counts.get(m) == N_PER_MODEL, f"model {m} got {counts.get(m)} seeds, expected {N_PER_MODEL}"
    return selected


def main():
    env = H._load_env_file(H.REPO_ROOT / ".env")
    token = env.get("HF_TOKEN", "").strip()
    if not token:
        print("No HF_TOKEN found -- cannot run live generation. Stopping cleanly.")
        return

    if not SEEDS_PATH.exists():
        print(f"Missing seed pool: {SEEDS_PATH}. Stopping.")
        return
    seeds_df = pd.read_csv(SEEDS_PATH)
    n_distinct = seeds_df["passage_id"].nunique()
    print(f"Seed pool: {len(seeds_df)} rows, {n_distinct} distinct passage_id")
    assert n_distinct >= N_TOTAL, f"reuse guard: only {n_distinct} distinct seeds, need >= {N_TOTAL}"

    assignment = build_assignment(seeds_df)
    print(f"Selected {N_TOTAL} distinct seeds (smallest sha256(passage_id) first), "
          f"assigned round-robin: {N_PER_MODEL} per model ({', '.join(MODEL_ORDER)})")
    print("Model mapping (round-robin index % 3): "
          + ", ".join(f"{m}=idx{i}%3" for i, m in enumerate(MODEL_ORDER)))

    budget = H.LiveBudget(max_requests=BUDGET_MAX_REQUESTS, max_tokens=BUDGET_MAX_TOKENS)
    clients = {m: H.HuggingFaceClient(budget, token, disable_thinking=MODEL_DISABLE_THINKING[m]) for m in MODEL_ORDER}
    model_cfgs = {
        m: ModelConfig(name=m, provider="huggingface", endpoint=MODEL_IDS[m],
                        max_tokens=MODEL_MAX_TOKENS[m], temperature=0.9)
        for m in MODEL_ORDER
    }

    git_sha = H._git_commit_sha()
    git_dirty = H._git_dirty()

    buffer: list[dict] = []
    n_done = 0
    n_calls_by_model = {m: 0 for m in MODEL_ORDER}
    n_fail_by_model = {m: 0 for m in MODEL_ORDER}  # empty + refusal + error (genuine breakage)
    seed_density_all = {"orth": [], "gram": []}
    output_density_all = {"orth": [], "gram": []}
    halted = None
    stop_point = None

    def flush():
        nonlocal buffer
        if buffer:
            H.append_chunk(buffer, OUT_PATH, OUTPUT_COLUMNS)
            buffer = []

    def process_one(row) -> dict:
        model_short = row["model_short_name"]
        model_cfg = model_cfgs[model_short]
        client = clients[model_short]
        passage_id = str(row["passage_id"])
        book_id = str(row["book_id"])
        seed_text = str(row["text"])

        base_prompt, tid = render(GENRE, MODE, source_text=seed_text, target_length=None)
        sent_prompt = base_prompt
        if MODEL_DISABLE_THINKING[model_short]:
            sent_prompt = f"{base_prompt} /no_think"

        error = ""
        completion = ""
        tokens_in = tokens_out = 0
        finish_reason = ""
        try:
            result = client.generate(base_prompt, model_cfg, row_id=passage_id, mode=MODE)
            completion = result.text
            tokens_in = result.tokens_in
            tokens_out = result.tokens_out
            finish_reason = result.finish_reason
        except H.LiveBudgetExceeded:
            raise
        except H.HFGatedModelError as e:
            error = f"HFGatedModelError: {e}"
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"

        seed_orth = marker_density(seed_text)
        seed_gram = grammatical_aave_proxy(seed_text)
        out_orth = marker_density(completion) if completion else 0.0
        out_gram = grammatical_aave_proxy(completion) if completion else 0.0
        sim = similarity(seed_text, completion)
        refusal = is_refusal(completion)
        empty = is_empty(completion)
        think_leak = has_thinking_leak(completion)

        rec = {
            "generation_id": f"mgt_modeb_run2_{passage_id}_{model_short}",
            "source_passage_id": passage_id,
            "source_book_id": book_id,
            "model_short_name": model_short,
            "model_full_id": MODEL_IDS[model_short],
            "mode": MODE,
            "template_version": TEMPLATE_VERSION,
            "prompt": sent_prompt,
            "completion": completion,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "finish_reason": finish_reason,
            "seed_orthographic_density": round(seed_orth, 6),
            "seed_grammatical_proxy": round(seed_gram, 6),
            "output_orthographic_density": round(out_orth, 6),
            "output_grammatical_proxy": round(out_gram, 6),
            "seed_vs_output_similarity": round(sim, 6),
            "refusal_bool": refusal,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_sha,
            "git_dirty": git_dirty,
        }
        rec["_is_fail"] = bool(error) or empty or refusal or think_leak
        rec["_think_leak"] = think_leak
        rec["_empty"] = empty
        return rec

    print(f"\nPILOT -- {GATE1_TOTAL} items ({GATE1_N_PER_MODEL} per model)")
    pilot_items = {m: [] for m in MODEL_ORDER}
    for i in range(GATE1_TOTAL):
        row = assignment.iloc[i]
        try:
            rec = process_one(row)
        except H.LiveBudgetExceeded as e:
            halted = f"LiveBudgetExceeded during pilot: {e}"
            break
        model_short = rec["model_short_name"]
        n_calls_by_model[model_short] += 1
        if rec["_is_fail"]:
            n_fail_by_model[model_short] += 1
        pilot_items[model_short].append(rec)
        seed_density_all["orth"].append(rec["seed_orthographic_density"])
        seed_density_all["gram"].append(rec["seed_grammatical_proxy"])
        output_density_all["orth"].append(rec["output_orthographic_density"])
        output_density_all["gram"].append(rec["output_grammatical_proxy"])
        buffer.append({k: v for k, v in rec.items() if not k.startswith("_")})
        n_done += 1
    flush()

    pilot_clean = True
    if halted:
        pilot_clean = False
        print(f"STOPPED: {halted}")
    else:
        print("\nPILOT REPORT (mechanical cleanliness only):")
        for m in MODEL_ORDER:
            for rec in pilot_items[m]:
                wc = len(rec["completion"].split()) if rec["completion"] else 0
                is_prose = wc > 0 and bool(re.search(r"[A-Za-z]{3,}", rec["completion"] or ""))
                print(f"  {m} | {rec['source_passage_id'][:12]} | finish_reason={rec['finish_reason']!r} "
                      f"| words={wc} | non_empty_english_prose={is_prose} | refusal={rec['refusal_bool']} "
                      f"| empty={rec['_empty']} | think_leak={rec['_think_leak']} | error={rec['error']!r}")
                if m == "qwen3" and rec["_think_leak"]:
                    pilot_clean = False
                if rec["_is_fail"]:
                    pilot_clean = False

    if pilot_clean and not halted:
        print("\nPILOT CLEAN")
    elif not halted:
        halted = "PILOT FAILED (genuine mechanical breakage: empty/refusal/think-leak/error)"
        print(f"\n{halted}")

    if not halted:
        print(f"\nFULL RUN -- remaining {N_TOTAL - GATE1_TOTAL} items, "
              f"checkpoint every {CHECKPOINT_EVERY} total items")
        for i in range(GATE1_TOTAL, N_TOTAL):
            row = assignment.iloc[i]
            try:
                rec = process_one(row)
            except H.LiveBudgetExceeded as e:
                halted = f"LiveBudgetExceeded: {e}"
                stop_point = {"index": i, "n_done_before_stop": n_done}
                break

            model_short = rec["model_short_name"]
            n_calls_by_model[model_short] += 1
            if rec["_is_fail"]:
                n_fail_by_model[model_short] += 1
            seed_density_all["orth"].append(rec["seed_orthographic_density"])
            seed_density_all["gram"].append(rec["seed_grammatical_proxy"])
            output_density_all["orth"].append(rec["output_orthographic_density"])
            output_density_all["gram"].append(rec["output_grammatical_proxy"])
            buffer.append({k: v for k, v in rec.items() if not k.startswith("_")})
            n_done += 1

            if n_done % CHECKPOINT_EVERY == 0:
                flush()
                print(f"  [{n_done}/{N_TOTAL}] tokens_used={budget.tokens_used}/{BUDGET_MAX_TOKENS} "
                      f"requests_used={budget.requests_used}/{BUDGET_MAX_REQUESTS}")
                print(f"    running median seed_orth={statistics.median(seed_density_all['orth']):.4f} "
                      f"seed_gram={statistics.median(seed_density_all['gram']):.4f} "
                      f"out_orth={statistics.median(output_density_all['orth']):.4f} "
                      f"out_gram={statistics.median(output_density_all['gram']):.4f}")
                for m in MODEL_ORDER:
                    nc = n_calls_by_model[m]
                    if nc == 0:
                        continue
                    fail_rate = n_fail_by_model[m] / nc
                    print(f"    {m}: n={nc} empty+refusal+error_rate={fail_rate:.1%}")
                    if fail_rate > ROLLING_FAIL_RATE_HALT:
                        halted = f"ROLLING HALT: {m} empty+refusal+error_rate={fail_rate:.1%} (>20%) at n={n_done}"
                        stop_point = {"index": i, "n_done_before_stop": n_done, "model": m}
                        break
            if halted:
                break

    flush()

    print("\n" + "=" * 70)
    print("AAVE FICTION MODE_B SAE MGT -- RUN2 SUMMARY")
    print("=" * 70)
    print(f"Target: N={N_TOTAL} ({N_PER_MODEL} per model)")
    print(f"Completed: {n_done}")
    for m in MODEL_ORDER:
        print(f"  {m}: n={n_calls_by_model[m]} empty+refusal+error={n_fail_by_model[m]}")
    print(f"Budget: tokens_used={budget.tokens_used}/{BUDGET_MAX_TOKENS} "
          f"requests_used={budget.requests_used}/{BUDGET_MAX_REQUESTS}")
    if halted:
        print(f"\nHALTED: {halted}")
        print(f"Stop point: {stop_point}")
    else:
        print("\nCompleted full plan without hitting a halt condition.")
    print(f"\nWrote -> {OUT_PATH}")

    return {
        "n_done": n_done, "n_calls_by_model": n_calls_by_model,
        "n_fail_by_model": n_fail_by_model,
        "halted": halted, "stop_point": stop_point,
        "budget": {"tokens_used": budget.tokens_used, "requests_used": budget.requests_used},
    }


if __name__ == "__main__":
    main()
