"""
AAVE fiction seed pool -> SAE MGT via mode_b rewrite (run1).

Reads the frozen 1,216-passage seed pool (data/interim/aave_book_seeds.csv,
built and frozen in a prior session; NOT regenerated or modified here),
selects 700 distinct seeds deterministically (harness.select_rows, smallest
sha256(passage_id) first), and assigns them round-robin across four
resolved, live-probed HF router models (175 each): Qwen3.5-9B, Llama-4-Scout,
DeepSeek-V3.2, Kimi-K2.6 (see docs/decisions.md, 2026-07-26 entry). Mode is
always "b" (rewrite) -- this produces SAE MGT from AAVE human seeds only;
no model is ever asked to write AAVE (see templates.py).

The round-robin assignment (seed i -> MODEL_ORDER[i % 4], over the
select_rows-sorted 700) doubles as the processing order, so every 50-item
checkpoint naturally contains a roughly even mix of all four models rather
than one model's whole batch -- this is what makes the per-model rolling
error/contamination rates meaningful at each checkpoint.

Gate 1 (first 20 items = first 5 per model, since 20 items round-robinned
over 4 models gives exactly 5 each) is a hard pilot check before any
further spend; Gate 2 (remaining 680) has rolling per-model halts every 50
total items. Both gates share one continuous output file, checkpointed
incrementally so a halt never loses completed work.

Read-only on data/processed/, data/interim/aave_book_seeds.csv, templates.py,
and any loader. Writes only data/interim/aave_fiction_mgt_modeb_run1.csv.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generate import harness as H
from src.generate.config import ModelConfig
from src.generate.templates import render, TEMPLATE_VERSION

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("generate.run_fiction_modeb_run1")

SEEDS_PATH = H.INTERIM / "aave_book_seeds.csv"
OUT_PATH = H.INTERIM / "aave_fiction_mgt_modeb_run1.csv"

N_TOTAL = 700
N_PER_MODEL = 175
GATE1_N_PER_MODEL = 5
GATE1_TOTAL = GATE1_N_PER_MODEL * 4  # 20
CHECKPOINT_EVERY = 50
CONTAMINATION_THRESHOLD = 0.6
MODE = "b"
GENRE = "short_fiction"

MODEL_ORDER = ["qwen3", "llama", "deepseek", "kimi"]
MODEL_IDS = {
    "qwen3": "Qwen/Qwen3.5-9B",
    "llama": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "kimi": "moonshotai/Kimi-K2.6",
}
# Kimi does not honor /no_think (confirmed in the liveness probe); it needs a
# larger output budget instead to clear its reasoning pass before answering.
MODEL_MAX_TOKENS = {"qwen3": 800, "llama": 800, "deepseek": 800, "kimi": H.HuggingFaceClient.HARD_MAX_TOKENS_CEILING}
MODEL_DISABLE_THINKING = {"qwen3": True, "llama": False, "deepseek": False, "kimi": False}

# Generous ceiling: ~700 calls x (~400 in + up to ~1200 out for kimi, ~800 for
# others) comfortably clears 1.2M; max_requests kept far above 700 per the
# task brief ("e.g. 1000").
BUDGET_MAX_REQUESTS = 1000
BUDGET_MAX_TOKENS = 1_200_000

REFUSAL_PATTERNS = [
    r"\bI cannot\b", r"\bI can not\b", r"\bI'm sorry\b", r"\bI am sorry\b",
    r"\bas an AI\b", r"\bI am unable\b", r"\bI won't\b", r"\bI will not\b",
    r"\bI'm not able to\b", r"\bI don't feel comfortable\b",
]
THINK_LEAK_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)

OUTPUT_COLUMNS = [
    "generation_id", "source_passage_id", "source_book_id",
    "model_short_name", "model_full_id", "mode", "template_version",
    "prompt", "completion", "tokens_in", "tokens_out",
    "contamination_score", "contamination_flag",
    "finish_reason", "error", "timestamp", "git_commit", "git_dirty",
]


# A refusal is the model declining in its OWN voice, not a quote inside a
# rewritten story (e.g. '"That I cannot tell, Miss Etheridge"' is dialogue,
# not a refusal) -- confirmed as a real false positive on 2/20 Gate 1 items.
# A lead-window heuristic (only count matches near the start of the
# completion) was tried first and still false-positived, since a story can
# legitimately open with dialogue containing the phrase. The reliable
# discriminator is quote context: a match preceded by an odd number of
# quote marks is inside an still-open quotation (dialogue), not the
# model's own narration -- straight and curly quotes are counted together
# since prose typically opens/closes them in strict alternating pairs.
_QUOTE_CHARS = ('"', "“", "”")


def _inside_quotes(text: str, pos: int) -> bool:
    n_quotes = sum(text.count(c, 0, pos) for c in _QUOTE_CHARS)
    return n_quotes % 2 == 1


def is_refusal_or_empty(text: str) -> bool:
    if not text or not text.strip():
        return True
    for p in REFUSAL_PATTERNS:
        for m in re.finditer(p, text, re.IGNORECASE):
            if not _inside_quotes(text, m.start()):
                return True
    return False


def has_thinking_leak(text: str) -> bool:
    return bool(THINK_LEAK_RE.search(text or ""))


def contamination_score(seed_text: str, output_text: str) -> float:
    return difflib.SequenceMatcher(None, seed_text, output_text or "", autojunk=False).ratio()


def build_assignment(seeds_df: pd.DataFrame) -> pd.DataFrame:
    selected = H.select_rows(seeds_df, N_TOTAL, id_col="passage_id").reset_index(drop=True)
    assert len(selected) == N_TOTAL, f"select_rows returned {len(selected)}, expected {N_TOTAL}"
    selected["model_short_name"] = [MODEL_ORDER[i % 4] for i in range(len(selected))]
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
    print(f"Selected {N_TOTAL} distinct seeds, assigned round-robin: {N_PER_MODEL} per model "
          f"({', '.join(MODEL_ORDER)})")

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
    n_errors_by_model = {m: 0 for m in MODEL_ORDER}
    n_calls_by_model = {m: 0 for m in MODEL_ORDER}
    n_contam_by_model = {m: 0 for m in MODEL_ORDER}
    gate1_items = {m: [] for m in MODEL_ORDER}  # for the gate1-specific report
    halted = None
    halt_examples = []

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

        contam = contamination_score(seed_text, completion) if completion else 0.0
        contam_flag = contam >= CONTAMINATION_THRESHOLD

        rec = {
            "generation_id": f"mgt_modeb_{passage_id}_{model_short}",
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
            "contamination_score": round(contam, 4),
            "contamination_flag": contam_flag,
            "finish_reason": finish_reason,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_sha,
            "git_dirty": git_dirty,
        }
        rec["_is_fail_abc"] = bool(error) or is_refusal_or_empty(completion) or has_thinking_leak(completion) or \
            (len(completion.split()) < 50 if completion else True)
        return rec

    print(f"\nGATE 1 -- pilot ({GATE1_TOTAL} items, {GATE1_N_PER_MODEL} per model)")
    for i in range(GATE1_TOTAL):
        row = assignment.iloc[i]
        try:
            rec = process_one(row)
        except H.LiveBudgetExceeded as e:
            halted = f"LiveBudgetExceeded during Gate 1: {e}"
            break
        model_short = rec["model_short_name"]
        n_calls_by_model[model_short] += 1
        if rec["error"] or rec["_is_fail_abc"]:
            n_errors_by_model[model_short] += 1
        if rec["contamination_flag"]:
            n_contam_by_model[model_short] += 1
        gate1_items[model_short].append(rec)
        buffer.append({k: v for k, v in rec.items() if k != "_is_fail_abc"})
        n_done += 1

    flush()

    if halted:
        print(f"STOPPED: {halted}")
    else:
        gate1_report = {}
        gate1_fail = False
        for m in MODEL_ORDER:
            items = gate1_items[m]
            n_fail_abc = sum(1 for it in items if it["error"] or it["_is_fail_abc"])
            n_contam = sum(1 for it in items if it["contamination_flag"])
            gate1_report[m] = {"n": len(items), "n_fail_abc": n_fail_abc, "n_contam_flagged": n_contam}
            print(f"  {m}: n={len(items)} fail(a/b/c)={n_fail_abc}/5 contam_flagged={n_contam}/5")
            if n_fail_abc >= 2 or n_contam >= 3:
                gate1_fail = True
                halt_examples.extend([
                    {"model": m, "passage_id": it["source_passage_id"], "error": it["error"],
                     "completion_head": (it["completion"] or "")[:300], "contamination_score": it["contamination_score"]}
                    for it in items if it["error"] or it["_is_fail_abc"] or it["contamination_flag"]
                ])

        if gate1_fail:
            halted = "GATE 1 FAILED"
            print(f"\n{halted} -- halting before Gate 2. Failing model(s) detail:")
            print(json.dumps(halt_examples, indent=2, default=str)[:4000])
        else:
            print("\nGATE 1 PASSED")
            print(json.dumps(gate1_report, indent=2))

    stop_point = None
    if not halted:
        print(f"\nGATE 2 -- full run, remaining {N_TOTAL - GATE1_TOTAL} items, "
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
            if rec["error"] or rec["_is_fail_abc"]:
                n_errors_by_model[model_short] += 1
            if rec["contamination_flag"]:
                n_contam_by_model[model_short] += 1
            buffer.append({k: v for k, v in rec.items() if k != "_is_fail_abc"})
            n_done += 1

            if n_done % CHECKPOINT_EVERY == 0:
                flush()
                print(f"  [{n_done}/{N_TOTAL}] tokens_used={budget.tokens_used}/{BUDGET_MAX_TOKENS} "
                      f"requests_used={budget.requests_used}/{BUDGET_MAX_REQUESTS}")
                for m in MODEL_ORDER:
                    nc = n_calls_by_model[m]
                    if nc == 0:
                        continue
                    err_rate = n_errors_by_model[m] / nc
                    contam_rate = n_contam_by_model[m] / nc
                    print(f"    {m}: n={nc} err+refusal_rate={err_rate:.1%} contam_rate={contam_rate:.1%}")
                    if err_rate > 0.20 or contam_rate > 0.40:
                        halted = (f"ROLLING HALT: {m} err+refusal_rate={err_rate:.1%} "
                                  f"(>20%) or contam_rate={contam_rate:.1%} (>40%) at n={n_done}")
                        stop_point = {"index": i, "n_done_before_stop": n_done, "model": m}
                        break
            if halted:
                break

    flush()

    print("\n" + "=" * 70)
    print("AAVE FICTION MODE_B SAE MGT -- RUN1 SUMMARY")
    print("=" * 70)
    print(f"Target: N={N_TOTAL} ({N_PER_MODEL} per model)")
    print(f"Completed: {n_done}")
    for m in MODEL_ORDER:
        print(f"  {m}: n={n_calls_by_model[m]} errors={n_errors_by_model[m]} contam_flagged={n_contam_by_model[m]}")
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
        "n_errors_by_model": n_errors_by_model, "n_contam_by_model": n_contam_by_model,
        "halted": halted, "stop_point": stop_point,
        "budget": {"tokens_used": budget.tokens_used, "requests_used": budget.requests_used},
    }


if __name__ == "__main__":
    main()
