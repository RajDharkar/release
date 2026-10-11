"""
MGT generation harness (Part 4 of the generation scaffold).

Runner for line 2 (SAE-HWT -> LLM -> SAE-MGT) and line 4
(DIA-HWT -> LLM -> SAE-MGT). DryRunClient is the default everywhere; a
single real client is only selected if working credentials are found
(never printed), and every real call is metered by LiveBudget. Output
label is always "MGT" -- Multi-VALUE's arm C/D output is labeled
HGT(MV-perturbation) elsewhere and is untouched by this module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import ModelConfig, load_config
from .templates import TEMPLATE_VERSION, render

log = logging.getLogger("generate.harness")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data/processed"
INTERIM = REPO_ROOT / "data/interim"
MANIFEST_PATH = REPO_ROOT / "manifests/generation_dryrun.json"
LIVE_CALLS_LOG = INTERIM / "live_calls_log.csv"
LIVE_RAW_DIR = INTERIM / "live_raw"

OUTPUT_COLUMNS = [
    "id", "text", "label", "source_row_id", "source_dataset", "genre", "line", "mode",
    "model", "template_id", "template_version", "topic", "sampling_params",
    "timestamp", "git_commit", "git_dirty",
]
ERROR_COLUMNS = ["source_row_id", "source_dataset", "line", "model", "mode", "error_type", "error_message"]
LIVE_CALLS_COLUMNS = [
    "timestamp", "model", "mode", "row_id", "tokens_in", "tokens_out", "latency", "finish_reason",
    "thinking_disable_method", "reasoning_content_chars", "max_tokens_requested", "raw_response_path",
]

LINE2_EXCLUSIONS = {
    "LiteraryTaste": "label==human only",
    "HAP-E": "label==human only; chunk_1 excluded",
    "MillerCenter": "needs_review==True excluded",
}
LINE4_GENRE_OVERRIDE = {"aave_book": "short_fiction", "aave_lyric": "lyrics", "aave_poetry": "poetry"}
LINE4_FILES = {"aave_book": "aave_book.csv", "aave_lyric": "aave_lyric.csv", "aave_poetry": "aave_poetry.csv"}


# --------------------------------------------------------------------------
# Eligibility loading
# --------------------------------------------------------------------------

def load_line2_eligible() -> dict[str, Optional[pd.DataFrame]]:
    """Mirrors src/perturb/arm_c_multivalue.py's eligibility logic exactly
    (read-only; that file is not imported or modified). A source maps to
    None if its backing file is missing -- callers must treat None as
    BLOCKED, not as zero-eligible-rows."""
    result: dict[str, Optional[pd.DataFrame]] = {}
    unified_path = PROCESSED / "unified.csv"
    oratory_path = PROCESSED / "sae_oratory.csv"

    if unified_path.exists():
        unified = pd.read_csv(unified_path, low_memory=False)
        lt = unified[
            (unified["source_dataset"] == "mj-storytelling/LiteraryTaste") & (unified["label"] == "human")
        ].copy()
        hape = unified[
            (unified["source_dataset"] == "browndw/human-ai-parallel-corpus")
            & (unified["label"] == "human")
            & (~unified["source_subset"].fillna("").str.contains("chunk_1"))
        ].copy()
        result["LiteraryTaste"] = lt
        result["HAP-E"] = hape
    else:
        result["LiteraryTaste"] = None
        result["HAP-E"] = None
        log.warning("BLOCKED: data/processed/unified.csv is missing -- LiteraryTaste and HAP-E unavailable for line2.")

    if oratory_path.exists():
        oratory = pd.read_csv(oratory_path, low_memory=False)
        result["MillerCenter"] = oratory[oratory["needs_review"] == False].copy()  # noqa: E712
    else:
        result["MillerCenter"] = None
        log.warning("BLOCKED: data/processed/sae_oratory.csv is missing -- MillerCenter unavailable for line2.")

    return result


def load_line4_eligible() -> dict[str, pd.DataFrame]:
    """line4 = aave_book + aave_lyric + aave_poetry, per spec, with NO
    needs_review filter (the exclusion list given in the spec -- Miller
    Center needs_review, HAP-E chunk_1, LiteraryTaste human-only -- is
    stated for line2 only; no line4 exclusion was specified).

    FLAG: aave_lyric.csv is needs_review==True on 100% of rows
    (13,088/13,088); aave_book.csv is needs_review==True on ~22%
    (5,978/27,278). Both are included in full since no line4 exclusion
    was given -- see generation_scaffold_report.md.

    FLAG: the underlying `genre` column reads "poetry" for BOTH
    aave_lyric.csv and aave_poetry.csv rows, but the task's genre list
    treats "lyrics" as distinct from "poetry" and names aave_lyric.csv as
    the lyrics source. This loader assigns `_effective_genre` by SOURCE
    FILE identity (not by trusting the stored `genre` column) so lyrics
    template selection works as intended -- this is a harness-internal
    labeling choice, not a rewrite of the underlying CSV's own genre
    column, and is called out here rather than applied silently.
    """
    result: dict[str, pd.DataFrame] = {}
    for src, fname in LINE4_FILES.items():
        df = pd.read_csv(PROCESSED / fname, low_memory=False).copy()
        df["_effective_genre"] = LINE4_GENRE_OVERRIDE[src]
        result[src] = df
    return result


# --------------------------------------------------------------------------
# Deterministic mode assignment + row selection
# --------------------------------------------------------------------------

def compute_seed(row_id: str) -> int:
    return int(hashlib.sha256(str(row_id).encode()).hexdigest()[:8], 16) or 1


def _row_hash(row_id: str) -> str:
    return hashlib.sha256(str(row_id).encode()).hexdigest()


def assign_modes(row_ids: list[str]) -> dict[str, str]:
    """Deterministic, EXACTLY balanced 50/50 split within a cell (a set of
    row_ids sharing a (line, source) pair). Sorts row_ids by
    sha256(row_id) hex digest and assigns the first ceil(n/2) to mode_a,
    the rest to mode_b.

    Superseded the earlier per-row parity scheme
    (int(sha256(row_id), 16) % 2), which was a property of each row in
    isolation and therefore had no mechanism to correct itself toward
    balance in a small sample -- confirmed empirically in the first dry
    run (all 9 measured cells landed outside the required 50/50-within-±1
    band; see reports/generation_scaffold_report.md). This rank-based
    scheme is still a pure function of the row_ids themselves (not of
    input order, not of Python's builtin hash()/PYTHONHASHSEED, not of
    which model or line is asking), so it stays deterministic and
    order-independent -- it just also guarantees exact balance."""
    ordered = sorted({str(r) for r in row_ids}, key=_row_hash)
    n_a = math.ceil(len(ordered) / 2)
    return {rid: ("a" if i < n_a else "b") for i, rid in enumerate(ordered)}


def select_rows(df: pd.DataFrame, n: int, id_col: str = "id") -> pd.DataFrame:
    """Deterministic, order-independent selection: the n rows with the
    smallest sha256(id) hash, sorted by that hash. Not a positional slice
    -- reordering the input DataFrame changes nothing."""
    tmp = df.copy()
    tmp["_hash"] = tmp[id_col].astype(str).map(_row_hash)
    tmp = tmp.sort_values("_hash", kind="mergesort")
    return tmp.head(min(n, len(tmp))).drop(columns=["_hash"])


# --------------------------------------------------------------------------
# Provider layer
# --------------------------------------------------------------------------

@dataclass
class GenerationResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency: float
    finish_reason: str
    reasoning_content_chars: int = 0
    thinking_disable_method: str = ""
    raw_response_path: str = ""
    no_think_artifact_stripped: bool = False


# Matches a trailing "/no_think" or bare "no_think" artifact (any casing,
# with or without the leading slash, with or without whitespace around
# the slash) at the very end of a response string -- e.g. "...content.
# /no_think", "...content / no_think", "...content\nno_think". Anchored
# at \Z (absolute end) so a legitimate mid-text mention is never touched
# -- only a genuine trailing echo of the suffix this client itself sent.
_NO_THINK_TRAILING_RE = re.compile(r"\s*/?\s*no_think\s*\Z", re.IGNORECASE)


def strip_no_think_artifact(text: str) -> tuple[str, bool]:
    """Response-side fix for the /no_think leak (see
    reports/generation_scaffold_report.md and the contamination audit):
    the ROOT CAUSE is that this client appends '/no_think' to the PROMPT
    with no separator, and some models occasionally echo it back as if it
    were trailing content. Preventing that on the prompt side turned out
    to be unreliable (a trailing newline reduced but did not eliminate
    recurrence, and the numbered-list parser's own regex only incidentally
    filtered it out for ONE caller). This function is the actual fix:
    applied here, in the single shared response path, it covers every
    caller uniformly -- mode_a, mode_b, and both of partial.py's paths
    (the primary numbered-list call AND the per-segment fallback call) --
    without depending on any caller's own parsing logic. Returns
    (cleaned_text, stripped: bool)."""
    if not text:
        return text, False
    cleaned = _NO_THINK_TRAILING_RE.sub("", text)
    return cleaned, cleaned != text


class GenerationClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, model: ModelConfig, *, row_id: str, mode: str) -> GenerationResult:
        ...


class DryRunClient(GenerationClient):
    """Zero network, zero tokens. Default client everywhere."""

    def generate(self, prompt: str, model: ModelConfig, *, row_id: str, mode: str) -> GenerationResult:
        return GenerationResult(
            text=f"[DRYRUN output for {model.name}/{mode}/{row_id}]",
            tokens_in=0, tokens_out=0, latency=0.0, finish_reason="dryrun",
        )


class LiveBudgetExceeded(Exception):
    pass


@dataclass
class LiveBudget:
    max_requests: int = 30
    max_tokens: int = 60_000
    requests_used: int = 0
    tokens_used: int = 0

    def remaining_requests(self) -> int:
        return self.max_requests - self.requests_used

    def remaining_tokens(self) -> int:
        return self.max_tokens - self.tokens_used

    def check(self) -> None:
        if self.requests_used >= self.max_requests:
            raise LiveBudgetExceeded(f"max_requests ({self.max_requests}) reached")
        if self.tokens_used >= self.max_tokens:
            raise LiveBudgetExceeded(f"max_tokens ({self.max_tokens}) reached")

    def record(self, tokens_in: int, tokens_out: int) -> None:
        self.requests_used += 1
        self.tokens_used += tokens_in + tokens_out


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


PROVIDER_ENV_KEYS = {
    "huggingface": "HF_TOKEN",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
}


def detect_working_credentials() -> dict[str, bool]:
    """Presence check only -- never returns or logs the actual values."""
    env = {**_load_env_file(REPO_ROOT / ".env"), **os.environ}
    return {provider: bool(env.get(key, "").strip()) for provider, key in PROVIDER_ENV_KEYS.items()}


class _NotConfiguredClient(GenerationClient):
    def __init__(self, provider: str):
        self.provider = provider

    def generate(self, prompt, model, *, row_id, mode):
        raise NotImplementedError(
            f"Provider '{self.provider}' has no real-client implementation in this scaffold."
        )


HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


class HFGatedModelError(Exception):
    """Model exists on the router but returned a gated/license error --
    the model owner must accept its license on the model's HF page."""


class HuggingFaceClient(GenerationClient):
    """Real client against the HF Inference Providers router
    (OpenAI-compatible chat completions). Open-model provider, preferred
    when available. `model.endpoint` must carry the resolved HF model id
    (e.g. "Qwen/Qwen3-8B") -- see resolve_hf_model() -- not the project's
    short `model.name` label. The token is held only in-memory on this
    instance; it is never logged, printed, or included in any exception
    message (errors reference the model id and HTTP status only)."""

    HARD_MAX_TOKENS_CEILING = 1200  # was a hardcoded 800; mode_b callers may now request up to this

    def __init__(self, budget: LiveBudget, token: str, disable_thinking: bool = True):
        self.budget = budget
        self._token = token
        self.disable_thinking = disable_thinking

    def generate(self, prompt: str, model: ModelConfig, *, row_id: str, mode: str) -> GenerationResult:
        self.budget.check()
        headers = {"Authorization": f"Bearer {self._token}"}

        methods_applied = []
        sent_prompt = prompt
        payload = {
            "model": model.endpoint,
            "max_tokens": min(model.max_tokens, self.HARD_MAX_TOKENS_CEILING),
            "temperature": model.temperature,
        }
        if self.disable_thinking:
            # Method 1: chat_template_kwargs (vLLM/SGLang OpenAI-compat convention).
            # Empirically confirmed NOT honored by this router's current Qwen3
            # backend (still produced full reasoning_content and an empty/null
            # `content` on a diagnostic call) -- sent anyway since it's harmless
            # and may be honored by other backends/providers.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            methods_applied.append("chat_template_kwargs")
            # Method 2: Qwen3's documented soft-switch convention -- append
            # "/no_think" to the user turn. Empirically confirmed effective
            # (reasoning_content collapses to near-nothing, content is populated,
            # finish_reason=stop instead of length) -- this is the one that
            # actually works for this router/backend.
            sent_prompt = f"{prompt} /no_think"
            methods_applied.append("no_think_suffix")

        payload["messages"] = [{"role": "user", "content": sent_prompt}]

        t0 = time.time()
        resp = requests.post(f"{HF_ROUTER_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
        elapsed = time.time() - t0

        if resp.status_code == 402:
            raise LiveBudgetExceeded(f"HF router returned 402 (quota/payment required) for model={model.endpoint}")
        if resp.status_code in (401, 403):
            raise HFGatedModelError(
                f"HF router returned {resp.status_code} for model={model.endpoint} -- "
                f"likely gated (owner must accept the license on the model's HF page): {resp.text[:200]}"
            )
        resp.raise_for_status()

        data = resp.json()

        # Retain the full raw response body for every live call, so an
        # empty output is diagnosable (e.g. via reasoning_content) instead
        # of a mystery -- this is what caught the thinking-mode issue.
        LIVE_RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{row_id}_{uuid.uuid4().hex[:8]}.json"
        raw_path = LIVE_RAW_DIR / raw_filename
        with open(raw_path, "w") as f:
            json.dump({"request": payload, "response": data}, f, indent=2, default=str)

        choice = data["choices"][0]
        message = choice.get("message", {})
        text = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason", "unknown")
        usage = data.get("usage") or {}
        tokens_in = usage.get("prompt_tokens") or 0
        tokens_out = usage.get("completion_tokens") or 0

        # Response-side /no_think cleanup (see strip_no_think_artifact's
        # docstring) -- scoped to disable_thinking since that's the only
        # condition under which this client ever sends the suffix in the
        # first place. Applied here, after the raw response has already
        # been archived above, so the raw capture stays the untouched
        # ground truth for any future audit.
        no_think_stripped = False
        if self.disable_thinking:
            text, no_think_stripped = strip_no_think_artifact(text)

        self.budget.record(tokens_in, tokens_out)
        return GenerationResult(
            text=text, tokens_in=tokens_in, tokens_out=tokens_out, latency=elapsed, finish_reason=finish_reason,
            reasoning_content_chars=len(reasoning_content), thinking_disable_method="+".join(methods_applied),
            raw_response_path=str(raw_path.relative_to(REPO_ROOT)),
            no_think_artifact_stripped=no_think_stripped,
        )


def list_hf_router_models(token: str) -> list[dict]:
    resp = requests.get(f"{HF_ROUTER_BASE_URL}/models", headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def ping_hf_model(token: str, model_id: str, timeout: float = 30) -> tuple[bool, float, str]:
    """1-token ping. Returns (ok, elapsed_seconds, detail)."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{HF_ROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": model_id, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 1},
            timeout=timeout,
        )
        elapsed = time.time() - t0
        if resp.status_code == 200:
            return True, elapsed, "ok"
        return False, elapsed, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, f"{type(e).__name__}: {e}"


def resolve_hf_models(token: str) -> dict[str, list[str]]:
    """Resolve current model ids on the router for qwen3, llama, mistral --
    by family keyword, not a hardcoded/memorized id (the catalog changes
    over time)."""
    models = list_hf_router_models(token)
    families: dict[str, list[str]] = {"qwen3": [], "llama": [], "mistral": []}
    for m in models:
        mid = m["id"]
        low = mid.lower()
        if "qwen3" in low:
            families["qwen3"].append(mid)
        elif "llama" in low and "guard" not in low:
            families["llama"].append(mid)
        elif "mistral" in low:
            families["mistral"].append(mid)
    return families


PREFERRED_HF_MODEL_ID = "Qwen/Qwen3.5-9B"

# Parameter-count extraction, used to rank fallback candidates by size.
# Requires the digit run to be a standalone token bounded by non-digits
# (or string start/end) before the "B" -- e.g. matches "14" in
# "Qwen3-14B" and "4" in "Qwen3-4B-Instruct-2507", and does NOT treat
# "Qwen3-14B" as if it contained "4B" the way a bare substring check
# ("4B" in model_id) incorrectly did in an earlier version of this
# harness (that bug picked a 14B model instead of the intended 4B one).
_SIZE_RE = re.compile(r"(?:^|[^0-9])(\d+(?:\.\d+)?)B(?:[^0-9]|$)")


def _param_size_b(model_id: str) -> float:
    m = _SIZE_RE.search(model_id)
    return float(m.group(1)) if m else float("inf")


def pick_hf_model(token: str, preferred_exact_id: str = PREFERRED_HF_MODEL_ID, max_fallbacks: int = 4) -> tuple[str, dict]:
    """Model selection: pings `preferred_exact_id` first via EXACT
    membership check (not substring) against the resolved catalog; only
    falls back to the smallest-by-parameter-count candidates (ranked via
    _param_size_b's anchored regex) if the preferred id isn't present or
    doesn't respond to its ping."""
    families = resolve_hf_models(token)
    all_ids = families["qwen3"] + families["llama"] + families["mistral"]
    ping_log = []

    ordered: list[str] = []
    if preferred_exact_id in all_ids:  # exact match, not `in` on the string itself
        ordered.append(preferred_exact_id)
    fallback_pool = sorted((m for m in all_ids if m != preferred_exact_id), key=_param_size_b)
    ordered.extend(fallback_pool[:max_fallbacks])

    for mid in ordered:
        ok, elapsed, detail = ping_hf_model(token, mid)
        ping_log.append({"model_id": mid, "ok": ok, "elapsed_s": round(elapsed, 3), "detail": detail if not ok else "ok"})
        if ok:
            return mid, {"families_found": {k: len(v) for k, v in families.items()}, "pings": ping_log}

    raise RuntimeError(f"No candidate model responded to a 1-token ping. Log: {ping_log}")


class AnthropicClient(GenerationClient):
    def __init__(self, budget: LiveBudget):
        self.budget = budget

    def generate(self, prompt, model, *, row_id, mode):
        raise NotImplementedError("AnthropicClient real inference is not wired up in this scaffold.")


PROVIDER_CLIENTS = {
    # huggingface is special-cased in select_real_client() below (needs the token)
    "anthropic": AnthropicClient,
}
OPEN_MODEL_PROVIDER_PREFERENCE = ["huggingface", "together", "mistral"]


def select_real_client(budget: LiveBudget) -> tuple[Optional[GenerationClient], Optional[str]]:
    """Returns (client, provider_name) or (None, None) if no provider has
    working credentials. Prefers an open-model provider over Anthropic."""
    creds = detect_working_credentials()
    working = [p for p, ok in creds.items() if ok]
    if not working:
        return None, None

    chosen = next((p for p in OPEN_MODEL_PROVIDER_PREFERENCE if p in working), None)
    if chosen is None and "anthropic" in working:
        chosen = "anthropic"
    if chosen is None:
        chosen = working[0]

    if chosen == "huggingface":
        env = {**_load_env_file(REPO_ROOT / ".env"), **os.environ}
        token = env.get("HF_TOKEN", "").strip()
        return HuggingFaceClient(budget, token), chosen

    client_cls = PROVIDER_CLIENTS.get(chosen)
    if client_cls is None:
        return _NotConfiguredClient(chosen), chosen
    return client_cls(budget), chosen


def log_live_call(row: dict) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    write_header = not LIVE_CALLS_LOG.exists()
    df.to_csv(LIVE_CALLS_LOG, mode="a", header=write_header, index=False)


# --------------------------------------------------------------------------
# Provenance + row generation
# --------------------------------------------------------------------------

def _git_commit_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:
        return None


def generate_one(
    client: GenerationClient,
    model_cfg: ModelConfig,
    *,
    genre: str,
    source_row_id: str,
    source_dataset: str,
    line: int,
    mode: str,
    text: str,
    topic: Optional[str],
    target_length: Optional[int],
    git_sha: Optional[str],
    git_dirty: Optional[bool],
) -> tuple[dict, float]:
    prompt, tid = render(genre, mode, topic=topic, source_text=text, target_length=target_length)
    t0 = time.time()
    result = client.generate(prompt, model_cfg, row_id=source_row_id, mode=mode)
    elapsed = time.time() - t0
    row = {
        "id": f"mgt_{source_row_id}_{model_cfg.name}_{mode}",
        "text": result.text,
        "label": "MGT",
        "source_row_id": source_row_id,
        "source_dataset": source_dataset,
        "genre": genre,
        "line": line,
        "mode": mode,
        "model": model_cfg.name,
        "template_id": tid,
        "template_version": TEMPLATE_VERSION,
        "topic": topic if mode == "a" else None,
        "sampling_params": json.dumps({"temperature": model_cfg.temperature, "max_tokens": model_cfg.max_tokens}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
    }
    return row, elapsed


def append_chunk(rows: list[dict], path: Path, columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)
