"""
Config for the MGT generation harness (Part 1 of the generation scaffold).

Dataclass-based, overridable from a `generation:` block appended to the
repo's existing config.yaml. N per cell and per-genre target lengths
default to None ("unset") -- these are pending advisor decisions, and the
harness must refuse to run in full (non-dry-run, non-smoke) mode until
they are filled in. See docs/decisions.md for the project's convention of
flagging pending decisions rather than picking a default silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML_PATH = REPO_ROOT / "config.yaml"

DEFAULT_GENRES = ["poetry", "short_fiction", "oratory", "lyrics"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str          # "stub" | "dryrun" | a real provider key (e.g. "huggingface", "anthropic")
    endpoint: str          # provider-specific endpoint/model-id stub; not resolved until a real client needs it
    max_tokens: int = 800
    temperature: float = 0.9


DEFAULT_MODELS: list[ModelConfig] = [
    ModelConfig(name="qwen3", provider="stub", endpoint="stub://qwen3"),
    ModelConfig(name="llama", provider="stub", endpoint="stub://llama"),
    ModelConfig(name="mistral", provider="stub", endpoint="stub://mistral"),
    # 4th (closed) model: TBD -- advisor decision, not added here.
]


@dataclass(frozen=True)
class GenerationConfig:
    models: list[ModelConfig]
    n_per_cell: Optional[int] = None
    mode_split_ratio: float = 0.5
    genres: list[str] = field(default_factory=lambda: list(DEFAULT_GENRES))
    oratory_enabled: bool = False
    target_length_words: dict[str, Optional[int]] = field(default_factory=dict)

    def active_genres(self) -> list[str]:
        return [g for g in self.genres if g != "oratory" or self.oratory_enabled]

    def refuse_full_run_reason(self) -> Optional[str]:
        """None if a full (non-dry-run, non-smoke) run is allowed to proceed;
        otherwise a human-readable reason to refuse."""
        if self.n_per_cell is None:
            return "n_per_cell is unset (pending advisor decision) -- refusing full run"
        missing_lengths = [
            g for g in self.active_genres()
            if self.target_length_words.get(g) is None
        ]
        if missing_lengths:
            return (
                f"target_length_words unset for genre(s) {missing_lengths} "
                "(pending advisor decision) -- refusing full run"
            )
        return None


def _load_yaml_block() -> dict:
    if not CONFIG_YAML_PATH.exists():
        return {}
    with open(CONFIG_YAML_PATH) as f:
        full = yaml.safe_load(f) or {}
    return full.get("generation", {}) or {}


def load_config() -> GenerationConfig:
    block = _load_yaml_block()

    models_raw = block.get("models")
    if models_raw:
        models = [ModelConfig(**m) for m in models_raw]
    else:
        models = list(DEFAULT_MODELS)

    return GenerationConfig(
        models=models,
        n_per_cell=block.get("n_per_cell"),
        mode_split_ratio=block.get("mode_split_ratio", 0.5),
        genres=block.get("genres") or list(DEFAULT_GENRES),
        oratory_enabled=block.get("oratory_enabled", False),
        target_length_words=block.get("target_length_words") or {},
    )
