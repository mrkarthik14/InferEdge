"""Typed internal contracts for the phase-1 experiment (not HTTP schemas)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class ModelDependencyError(RuntimeError):
    """Local model assets or optional ML dependencies are unavailable."""


def positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BaselineConfig:
    backend: Literal["mock", "transformers"] = "mock"
    model_path: Path | None = None
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if self.backend not in ("mock", "transformers"):
            raise ValueError("backend must be mock or transformers")
        positive_int("cpu_threads", self.cpu_threads)
        if self.backend == "transformers" and self.model_path is None:
            raise ValueError("transformers requires an explicit local --model-path")
        if self.backend == "mock" and self.model_path is not None:
            raise ValueError("--model-path is only valid for the transformers backend")
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path).expanduser().resolve())


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        positive_int("max_new_tokens", self.max_new_tokens)


@dataclass(frozen=True)
class Timings:
    load_seconds: float
    tokenization_seconds: float
    generation_seconds: float
    text_decode_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class GenerationResult:
    backend: str
    model_id: str
    synthetic: bool
    text: str
    generated_token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: Literal["length", "eos"]
    timings: Timings
