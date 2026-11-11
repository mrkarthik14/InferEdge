"""Model construction only. Every invocation returns a new model bundle."""

from dataclasses import dataclass
from typing import Any

from inferedge.contracts import BaselineConfig, ModelDependencyError
from inferedge.mock import MockModel, MockTokenizer


@dataclass
class ModelBundle:
    backend: str
    model_id: str
    tokenizer: Any
    model: Any
    torch: Any = None
    generation_config_type: Any = None
    context_limit: int | None = None


def _context_limit(model: Any, tokenizer: Any) -> int | None:
    candidates = [getattr(tokenizer, "model_max_length", None)]
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        candidates.append(getattr(model.config, name, None))
    # Hugging Face uses enormous sentinel values for an unknown tokenizer limit.
    limits = [v for v in candidates if isinstance(v, int) and 0 < v < 10**9]
    return min(limits) if limits else None


def load_model(config: BaselineConfig) -> ModelBundle:
    if config.backend == "mock":
        return ModelBundle("mock", "deterministic-byte-mock-v1", MockTokenizer(), MockModel())

    path = config.model_path
    if path is None or not path.is_dir() or not (path / "config.json").is_file():
        raise ModelDependencyError(
            "Provide a local Transformers model directory containing config.json, "
            "tokenizer assets and safetensors weights. Model IDs and downloads are unsupported."
        )
    if not any(path.glob("*.safetensors")):
        raise ModelDependencyError("Local model directory has no safetensors weights")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError as exc:
        raise ModelDependencyError(
            "Install the CPU PyTorch wheel and InferEdge's transformers extra; see README.md"
        ) from exc

    torch.set_num_threads(config.cpu_threads)
    # Explicit local paths and local_files_only are required on BOTH loaders.
    # No fallback to a Hub ID, network lookup, remote Python or pickle weights.
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()
    return ModelBundle(
        "transformers", str(path), tokenizer, model, torch, GenerationConfig,
        _context_limit(model, tokenizer),
    )
