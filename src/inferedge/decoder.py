"""Uncached greedy reference. A custom token loop is deferred to phase 2."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from inferedge.contracts import GenerationRequest
from inferedge.loader import ModelBundle


@dataclass(frozen=True)
class DecodedOutput:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    finish_reason: Literal["length", "eos"]
    tokenization_seconds: float
    generation_seconds: float
    text_decode_seconds: float


def decode_greedy(bundle: ModelBundle, request: GenerationRequest) -> DecodedOutput:
    start = perf_counter()
    if bundle.backend == "mock":
        inputs = bundle.tokenizer.encode(request.prompt)
        prompt_tokens = len(inputs)
    else:
        inputs = bundle.tokenizer(request.prompt, return_tensors="pt", add_special_tokens=True)
        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[-1]
    if prompt_tokens == 0:
        raise ValueError("tokenizer produced an empty prompt")
    if bundle.context_limit is not None and prompt_tokens + request.max_new_tokens > bundle.context_limit:
        raise ValueError(
            f"prompt ({prompt_tokens}) + output ({request.max_new_tokens}) exceeds "
            f"model context ({bundle.context_limit})"
        )
    tokenized = perf_counter()

    eos = bundle.tokenizer.eos_token_id
    if bundle.backend == "mock":
        output_ids = bundle.model.generate(inputs, request.max_new_tokens)
    else:
        if eos is None:
            eos = getattr(bundle.model.generation_config, "eos_token_id", None)
        pad = bundle.tokenizer.pad_token_id
        if pad is None and eos is not None:
            pad = eos[0] if isinstance(eos, (list, tuple)) else eos
        # A fresh config avoids inheriting sampling/beam/repetition settings from
        # the model's generation_config.json. Only EOS may end generation early.
        generation_config = bundle.generation_config_type(
            max_new_tokens=request.max_new_tokens, do_sample=False, num_beams=1,
            use_cache=False, eos_token_id=eos, pad_token_id=pad,
        )
        with bundle.torch.inference_mode():
            output = bundle.model.generate(**inputs, generation_config=generation_config)
        output_ids = output[0].tolist()
    generated = perf_counter()

    token_ids = output_ids[prompt_tokens:]
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    finish_reason = "eos" if token_ids and token_ids[-1] in eos_ids else "length"
    if bundle.backend == "mock":
        text = bundle.tokenizer.decode(token_ids)
    else:
        text = bundle.tokenizer.decode(token_ids, skip_special_tokens=True)
    decoded = perf_counter()
    return DecodedOutput(
        text, tuple(token_ids), prompt_tokens, finish_reason,
        tokenized - start, generated - tokenized, decoded - generated,
    )
