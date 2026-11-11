"""Intentionally naive: reload the model on EVERY generation request."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

from inferedge.contracts import BaselineConfig, GenerationRequest, GenerationResult, Timings
from inferedge.decoder import decode_greedy
from inferedge.loader import load_model


class NaiveBaseline:
    def __init__(self, config: BaselineConfig):
        self.config = config

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = perf_counter()
        bundle = load_model(self.config)
        loaded = perf_counter()
        output = decode_greedy(bundle, request)
        completed = perf_counter()
        # The runner retains no model, tokenizer, input tensor or generation cache.
        return GenerationResult(
            backend=bundle.backend, model_id=bundle.model_id,
            synthetic=bundle.backend == "mock", text=output.text,
            generated_token_ids=output.token_ids, prompt_tokens=output.prompt_tokens,
            generated_tokens=len(output.token_ids), finish_reason=output.finish_reason,
            timings=Timings(
                loaded - start, output.tokenization_seconds, output.generation_seconds,
                output.text_decode_seconds, completed - start,
            ),
        )


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("mock", "transformers"), default="mock")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--prompt", default="Private inference stays on this machine.")
    parser.add_argument("--max-new-tokens", type=int, default=16)


def config_and_request(args: argparse.Namespace) -> tuple[BaselineConfig, GenerationRequest]:
    return (
        BaselineConfig(args.backend, args.model_path, args.cpu_threads),
        GenerationRequest(args.prompt, args.max_new_tokens),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_generation_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config, request = config_and_request(args)
        result = NaiveBaseline(config).generate(request)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"inferedge-baseline: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
