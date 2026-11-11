"""Sequential baseline experiments. No resident-model or concurrent serving yet."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from inferedge import __version__
from inferedge.baseline import NaiveBaseline, add_generation_arguments, config_and_request
from inferedge.contracts import BaselineConfig, GenerationRequest, positive_int


@dataclass(frozen=True)
class BenchmarkConfig:
    scenario: Literal["repeated", "cold"] = "repeated"
    runs: int = 5
    warmup: int = 1
    cold_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.scenario not in ("repeated", "cold"):
            raise ValueError("scenario must be repeated or cold")
        positive_int("runs", self.runs)
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        if self.scenario == "cold" and self.warmup:
            raise ValueError("cold runs require warmup=0")
        if not math.isfinite(self.cold_timeout_seconds) or self.cold_timeout_seconds <= 0:
            raise ValueError("cold timeout must be finite and positive")


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation at zero-based rank (n - 1) * q."""
    if not values or not 0 <= q <= 1:
        raise ValueError("percentile needs samples and q in [0, 1]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def describe(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("timings must be finite, nonnegative and nonempty")
    return {
        "min": min(values), "max": max(values), "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
    }


def _machine_metadata() -> dict[str, Any]:
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    cpu_model = platform.processor() or None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    quota = None
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        pass
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "architecture": platform.machine(), "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": quota, "inferedge_version": __version__, "packages": versions,
    }


def _cold_request(config: BaselineConfig, request: GenerationRequest, timeout: float) -> tuple[dict, float]:
    command = [
        sys.executable, "-m", "inferedge.baseline", "--backend", config.backend,
        "--cpu-threads", str(config.cpu_threads), "--prompt=" + request.prompt,
        "--max-new-tokens", str(request.max_new_tokens),
    ]
    if config.model_path is not None:
        command.extend(["--model-path", str(config.model_path)])
    start = perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"cold sample exceeded {timeout:g}s; child terminated") from exc
    elapsed = perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"cold sample failed (exit {completed.returncode}): {completed.stderr.strip()}")
    return json.loads(completed.stdout), elapsed


def _sample(result: dict, index: int, process_seconds: float | None) -> dict[str, Any]:
    return {
        "index": index, "prompt_tokens": result["prompt_tokens"],
        "generated_tokens": result["generated_tokens"], "finish_reason": result["finish_reason"],
        "text_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(json.dumps(result["generated_token_ids"]).encode()).hexdigest(),
        "timings": result["timings"], "process_wall_seconds": process_seconds,
    }


def run_benchmark(
    config: BaselineConfig, request: GenerationRequest, experiment: BenchmarkConfig,
) -> dict[str, Any]:
    runner = NaiveBaseline(config)
    for _ in range(experiment.warmup):
        runner.generate(request)
    samples = []
    start = perf_counter()
    for index in range(experiment.runs):
        if experiment.scenario == "cold":
            result, process_seconds = _cold_request(config, request, experiment.cold_timeout_seconds)
        else:
            result, process_seconds = asdict(runner.generate(request)), None
        samples.append(_sample(result, index, process_seconds))
    measurement_wall_seconds = perf_counter() - start
    summary: dict[str, Any] = {}
    for name in ("load_seconds", "tokenization_seconds", "generation_seconds", "text_decode_seconds", "total_seconds"):
        summary[name] = describe([s["timings"][name] for s in samples])
    if experiment.scenario == "cold":
        summary["process_wall_seconds"] = describe([s["process_wall_seconds"] for s in samples])
    generated = sum(s["generated_tokens"] for s in samples)
    for label, timing in (("generation_only_tokens_per_second", "generation_seconds"),
                          ("request_tokens_per_second", "total_seconds")):
        elapsed = sum(s["timings"][timing] for s in samples)
        summary[label] = generated / elapsed if elapsed > 0 else None
    summary["generated_tokens_total"] = generated
    summary["outputs_identical"] = len({s["token_ids_sha256"] for s in samples}) == 1
    return {
        "schema_version": 1, "benchmark_id": str(uuid4()),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "backend": config.backend, "model_id": result["model_id"],
        "synthetic": config.backend == "mock",
        "warning": "Synthetic mock timings are NOT language-model performance." if config.backend == "mock" else None,
        "machine": _machine_metadata(),
        "configuration": {
            **asdict(experiment), "cpu_threads": config.cpu_threads, "device": "cpu",
            "dtype": "float32" if config.backend == "transformers" else None,
            "use_cache": False, "do_sample": False, "reload_model_per_request": True,
            "max_new_tokens": request.max_new_tokens,
            "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
            "prompt_utf8_bytes": len(request.prompt.encode()),
        },
        "timing_clock": "time.perf_counter",
        "percentile_method": "linear interpolation at (n - 1) * q",
        "measurement_wall_seconds": measurement_wall_seconds,
        "samples": samples, "summary": summary,
    }


def write_report(report: dict, path: Path) -> None:
    # Replace only after a full successful run and valid JSON serialization.
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_generation_arguments(parser)
    parser.add_argument("--scenario", choices=("repeated", "cold"), default="repeated")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, help="default: repeated=1, cold=0; weights always reload")
    parser.add_argument("--cold-timeout-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, help="write complete JSON atomically; otherwise stdout")
    args = parser.parse_args(argv)
    try:
        config, request = config_and_request(args)
        warmup = args.warmup if args.warmup is not None else (1 if args.scenario == "repeated" else 0)
        experiment = BenchmarkConfig(args.scenario, args.runs, warmup, args.cold_timeout_seconds)
        report = run_benchmark(config, request, experiment)
        if args.output is not None:
            write_report(report, args.output)
            print(f"Wrote {len(report['samples'])} {config.backend} samples to {args.output}")
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"inferedge-benchmark: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
