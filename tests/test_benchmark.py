import json
from pathlib import Path
import subprocess
import sys

import pytest

from inferedge import benchmark
from inferedge.contracts import BaselineConfig, GenerationRequest


def test_percentiles_have_defined_interpolation():
    summary = benchmark.describe([4.0, 1.0, 3.0, 2.0])
    assert summary == pytest.approx({"min": 1.0, "max": 4.0, "mean": 2.5, "p50": 2.5, "p95": 3.85})
    assert benchmark.describe([2.0])["p95"] == 2.0


@pytest.mark.parametrize("samples", [[], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_timing_samples(samples):
    with pytest.raises(ValueError):
        benchmark.describe(samples)


@pytest.mark.parametrize("kwargs", [
    {"runs": 0}, {"runs": True}, {"warmup": -1}, {"warmup": True},
    {"scenario": "cold", "warmup": 1}, {"scenario": "warm"},
    {"cold_timeout_seconds": 0}, {"cold_timeout_seconds": float("nan")},
])
def test_invalid_experiment(kwargs):
    with pytest.raises(ValueError):
        benchmark.BenchmarkConfig(**kwargs)


def test_warmup_is_excluded_and_samples_match_summary(monkeypatch):
    requests = []
    original = benchmark.NaiveBaseline.generate

    def counted(self, request):
        requests.append(request)
        return original(self, request)

    monkeypatch.setattr(benchmark.NaiveBaseline, "generate", counted)
    report = benchmark.run_benchmark(
        BaselineConfig(), GenerationRequest("hello", 8), benchmark.BenchmarkConfig(runs=3, warmup=2)
    )
    assert len(requests) == 5
    assert len(report["samples"]) == 3
    assert report["synthetic"] is True
    assert report["configuration"]["reload_model_per_request"] is True
    assert report["summary"]["generated_tokens_total"] == 24
    assert report["summary"]["outputs_identical"] is True
    total = [s["timings"]["total_seconds"] for s in report["samples"]]
    assert report["summary"]["total_seconds"] == benchmark.describe(total)
    assert report["summary"]["request_tokens_per_second"] == pytest.approx(24 / sum(total))
    assert all(s["process_wall_seconds"] is None for s in report["samples"])
    assert "prompt" not in report["configuration"]
    assert all("text" not in sample for sample in report["samples"])


@pytest.mark.parametrize("prompt", ["hello", "--looks-like-an-option\n你好"])
def test_cold_runs_use_fresh_subprocesses_and_produce_same_tokens(prompt):
    report = benchmark.run_benchmark(
        BaselineConfig(), GenerationRequest(prompt, 5),
        benchmark.BenchmarkConfig(scenario="cold", runs=2, warmup=0),
    )
    assert len(report["samples"]) == 2
    assert report["summary"]["outputs_identical"] is True
    assert report["summary"]["generated_tokens_total"] == 10
    assert all(s["process_wall_seconds"] >= s["timings"]["total_seconds"] for s in report["samples"])


def test_cold_timeout_fails_experiment(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="child terminated"):
        benchmark.run_benchmark(
            BaselineConfig(), GenerationRequest("hello"),
            benchmark.BenchmarkConfig(scenario="cold", runs=1, warmup=0),
        )


def test_failed_cold_sample_fails_experiment():
    with pytest.raises(RuntimeError, match="cold sample failed"):
        benchmark.run_benchmark(
            BaselineConfig("transformers", Path("/nonexistent/inferedge-model")),
            GenerationRequest("hello"), benchmark.BenchmarkConfig(scenario="cold", runs=1, warmup=0),
        )


def test_failed_experiment_does_not_replace_existing_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("existing report")
    status = benchmark.main([
        "--backend", "transformers", "--model-path", str(tmp_path / "missing"),
        "--output", str(path),
    ])
    assert status == 1
    assert path.read_text() == "existing report"


def test_nonfinite_report_cannot_replace_existing_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("existing report")
    with pytest.raises(ValueError):
        benchmark.write_report({"latency": float("nan")}, path)
    assert path.read_text() == "existing report"


def test_cli_writes_parseable_complete_report(tmp_path):
    output = tmp_path / "nested" / "report.json"
    result = subprocess.run([
        sys.executable, "-m", "inferedge.benchmark", "--backend", "mock",
        "--runs", "2", "--warmup", "0", "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert len(report["samples"]) == 2
    assert report["synthetic"] is True
    assert not list(output.parent.glob("*.tmp"))


def test_invalid_cli_parameters_return_nonzero():
    result = subprocess.run([
        sys.executable, "-m", "inferedge.benchmark", "--runs", "0",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "runs must be a positive integer" in result.stderr
    assert not result.stdout
