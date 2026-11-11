from dataclasses import asdict
from pathlib import Path
import weakref

import pytest

from inferedge import baseline
from inferedge.contracts import BaselineConfig, GenerationRequest, ModelDependencyError
from inferedge.loader import load_model


def test_baseline_reloads_and_releases_model_each_request(monkeypatch):
    references = []

    def observed_loader(config):
        bundle = load_model(config)
        references.append(weakref.ref(bundle.model))
        return bundle

    monkeypatch.setattr(baseline, "load_model", observed_loader)
    runner = baseline.NaiveBaseline(BaselineConfig())
    first = runner.generate(GenerationRequest("Private inference", 12))
    second = runner.generate(GenerationRequest("Private inference", 12))
    assert len(references) == 2
    assert all(ref() is None for ref in references)
    assert first.generated_token_ids == second.generated_token_ids
    assert first.text == second.text
    assert first.generated_tokens == 12
    assert first.finish_reason == "length"
    assert first.synthetic is True


def test_mock_is_stable_and_prompt_sensitive():
    runner = baseline.NaiveBaseline(BaselineConfig())
    a = runner.generate(GenerationRequest("hello", 8))
    b = runner.generate(GenerationRequest("world", 8))
    assert a.generated_token_ids != b.generated_token_ids
    assert a.generated_token_ids == (127, 111, 104, 104, 123, 57, 123, 129)
    assert a.text == "|leex6x~"


def test_unicode_prompt_counts_bytes_and_returns_only_new_tokens():
    result = baseline.NaiveBaseline(BaselineConfig()).generate(GenerationRequest("你好🙂", 4))
    assert result.prompt_tokens == 1 + len("你好🙂".encode("utf-8"))
    assert len(result.generated_token_ids) == len(result.text) == 4


def test_timings_include_model_load(monkeypatch):
    monkeypatch.setattr(baseline, "perf_counter", iter([10.0, 13.0, 18.0]).__next__)
    result = baseline.NaiveBaseline(BaselineConfig()).generate(GenerationRequest("hello"))
    assert result.timings.load_seconds == 3.0
    assert result.timings.total_seconds == 8.0


def test_stage_timings_are_nonnegative_and_fit_total():
    result = baseline.NaiveBaseline(BaselineConfig()).generate(GenerationRequest("test"))
    times = asdict(result.timings)
    assert all(t >= 0 for t in times.values())
    assert sum(v for k, v in times.items() if k != "total_seconds") <= times["total_seconds"]


@pytest.mark.parametrize("tokens", [0, -1, True, 1.5])
def test_invalid_output_length(tokens):
    with pytest.raises(ValueError, match="positive integer"):
        GenerationRequest("test", tokens)


@pytest.mark.parametrize("prompt", ["", "  ", None, 42])
def test_invalid_prompt(prompt):
    with pytest.raises(ValueError, match="nonempty"):
        GenerationRequest(prompt)


@pytest.mark.parametrize("kwargs", [
    {"backend": "unknown"}, {"cpu_threads": 0}, {"cpu_threads": True},
    {"backend": "transformers"}, {"model_path": Path("anything")},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BaselineConfig(**kwargs)


def test_missing_local_model_is_an_error_not_a_mock_fallback(tmp_path):
    with pytest.raises(ModelDependencyError, match="local Transformers model"):
        baseline.NaiveBaseline(BaselineConfig("transformers", tmp_path / "missing")).generate(
            GenerationRequest("hello")
        )


def test_pickle_only_model_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"not safe weights")
    with pytest.raises(ModelDependencyError, match="safetensors"):
        load_model(BaselineConfig("transformers", tmp_path))
