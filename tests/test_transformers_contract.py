"""Contract tests using test doubles; these do not validate actual model inference."""

from contextlib import contextmanager
import sys
from types import SimpleNamespace

import pytest

from inferedge.baseline import NaiveBaseline
from inferedge.contracts import BaselineConfig, GenerationRequest
from inferedge.decoder import decode_greedy
from inferedge.loader import load_model


@pytest.fixture
def fake_ml_stack(monkeypatch, tmp_path):
    calls = {}

    class Tensor:
        shape = (1, 2)

        def to(self, device):
            calls.setdefault("tensor_devices", []).append(device)
            return self

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = None
        model_max_length = 10**30

        def __call__(self, prompt, **kwargs):
            calls["tokenization"] = kwargs
            return {"input_ids": Tensor(), "attention_mask": Tensor()}

        def decode(self, tokens, **kwargs):
            calls["decoded"] = (tokens, kwargs)
            return "answer"

    class Model:
        config = SimpleNamespace(max_position_embeddings=32)
        generation_config = SimpleNamespace(eos_token_id=2)

        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def generate(self, **kwargs):
            assert calls["inference_active"] is True
            calls["generation"] = kwargs
            return [SimpleNamespace(tolist=lambda: [10, 11, 42, 2])]

    tokenizer, model = Tokenizer(), Model()

    def tokenizer_load(path, **kwargs):
        calls["tokenizer_load"] = (path, kwargs)
        return tokenizer

    def model_load(path, **kwargs):
        calls["model_load"] = (path, kwargs)
        return model

    @contextmanager
    def inference_mode():
        calls["inference_active"] = True
        try:
            yield
        finally:
            calls["inference_active"] = False

    torch = SimpleNamespace(
        float32="float32", set_num_threads=lambda n: calls.update(threads=n),
        inference_mode=inference_mode,
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_load),
        GenerationConfig=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").touch()
    return BaselineConfig("transformers", tmp_path, cpu_threads=2), calls


def test_loaders_are_local_only_cpu_eval_and_safe(fake_ml_stack):
    config, calls = fake_ml_stack
    bundle = load_model(config)
    assert calls["tokenizer_load"] == (str(config.model_path), {
        "local_files_only": True, "trust_remote_code": False,
    })
    assert calls["model_load"] == (str(config.model_path), {
        "local_files_only": True, "trust_remote_code": False,
        "use_safetensors": True, "torch_dtype": "float32",
    })
    assert calls["threads"] == 2
    assert calls["eval"] is True
    assert calls["model_device"] == "cpu"
    assert bundle.context_limit == 32


def test_uncached_greedy_output_excludes_prompt_and_counts_eos(fake_ml_stack):
    config, calls = fake_ml_stack
    result = NaiveBaseline(config).generate(GenerationRequest("hello", 3))
    settings = calls["generation"]["generation_config"]
    assert vars(settings) == {
        "max_new_tokens": 3, "do_sample": False, "num_beams": 1,
        "use_cache": False, "eos_token_id": 2, "pad_token_id": 2,
    }
    assert calls["tensor_devices"] == ["cpu", "cpu"]
    assert "attention_mask" in calls["generation"]
    assert result.generated_token_ids == (42, 2)
    assert result.generated_tokens == 2
    assert result.prompt_tokens == 2
    assert result.finish_reason == "eos"
    assert result.synthetic is False
    assert calls["decoded"] == ([42, 2], {"skip_special_tokens": True})
    assert calls["inference_active"] is False


def test_known_model_context_overflow_is_rejected_before_generation(fake_ml_stack):
    config, calls = fake_ml_stack
    bundle = load_model(config)
    with pytest.raises(ValueError, match="exceeds model context"):
        decode_greedy(bundle, GenerationRequest("hello", 31))
    assert "generation" not in calls


def test_eos_fallback_from_model_config(fake_ml_stack):
    config, calls = fake_ml_stack
    bundle = load_model(config)
    bundle.tokenizer.eos_token_id = None
    result = decode_greedy(bundle, GenerationRequest("hello", 3))
    assert calls["generation"]["generation_config"].eos_token_id == 2
    assert result.finish_reason == "eos"
