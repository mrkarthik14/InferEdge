"""Opt-in actual-model check; never downloads or substitutes synthetic weights."""

import os
from pathlib import Path
import socket

import pytest

from inferedge.baseline import NaiveBaseline
from inferedge.contracts import BaselineConfig, GenerationRequest


@pytest.mark.local_model
def test_local_model_offline_repeatability(monkeypatch):
    path = os.environ.get("INFEREDGE_TEST_MODEL_PATH")
    if not path:
        pytest.skip("INFEREDGE_TEST_MODEL_PATH is unset; no local model was supplied")
    # If explicitly requested, missing dependencies or bad assets FAIL, not skip.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def deny_network(*args, **kwargs):
        raise AssertionError("offline inference attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    runner = NaiveBaseline(BaselineConfig("transformers", Path(path)))
    request = GenerationRequest("Private inference", 4)
    first, second = runner.generate(request), runner.generate(request)
    assert first.generated_token_ids == second.generated_token_ids
    assert 0 < first.generated_tokens <= 4
    assert first.synthetic is False
