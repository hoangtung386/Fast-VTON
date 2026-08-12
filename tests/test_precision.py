"""Tests for picking an autocast dtype the GPU on hand can run."""

import pytest
import torch

from src.vton import resolve_mixed_precision


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch):
    """Pretend a CUDA device is present, with bf16 support under our control."""

    def _configure(*, bf16: bool) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)

    return _configure


def test_auto_picks_bf16_on_ampere(gpu) -> None:
    gpu(bf16=True)
    assert resolve_mixed_precision("auto", "cuda") == "bf16"


def test_auto_falls_back_to_fp16_on_turing(gpu) -> None:
    """A T4 has no bf16; auto must degrade rather than crash at the first step."""
    gpu(bf16=False)
    assert resolve_mixed_precision("auto", "cuda") == "fp16"


def test_auto_disables_autocast_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_mixed_precision("auto", "cpu") == "no"


def test_explicit_bf16_on_turing_fails_before_the_weights_load(gpu) -> None:
    """The whole point: complain at the CLI, not minutes later inside autocast."""
    gpu(bf16=False)
    with pytest.raises(ValueError, match=r"compute capability 8\.0"):
        resolve_mixed_precision("bf16", "cuda")


def test_explicit_bf16_passes_through_on_ampere(gpu) -> None:
    gpu(bf16=True)
    assert resolve_mixed_precision("bf16", "cuda") == "bf16"


@pytest.mark.parametrize("requested", ["no", "fp16"])
def test_explicit_choices_are_never_overridden(gpu, requested: str) -> None:
    gpu(bf16=True)
    assert resolve_mixed_precision(requested, "cuda") == requested


def test_rejects_unknown_request() -> None:
    with pytest.raises(ValueError, match="mixed precision must be one of"):
        resolve_mixed_precision("int4", "cuda")


def test_resolved_value_is_always_accepted_by_the_config(gpu) -> None:
    """Stage1Config validates this field; auto must never leak through to it."""
    from src.vton import Stage1Config

    for bf16 in (True, False):
        gpu(bf16=bf16)
        Stage1Config(mixed_precision=resolve_mixed_precision("auto", "cuda"))
