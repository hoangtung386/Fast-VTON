"""Tests for the virtual try-on configuration objects."""

from pathlib import Path

import pytest

from swiftedit.constants import VAE_SCALE_FACTOR
from swiftedit.vton.config import CheckpointConfig, DataConfig, Stage1Config


def test_latent_dimensions_match_vae_downsampling() -> None:
    data = DataConfig(height=512, width=384)
    assert data.latent_height == 512 // VAE_SCALE_FACTOR
    assert data.latent_width == 384 // VAE_SCALE_FACTOR
    assert data.pil_size == (384, 512)


def test_rejects_resolution_not_divisible_by_eight() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        DataConfig(height=513, width=384)


def test_effective_batch_size() -> None:
    config = Stage1Config(batch_size=8, gradient_accumulation_steps=2)
    assert config.effective_batch_size == 16


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mixed_precision": "fp8"}, "mixed_precision"),
        ({"warmup_fraction": 1.5}, "warmup_fraction"),
        ({"batch_size": 0}, "must be >= 1"),
    ],
)
def test_rejects_invalid_hyperparameters(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Stage1Config(**kwargs)


def test_checkpoint_paths_are_derived_from_root() -> None:
    checkpoints = CheckpointConfig(root=Path("/weights"))
    assert checkpoints.inversion_dir == Path("/weights/inverse_ckpt-120k")
    assert checkpoints.generator_dir == Path("/weights/sbv2_0.5")
    assert checkpoints.ip_adapter_path == Path("/weights/ip_adapter_ckpt-90k/ip_adapter.bin")


def test_validate_reports_missing_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing SwiftEdit checkpoints"):
        CheckpointConfig(root=tmp_path).validate()
