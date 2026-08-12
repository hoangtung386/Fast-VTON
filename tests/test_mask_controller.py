"""Tests for ARaM grid inference and mask resampling."""

import pytest
import torch

from src.attention.mask_controller import MaskController


def _controller(height: int, width: int) -> MaskController:
    return MaskController(torch.zeros(height, width))


@pytest.mark.parametrize(
    ("shape", "num_tokens", "expected"),
    [
        ((64, 64), 64 * 64, (64, 64)),
        ((64, 64), 32 * 32, (32, 32)),
        ((64, 64), 8 * 8, (8, 8)),
        # 512x384 latents: the case the original int(sqrt(n)) could not express.
        ((64, 48), 64 * 48, (64, 48)),
        ((64, 48), 32 * 24, (32, 24)),
        ((64, 48), 8 * 6, (8, 6)),
    ],
)
def test_spatial_shape(shape: tuple[int, int], num_tokens: int, expected: tuple[int, int]) -> None:
    assert _controller(*shape).spatial_shape(num_tokens) == expected


def test_spatial_shape_rejects_incompatible_token_count() -> None:
    with pytest.raises(ValueError, match="cannot map"):
        _controller(64, 48).spatial_shape(1000)


def test_resized_mask_preserves_binary_values() -> None:
    mask = torch.zeros(64, 48)
    mask[16:48, 12:36] = 1.0
    controller = MaskController(mask)

    resized = controller.resized_mask(32 * 24)

    assert resized.shape == (32 * 24, 1)
    assert set(resized.unique().tolist()) == {0.0, 1.0}, "nearest resampling must stay binary"
    assert resized.mean().item() == pytest.approx(mask.mean().item(), abs=0.02)


def test_rejects_non_2d_mask() -> None:
    with pytest.raises(ValueError, match="2-D latent grid"):
        MaskController(torch.zeros(1, 1, 64, 48))
