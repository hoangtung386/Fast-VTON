"""Tests that widening ``conv_in`` leaves the network's behaviour untouched."""

import pytest
import torch
from diffusers import UNet2DConditionModel

from swiftedit.models.generator import expand_conv_in


@pytest.fixture(scope="module")
def tiny_unet() -> UNet2DConditionModel:
    """A minimal cross-attention UNet that runs quickly on CPU."""
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 8),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=4,
    ).eval()


def test_zero_init_preserves_output(tiny_unet: UNet2DConditionModel) -> None:
    torch.manual_seed(1)
    latent = torch.randn(1, 4, 8, 8)
    condition = torch.randn(1, 3, 8)
    timestep = torch.tensor([1])

    with torch.no_grad():
        before = tiny_unet(latent, timestep, condition).sample

    expand_conv_in(tiny_unet, 9)
    padded = torch.cat([latent, torch.zeros(1, 5, 8, 8)], dim=1)
    with torch.no_grad():
        after = tiny_unet(padded, timestep, condition).sample

    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert tiny_unet.config.in_channels == 9
    assert tiny_unet.conv_in.weight.shape[1] == 9
    assert torch.all(tiny_unet.conv_in.weight[:, 4:] == 0)


def test_expansion_is_idempotent(tiny_unet: UNet2DConditionModel) -> None:
    weight = tiny_unet.conv_in.weight.clone()
    expand_conv_in(tiny_unet, 9)
    torch.testing.assert_close(tiny_unet.conv_in.weight, weight, rtol=0, atol=0)


def test_shrinking_is_rejected(tiny_unet: UNet2DConditionModel) -> None:
    with pytest.raises(ValueError, match="cannot shrink conv_in"):
        expand_conv_in(tiny_unet, 4)
