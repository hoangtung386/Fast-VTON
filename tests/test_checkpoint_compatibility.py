"""Guard the refactor against silently breaking ``ip_adapter.bin`` compatibility.

The released checkpoint is a flat ``state_dict`` keyed by module path, so renaming or
re-nesting any submodule breaks loading. These tests rebuild the module graph on the
meta device - no weights are allocated and no model is executed - and compare key sets
against the checkpoint on disk.
"""

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from diffusers import UNet2DConditionModel

from src.constants import CROSS_ATTENTION_DIM, IP_NUM_TOKENS
from src.models.generator import attention_hidden_size, install_ip_attn_processors
from src.models.projection import ImageProjModel
from src.vton.config import CheckpointConfig

CHECKPOINTS = CheckpointConfig()
HAVE_WEIGHTS = CHECKPOINTS.generator_dir.exists() and CHECKPOINTS.ip_adapter_path.exists()
needs_weights = pytest.mark.skipif(not HAVE_WEIGHTS, reason="SwiftEdit checkpoints not present")


def _empty_generator_graph(with_mask_controller: bool) -> nn.Module:
    """Build the IPSBV2Model submodule graph on the meta device."""
    config = UNet2DConditionModel.load_config(str(CHECKPOINTS.generator_dir))
    with init_empty_weights():
        unet = UNet2DConditionModel.from_config(config)
        projection = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=CROSS_ATTENTION_DIM,
            clip_extra_context_tokens=IP_NUM_TOKENS,
        )
        adapters = install_ip_attn_processors(
            unet, with_mask_controller=with_mask_controller, device="meta", seed_from_unet=False
        )

        graph = nn.Module()
        graph.unet = unet
        graph.image_proj_model = projection
        graph.adapter_modules = adapters
    return graph


@needs_weights
@pytest.mark.parametrize("with_mask_controller", [False, True])
def test_state_dict_keys_match_released_checkpoint(with_mask_controller: bool) -> None:
    checkpoint = torch.load(
        str(CHECKPOINTS.ip_adapter_path), map_location="cpu", mmap=True, weights_only=True
    )
    graph = _empty_generator_graph(with_mask_controller)

    assert set(graph.state_dict()) == set(checkpoint), (
        "module graph no longer matches ip_adapter.bin; loading would fail"
    )


@needs_weights
def test_adapter_modules_alias_unet_processors() -> None:
    """The duplicate key groups must reference the same tensors, not copies."""
    graph = _empty_generator_graph(with_mask_controller=True)
    state = graph.state_dict()

    unet_ip = sorted(k for k in state if k.startswith("unet.") and ".to_k_ip." in k)
    alias_ip = sorted(k for k in state if k.startswith("adapter_modules.") and ".to_k_ip." in k)

    assert len(unet_ip) == 16, "SD 2.1-base has 16 cross-attention sites"
    assert len(alias_ip) == len(unet_ip)

    processors = list(graph.unet.attn_processors.values())
    assert all(a is b for a, b in zip(processors, graph.adapter_modules, strict=True))


@needs_weights
def test_hidden_sizes_follow_block_layout() -> None:
    config = UNet2DConditionModel.load_config(str(CHECKPOINTS.generator_dir))
    with init_empty_weights():
        unet = UNet2DConditionModel.from_config(config)

    sizes = {
        name: attention_hidden_size(unet, name)
        for name in unet.attn_processors
        if not name.endswith("attn1.processor")
    }
    assert set(sizes.values()) <= set(unet.config.block_out_channels)
    assert sizes["mid_block.attentions.0.transformer_blocks.0.attn2.processor"] == 1280


def test_unknown_processor_name_is_rejected() -> None:
    with init_empty_weights():
        unet = UNet2DConditionModel(
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
        )
    with pytest.raises(ValueError, match="cannot infer hidden size"):
        attention_hidden_size(unet, "unknown_block.attn2.processor")
