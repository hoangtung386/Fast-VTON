"""Round-trip tests for the single-file model bundle."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from transformers import Dinov2Config, Dinov2Model

from swiftedit.models.generator import install_ip_attn_processors
from swiftedit.models.projection import ImageProjModel
from swiftedit.vton.export import bundle_summary, export_bundle, load_bundle
from swiftedit.vton.garment_encoder import GarmentEncoder

CROSS_DIM = 8
CLIP_DIM = 12
NUM_TOKENS = 4


class _StubGenerator(nn.Module):
    """Minimal stand-in exposing the attributes export_bundle reads."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=9,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(8, 8),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=CROSS_DIM,
            attention_head_dim=2,
            norm_num_groups=4,
        )
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=CROSS_DIM,
            clip_embeddings_dim=CLIP_DIM,
            clip_extra_context_tokens=NUM_TOKENS,
        )
        self.adapter_modules = install_ip_attn_processors(
            self.unet, with_mask_controller=True, device="cpu", seed_from_unet=False
        )
        self.aux_model = None


def _tiny_garment_encoder() -> GarmentEncoder:
    config = Dinov2Config(
        hidden_size=CROSS_DIM,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=28,
        patch_size=14,
    )
    return GarmentEncoder(cross_attention_dim=CROSS_DIM, backbone=Dinov2Model(config))


@pytest.fixture
def exported(tmp_path: Path) -> tuple[Path, _StubGenerator, GarmentEncoder]:
    torch.manual_seed(0)
    generator, garment = _StubGenerator(), _tiny_garment_encoder()

    # Move the zero-initialised head off zero so a silent all-zeros round trip would fail.
    with torch.no_grad():
        for param in garment.proj.parameters():
            param.add_(torch.randn_like(param) * 0.05)

    path = export_bundle(
        tmp_path / "model.pt",
        generator=generator,
        garment_encoder=garment,
        step=1234,
        height=512,
        width=384,
        include_frozen=False,
        dtype="fp32",
        null_embedding=torch.randn(1, 77, CROSS_DIM),
    )
    return path, generator, garment


def test_round_trip_preserves_weights(exported) -> None:
    path, generator, garment = exported
    loaded = load_bundle(path, device="cpu")

    assert loaded.manifest.step == 1234
    assert (loaded.manifest.height, loaded.manifest.width) == (512, 384)
    assert loaded.manifest.inpainting_channels == 9
    assert loaded.manifest.includes_frozen is False
    assert loaded.inversion_unet is None

    original = generator.state_dict()
    rebuilt = {
        **{f"unet.{k}": v for k, v in loaded.unet.state_dict().items()},
        **{f"image_proj_model.{k}": v for k, v in loaded.image_proj_model.state_dict().items()},
    }
    for name, tensor in rebuilt.items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)

    for name, tensor in loaded.garment_encoder.state_dict().items():
        torch.testing.assert_close(tensor, garment.state_dict()[name], rtol=0, atol=0)


def test_null_embedding_travels_with_the_bundle(exported) -> None:
    path, _, _ = exported
    loaded = load_bundle(path, device="cpu")
    assert loaded.null_embedding is not None
    assert loaded.null_embedding.shape == (1, 77, CROSS_DIM)


def test_summary_reports_groups_without_rebuilding(exported) -> None:
    path, _, _ = exported
    summary = bundle_summary(path)

    assert set(summary["parameters_by_group"]) == {"generator", "garment_encoder"}
    assert summary["total_parameters"] > 0
    assert summary["file_size_bytes"] == path.stat().st_size
    assert summary["manifest"]["step"] == 1234


def test_include_frozen_requires_inversion_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires inverse_model"):
        export_bundle(
            tmp_path / "x.pt",
            generator=_StubGenerator(),
            garment_encoder=_tiny_garment_encoder(),
            step=0,
            height=512,
            width=384,
            include_frozen=True,
        )


def test_rejects_unknown_dtype(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        export_bundle(
            tmp_path / "x.pt",
            generator=_StubGenerator(),
            garment_encoder=_tiny_garment_encoder(),
            step=0,
            height=512,
            width=384,
            include_frozen=False,
            dtype="int8",
        )
