"""Package a trained try-on model into a single self-contained file.

Training checkpoints hold only the ~33 M tensors Stage 1 moves, which is right for
resuming but useless on a deployment box that has none of the frozen weights. This
module writes the opposite artefact: one file carrying every module needed for
inference, plus the configs to rebuild them, so the target machine needs no Hugging Face
downloads and no local copy of ``swiftedit_weights/``.

A bundle of state dicts is preferred over ``torch.save(model)``. Pickling a live module
records the fully qualified class path of every submodule, so the file stops loading the
moment a class is renamed or moved - exactly what a refactor does. The bundle survives
that because it only stores tensors and plain configuration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection

from swiftedit.models.generator import (
    IPSBV2Model,
    expand_conv_in,
    install_ip_attn_processors,
)
from swiftedit.models.inversion import InverseModel
from swiftedit.models.projection import ImageProjModel
from swiftedit.vton.garment_encoder import GarmentEncoder

logger = logging.getLogger(__name__)

#: Bump when the payload layout changes incompatibly.
BUNDLE_FORMAT_VERSION = 1

_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class BundleManifest:
    """Everything needed to rebuild the module graph the weights belong to."""

    format_version: int
    step: int
    height: int
    width: int
    inpainting_channels: int
    ip_num_tokens: int
    clip_embeddings_dim: int
    dtype: str
    includes_frozen: bool
    garment_backbone: str
    generator_config: dict[str, Any]
    garment_config: dict[str, Any]
    inversion_config: dict[str, Any] | None = None
    vae_config: dict[str, Any] | None = None
    image_encoder_config: dict[str, Any] | None = None


def _to_dtype(state: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Move a state dict to CPU, casting only floating-point tensors."""
    return {
        name: (tensor.detach().to(dtype) if tensor.is_floating_point() else tensor.detach())
        .cpu()
        .clone()
        for name, tensor in state.items()
    }


def _plain_config(config: Any) -> dict[str, Any]:
    """Convert a diffusers/transformers config object to a JSON-safe dict."""
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return json.loads(json.dumps(raw, default=str))


def export_bundle(
    path: str | Path,
    *,
    generator: IPSBV2Model,
    garment_encoder: GarmentEncoder,
    step: int,
    height: int,
    width: int,
    inverse_model: InverseModel | None = None,
    include_frozen: bool = True,
    dtype: str = "fp16",
    null_embedding: torch.Tensor | None = None,
) -> Path:
    """Write one file holding the whole try-on model.

    Args:
        path: Destination ``.pt`` file. Parent directories are created.
        generator: Trained generator, already widened to the inpainting channel count.
        garment_encoder: Trained garment branch.
        step: Training step the weights come from, recorded in the manifest.
        height: Person image height the model was trained at.
        width: Person image width.
        inverse_model: Inversion network. Required when ``include_frozen`` is set, since
            try-on inference needs it to produce the starting noise.
        include_frozen: Bundle the frozen modules (inversion UNet, VAE, CLIP vision
            tower) too. This is what makes the file self-contained; turning it off
            roughly halves the size but the target machine then needs the original
            checkpoints and Hub access.
        dtype: Storage precision for floating-point tensors.
        null_embedding: Cached empty-prompt embedding to travel with the weights.

    Returns:
        The path written.
    """
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    if include_frozen and inverse_model is None:
        raise ValueError("include_frozen=True requires inverse_model")

    target = torch.device("cpu")
    store_dtype = _DTYPES[dtype]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generator": _to_dtype(generator.state_dict(), store_dtype),
        "garment_encoder": _to_dtype(garment_encoder.state_dict(), store_dtype),
    }

    manifest = BundleManifest(
        format_version=BUNDLE_FORMAT_VERSION,
        step=step,
        height=height,
        width=width,
        inpainting_channels=generator.unet.config.in_channels,
        ip_num_tokens=generator.image_proj_model.clip_extra_context_tokens,
        clip_embeddings_dim=generator.image_proj_model.proj.in_features,
        dtype=dtype,
        includes_frozen=include_frozen,
        garment_backbone=garment_encoder.backbone.config.name_or_path or "",
        generator_config=_plain_config(generator.unet.config),
        garment_config=_plain_config(garment_encoder.backbone.config),
    )

    if include_frozen and inverse_model is not None:
        aux = generator.aux_model
        payload["inversion_unet"] = _to_dtype(inverse_model.unet_inverse.state_dict(), store_dtype)
        payload["vae"] = _to_dtype(aux.vae.state_dict(), store_dtype)
        payload["image_encoder"] = _to_dtype(aux.image_encoder.state_dict(), store_dtype)
        manifest = BundleManifest(
            **{
                **asdict(manifest),
                "inversion_config": _plain_config(inverse_model.unet_inverse.config),
                "vae_config": _plain_config(aux.vae.config),
                "image_encoder_config": _plain_config(aux.image_encoder.config),
            }
        )

    if null_embedding is not None:
        payload["null_embedding"] = null_embedding.detach().to(target, store_dtype)

    payload["manifest"] = json.dumps(asdict(manifest))
    torch.save(payload, path)

    size_gb = path.stat().st_size / 1e9
    tensors = sum(len(v) for v in payload.values() if isinstance(v, dict))
    logger.info(
        "exported %s (%.2f GB, %d tensors, step %d, frozen modules %s)",
        path,
        size_gb,
        tensors,
        step,
        "included" if include_frozen else "excluded",
    )
    return path


@dataclass
class LoadedBundle:
    """Modules rebuilt from a bundle, ready for inference."""

    manifest: BundleManifest
    unet: UNet2DConditionModel
    image_proj_model: nn.Module
    garment_encoder: GarmentEncoder
    inversion_unet: UNet2DConditionModel | None = None
    vae: AutoencoderKL | None = None
    image_encoder: CLIPVisionModelWithProjection | None = None
    null_embedding: torch.Tensor | None = None


def load_bundle(path: str | Path, device: str | torch.device = "cuda") -> LoadedBundle:
    """Rebuild every module stored by :func:`export_bundle`.

    No Hugging Face access and no local checkpoint directory is required when the bundle
    was written with ``include_frozen=True``.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    manifest = BundleManifest(**json.loads(payload["manifest"]))
    if manifest.format_version != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"bundle format v{manifest.format_version} is not supported by this build "
            f"(expected v{BUNDLE_FORMAT_VERSION})"
        )

    device = torch.device(device)
    dtype = _DTYPES[manifest.dtype]

    unet = UNet2DConditionModel.from_config(manifest.generator_config)
    if manifest.inpainting_channels != unet.config.in_channels:
        expand_conv_in(unet, manifest.inpainting_channels)
    install_ip_attn_processors(unet, with_mask_controller=True, device="cpu", seed_from_unet=False)

    projection = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=manifest.clip_embeddings_dim,
        clip_extra_context_tokens=manifest.ip_num_tokens,
    )

    holder = nn.Module()
    holder.unet = unet
    holder.image_proj_model = projection
    holder.adapter_modules = nn.ModuleList(unet.attn_processors.values())
    holder.load_state_dict(payload["generator"])

    garment_encoder = GarmentEncoder.from_config(
        manifest.garment_config, cross_attention_dim=unet.config.cross_attention_dim
    )
    garment_encoder.load_state_dict(payload["garment_encoder"])

    bundle = LoadedBundle(
        manifest=manifest,
        unet=unet.to(device, dtype).eval(),
        image_proj_model=projection.to(device, dtype).eval(),
        garment_encoder=garment_encoder.to(device, dtype).eval(),
        null_embedding=payload.get("null_embedding"),
    )

    if manifest.includes_frozen:
        inversion = UNet2DConditionModel.from_config(manifest.inversion_config)
        inversion.load_state_dict(payload["inversion_unet"])
        bundle.inversion_unet = inversion.to(device, dtype).eval()

        vae = AutoencoderKL.from_config(manifest.vae_config)
        vae.load_state_dict(payload["vae"])
        bundle.vae = vae.to(device, torch.float32).eval()

        encoder = CLIPVisionModelWithProjection(
            CLIPVisionModelWithProjection.config_class(**manifest.image_encoder_config)
        )
        encoder.load_state_dict(payload["image_encoder"])
        bundle.image_encoder = encoder.to(device, dtype).eval()

    if bundle.null_embedding is not None:
        bundle.null_embedding = bundle.null_embedding.to(device, dtype)

    logger.info(
        "loaded bundle from step %d at %dx%d", manifest.step, manifest.height, manifest.width
    )
    return bundle


def bundle_summary(path: str | Path) -> dict[str, Any]:
    """Read a bundle's manifest and per-group sizes without rebuilding any module."""
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    manifest = json.loads(payload["manifest"])
    groups = {
        name: sum(t.numel() for t in value.values())
        for name, value in payload.items()
        if isinstance(value, dict)
    }
    return {
        "manifest": manifest,
        "parameters_by_group": groups,
        "total_parameters": sum(groups.values()),
        "file_size_bytes": Path(path).stat().st_size,
    }
