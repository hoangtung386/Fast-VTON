"""Offline feature cache for Stage 1.

Every module Stage 1 keeps frozen - the VAE, the inversion network, the CLIP vision
tower and the DINOv2 backbone - sees inputs that never change. Running them once and
caching the result removes an entire UNet forward pass from the training step and frees
the inversion network's memory.

Only frozen outputs are cached. The IP projection and the garment projection are
trainable, so the cache stops at their inputs: pooled CLIP embeddings and raw DINOv2
patch features respectively.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.constants import INVERSION_TIMESTEP
from src.models.auxiliary import AuxiliaryModel
from src.models.inversion import InverseModel
from src.vton.config import DataConfig
from src.vton.garment_encoder import GarmentEncoder, pad_to_square
from src.vton.masking import build_agnostic_mask

logger = logging.getLogger(__name__)

CACHE_META_FILE = "meta.json"


@dataclass(frozen=True)
class CacheSpec:
    """Name, per-sample shape and dtype of one cached array."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float16"

    @property
    def filename(self) -> str:
        """File the array is stored in."""
        return f"{self.name}.npy"


def cache_specs(
    data: DataConfig,
    clip_dim: int,
    garment_tokens: int | None,
    garment_dim: int,
) -> list[CacheSpec]:
    """Describe every array the cache holds for the given configuration."""
    latent = (4, data.latent_height, data.latent_width)
    specs = [
        CacheSpec("z_person", latent),
        CacheSpec("z_agnostic", latent),
        CacheSpec("inverted_noise", latent),
        CacheSpec("mask_latent", (1, data.latent_height, data.latent_width)),
        CacheSpec("clip_image_embeds", (clip_dim,)),
    ]
    if garment_tokens is not None:
        specs.append(CacheSpec("garment_features", (garment_tokens, garment_dim)))
    return specs


def _to_model_input(
    image: Image.Image, size: tuple[int, int], device: torch.device
) -> torch.Tensor:
    """Resize a PIL image and map it to a ``[-1, 1]`` tensor of shape ``(1, 3, H, W)``."""
    resized = image.convert("RGB").resize(size, Image.BILINEAR)
    return to_tensor(resized).unsqueeze(0).to(device) * 2 - 1


def _batched(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    """Yield consecutive chunks of ``size`` items."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


@torch.no_grad()
def build_cache(
    dataset: Any,
    output_dir: Path,
    inverse_model: InverseModel,
    aux_model: AuxiliaryModel,
    data: DataConfig,
    null_embedding: torch.Tensor,
    garment_encoder: GarmentEncoder | None = None,
    batch_size: int = 8,
) -> Path:
    """Materialise the Stage 1 cache as a directory of ``.npy`` memmaps.

    Args:
        dataset: Indexable dataset yielding ``image``, ``agnostic`` and ``cloth`` PIL
            images - the column layout of ``forgeml/viton_hd``.
        output_dir: Destination directory, created if absent.
        inverse_model: Frozen inversion network (also supplies the VAE).
        aux_model: Frozen CLIP vision tower.
        data: Resolution and mask-building settings.
        null_embedding: Cached empty-prompt embedding, ``(1, 77, 1024)``. Must come from
            the text encoder the inversion network was trained against - ``sd-turbo``,
            not SD 2.1-base.
        garment_encoder: When given, DINOv2 features are cached too. Costs about 6 GB
            for VITON-HD but removes the backbone from the training step entirely.
        batch_size: Samples processed per forward pass.

    Returns:
        ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    device = inverse_model.device

    processor = None
    garment_tokens = None
    garment_dim = 0
    if garment_encoder is not None:
        processor = GarmentEncoder.image_processor(resolution=data.garment_resolution)
        garment_dim = garment_encoder.backbone.config.hidden_size
        patch = garment_encoder.backbone.config.patch_size
        garment_tokens = 1 + (data.garment_resolution // patch) ** 2

    clip_dim = aux_model.image_encoder.config.projection_dim
    specs = cache_specs(data, clip_dim, garment_tokens, garment_dim)

    arrays = {
        spec.name: np.lib.format.open_memmap(
            output_dir / spec.filename,
            mode="w+",
            dtype=np.dtype(spec.dtype),
            shape=(total, *spec.shape),
        )
        for spec in specs
    }

    timestep = torch.full((1,), INVERSION_TIMESTEP, dtype=torch.int64, device=device)
    null_embedding = null_embedding.to(device, dtype=inverse_model.weight_dtype)

    for offset, indices in enumerate(_batched(range(total), batch_size)):
        samples = [dataset[i] for i in indices]
        size = data.pil_size

        person = torch.cat([_to_model_input(s["image"], size, device) for s in samples], dim=0)
        agnostic = torch.cat([_to_model_input(s["agnostic"], size, device) for s in samples], dim=0)
        masks = np.stack(
            [
                build_agnostic_mask(
                    s["image"],
                    s["agnostic"],
                    size,
                    data.mask_diff_threshold,
                    data.mask_morph_kernel,
                )
                for s in samples
            ]
        )
        mask_tensor = torch.from_numpy(masks).unsqueeze(1).float().to(device)

        z_person = inverse_model.encode_image(person)
        z_agnostic = inverse_model.encode_image(agnostic)
        inverted = inverse_model.invert(
            z_agnostic,
            null_embedding.expand(z_agnostic.shape[0], -1, -1),
            timestep,
        )
        mask_latent = torch.nn.functional.interpolate(
            mask_tensor, z_person.shape[-2:], mode="nearest"
        )
        clip_embeds = aux_model.encode_clip_image([s["agnostic"] for s in samples])

        batch_output = {
            "z_person": z_person,
            "z_agnostic": z_agnostic,
            "inverted_noise": inverted,
            "mask_latent": mask_latent,
            "clip_image_embeds": clip_embeds,
        }

        if garment_encoder is not None and processor is not None:
            # Letterbox first: the stock DINOv2 crop would cut the neckline and hem off
            # a 768x1024 product shot, and the encoder cannot infer what it never saw.
            pixel_values = processor(
                images=[pad_to_square(s["cloth"]) for s in samples], return_tensors="pt"
            ).pixel_values.to(device)
            batch_output["garment_features"] = garment_encoder.encode_frozen(pixel_values)

        start = offset * batch_size
        for name, tensor in batch_output.items():
            arrays[name][start : start + len(samples)] = (
                tensor.detach().to(torch.float16).cpu().numpy()
            )

        if offset % 50 == 0:
            logger.info("cached %d / %d samples", start + len(samples), total)

    for array in arrays.values():
        array.flush()

    meta = {
        "num_samples": total,
        "height": data.height,
        "width": data.width,
        "latent_height": data.latent_height,
        "latent_width": data.latent_width,
        "arrays": {spec.name: {"shape": list(spec.shape), "dtype": spec.dtype} for spec in specs},
    }
    (output_dir / CACHE_META_FILE).write_text(json.dumps(meta, indent=2))
    logger.info("cache written to %s", output_dir)
    return output_dir
