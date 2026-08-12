"""Selective parameter unfreezing for the virtual try-on stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch.nn as nn

from swiftedit.models.generator import IPSBV2Model
from swiftedit.vton.garment_encoder import GarmentEncoder

logger = logging.getLogger(__name__)

#: Prompt-branch key/value projections, ``W_y`` in Eq. 9 (25.56 M parameters).
PROMPT_KV_SUFFIXES: tuple[str, ...] = ("attn2.to_k.weight", "attn2.to_v.weight")


@dataclass(frozen=True)
class TrainableGroups:
    """Which parameter groups Stage 1 unfreezes.

    Static inspection of ``ip_adapter.bin`` shows the authors never updated a single
    generator weight - all 686 UNet tensors are bit-identical to ``sbv2_0.5``. The
    defaults here honour that: the only generator tensors that move are the prompt-branch
    K/V (which must adapt from CLIP text statistics to DINOv2 ones) and the widened
    ``conv_in``.

    Attributes:
        prompt_kv: Train ``attn2.to_k`` / ``attn2.to_v``.
        conv_in: Train the widened input convolution.
        image_projection: Train ``image_proj_model``; the agnostic person image is
            out of distribution for a projection fitted on natural photographs.
        image_kv: Train ``to_k_ip`` / ``to_v_ip``. Off by default to preserve the
            learned image prior; the first knob to try if identity is poorly kept.
        garment_projection: Train the garment encoder's projection head.
    """

    prompt_kv: bool = True
    conv_in: bool = True
    image_projection: bool = True
    image_kv: bool = False
    garment_projection: bool = True


def configure_trainable(
    generator: IPSBV2Model,
    garment_encoder: GarmentEncoder,
    groups: TrainableGroups | None = None,
) -> list[nn.Parameter]:
    """Freeze everything, then re-enable the selected groups.

    Args:
        generator: Model whose parameters are being gated.
        garment_encoder: Garment branch; its backbone is left frozen regardless.
        groups: Selection to apply, defaulting to :class:`TrainableGroups`.

    Returns:
        The parameters to hand to the optimiser.
    """
    groups = groups or TrainableGroups()

    generator.requires_grad_(False)
    garment_encoder.requires_grad_(False)

    if groups.prompt_kv:
        for name, param in generator.unet.named_parameters():
            if name.endswith(PROMPT_KV_SUFFIXES):
                param.requires_grad_(True)

    if groups.image_kv:
        for name, param in generator.unet.named_parameters():
            if ".to_k_ip." in name or ".to_v_ip." in name:
                param.requires_grad_(True)

    if groups.conv_in:
        generator.unet.conv_in.requires_grad_(True)

    if groups.image_projection:
        generator.image_proj_model.requires_grad_(True)

    if groups.garment_projection:
        garment_encoder.proj.requires_grad_(True)
        garment_encoder.out_norm.requires_grad_(True)

    seen: set[int] = set()
    trainable: list[nn.Parameter] = []
    for param in list(generator.parameters()) + list(garment_encoder.parameters()):
        # `adapter_modules` aliases tensors that also live under `unet`; de-duplicate so
        # the optimiser does not see the same parameter twice.
        if param.requires_grad and id(param) not in seen:
            seen.add(id(param))
            trainable.append(param)

    logger.info(
        "trainable parameters: %.2f M across %d tensors",
        sum(p.numel() for p in trainable) / 1e6,
        len(trainable),
    )
    return trainable


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters, de-duplicating shared tensors."""
    seen: set[int] = set()
    total = 0
    for param in module.parameters():
        if trainable_only and not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    return total
