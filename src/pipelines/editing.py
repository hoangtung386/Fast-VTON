"""Text-guided one-step editing: self-guided mask extraction plus ARaM rescaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.attention.mask_controller import MaskController
from src.constants import INVERSION_TIMESTEP
from src.models.generator import DEFAULT_CONTROLLER_BLOCKS, IPSBV2Model
from src.models.inversion import InverseModel


@dataclass(frozen=True)
class EditConfig:
    """Knobs for :func:`edit_image`.

    Attributes:
        scale_text: ``s_y`` - prompt-alignment gain inside the edit region.
        scale_edit: ``s_edit`` - source-image gain inside the edit region. Kept low so
            the source content does not overpower the requested change.
        scale_non_edit: ``s_non-edit`` - source-image gain outside it, i.e. background
            preservation strength.
        clamp_rate: Multiplier on the mean noise difference used to normalise the mask.
        mask_threshold: Binarisation threshold for the self-guided mask.
        resolution: Square input resolution the source image is resized to.
    """

    scale_text: float = 1.0
    scale_edit: float = 0.2
    scale_non_edit: float = 1.0
    clamp_rate: float = 3.0
    mask_threshold: float = 0.5
    resolution: int = 512


@torch.no_grad()
def extract_editing_mask(
    inverted_source: torch.Tensor,
    inverted_edit: torch.Tensor,
    clamp_rate: float,
    threshold: float,
) -> torch.Tensor:
    """Derive the self-guided editing mask from two inverted noise maps.

    A well-trained inversion network encodes the prompt into the noise it predicts, so
    the spatial difference between a source-conditioned and an edit-conditioned pass
    localises the region the prompt asks to change.

    Args:
        inverted_source: Noise predicted under the source prompt, ``(1, C, h, w)``.
        inverted_edit: Noise predicted under the edit prompt, same shape.
        clamp_rate: Multiplier on the mean difference used as the normalisation ceiling.
        threshold: Binarisation threshold in ``[0, 1]``.

    Returns:
        Binary mask of shape ``(h, w)``.
    """
    difference = (inverted_source - inverted_edit).abs().mean(dim=[0, 1])
    ceiling = (difference.mean() * clamp_rate).item()
    normalised = difference.clamp(0, ceiling) / ceiling
    # Vectorised equivalent of the original per-element Tensor.apply_ loop, which forced
    # a CPU round-trip on every call.
    return (normalised > threshold).to(dtype=difference.dtype)


@torch.no_grad()
def edit_image(
    image_path: str | Path,
    source_prompt: str,
    edit_prompt: str,
    inverse_model: InverseModel,
    generator: IPSBV2Model,
    config: EditConfig | None = None,
) -> torch.Tensor:
    """Edit an image in a single diffusion step.

    Args:
        image_path: Source image on disk.
        source_prompt: Description of the source image. May be empty.
        edit_prompt: Description of the desired result.
        inverse_model: Trained inversion network.
        generator: One-step generator with ARaM-capable attention processors.
        config: Editing knobs; defaults to :class:`EditConfig`.

    Returns:
        Decoded images in ``[0, 1]`` with shape ``(2, 3, H, W)``: the source-prompt
        reconstruction followed by the edited result.
    """
    config = config or EditConfig()
    device = inverse_model.device

    timestep = torch.full((1,), INVERSION_TIMESTEP, dtype=torch.int64, device=device)

    source_image = Image.open(image_path).convert("RGB")
    source_image = source_image.resize((config.resolution, config.resolution))
    pixel_values = to_tensor(source_image).unsqueeze(0).to(device) * 2 - 1

    latents = inverse_model.encode_image(pixel_values)
    encoder_hidden_states = inverse_model.encode_prompts([source_prompt, edit_prompt])
    inverted = inverse_model.invert(
        torch.cat([latents] * 2, dim=0), encoder_hidden_states, timestep
    )
    inverted_source, inverted_edit = inverted.chunk(2)

    mask = extract_editing_mask(
        inverted_source, inverted_edit, config.clamp_rate, config.mask_threshold
    )

    noise = generator.alpha_t * latents + generator.sigma_t * inverted_source
    controller = MaskController(
        mask,
        scale_text_hiddenstate=config.scale_text,
        scale_ip_fg=config.scale_edit,
        scale_ip_bg=config.scale_non_edit,
    )
    generator.set_controller(controller, where=DEFAULT_CONTROLLER_BLOCKS)
    try:
        images, _ = generator.gen_img(
            pil_image=source_image, prompts=[source_prompt, edit_prompt], noise=noise
        )
    finally:
        generator.set_controller(None)

    return images
