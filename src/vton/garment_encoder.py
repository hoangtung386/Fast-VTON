"""Garment conditioning: DINOv2 patch tokens projected into the prompt branch."""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model

from src.constants import CROSS_ATTENTION_DIM, DINOV2_REPO

#: Backdrop of the VITON-HD product shots, used when letterboxing them to a square.
GARMENT_PAD_COLOUR: tuple[int, int, int] = (255, 255, 255)


def pad_to_square(
    image: Image.Image, fill: tuple[int, int, int] = GARMENT_PAD_COLOUR
) -> Image.Image:
    """Letterbox an image to a square without discarding any of it.

    The DINOv2 processor's stock recipe - resize the shortest edge to 256, centre-crop
    224 - throws away a third of a 768x1024 product shot, taking the neckline and the
    hem with it. Those are exactly the features try-on has to reproduce, and no amount
    of encoder capacity recovers what preprocessing already deleted. Padding trades a
    little effective resolution for keeping the whole garment.

    Args:
        image: Source image, any aspect ratio.
        fill: Colour for the added bars; white matches the VITON-HD backdrop.

    Returns:
        A square image with the original centred inside it.
    """
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image.convert("RGB"), ((side - width) // 2, (side - height) // 2))
    return canvas


class GarmentEncoder(nn.Module):
    """Encode a flat garment photograph into cross-attention tokens.

    The prompt branch of the generator is the natural slot for the garment: ARaM gates
    it by the edit mask, which is precisely the region the new garment must occupy. The
    branch's cross-attention has no positional embedding on the key/value side, so it
    accepts any sequence length - dense DINOv2 patch tokens drop straight in where the
    77 CLIP text tokens used to sit, with no architectural change.

    DINOv2 is preferred over SAM here: SAM's encoder is trained for promptable
    segmentation and optimises for boundaries and objectness, not for the colour and
    texture fidelity that garment transfer lives or dies by.

    The backbone stays frozen; only ``proj`` and ``out_norm`` train. The final linear
    layer is zero-initialised so the garment condition starts as an exact zero and the
    generator behaves identically to its pretrained self at step 0.

    Args:
        model_name: DINOv2 checkpoint. ``dinov2-large`` has ``hidden_size == 1024``,
            already matching ``cross_attention_dim``.
        cross_attention_dim: Width expected by the generator's cross-attention.
        freeze_backbone: Keep the DINOv2 tower frozen.
    """

    def __init__(
        self,
        model_name: str = DINOV2_REPO,
        cross_attention_dim: int = CROSS_ATTENTION_DIM,
        freeze_backbone: bool = True,
        backbone: Dinov2Model | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = Dinov2Model.from_pretrained(model_name)
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, cross_attention_dim),
            nn.GELU(),
            nn.Linear(cross_attention_dim, cross_attention_dim),
        )
        self.out_norm = nn.LayerNorm(cross_attention_dim)

        final_linear = self.proj[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    @classmethod
    def from_config(
        cls,
        backbone_config: dict,
        cross_attention_dim: int = CROSS_ATTENTION_DIM,
    ) -> GarmentEncoder:
        """Build an untrained encoder from a plain config dict, without Hub access.

        Used when restoring an exported bundle on a machine that has no network and no
        cached DINOv2 checkpoint; the weights arrive from the bundle itself.
        """
        backbone = Dinov2Model(Dinov2Config(**backbone_config))
        return cls(cross_attention_dim=cross_attention_dim, backbone=backbone)

    @staticmethod
    def image_processor(
        model_name: str = DINOV2_REPO, resolution: int | None = None
    ) -> AutoImageProcessor:
        """Return the preprocessing pipeline for raw PIL garments.

        Centre-cropping is switched off: callers pass square images through
        :func:`pad_to_square`, so a crop would only shave off the edges again.

        Args:
            model_name: DINOv2 checkpoint whose normalisation statistics to use.
            resolution: Side length to resize to, overriding the checkpoint's 224.
                DINOv2 interpolates its position embeddings, so any multiple of the
                patch size works. Must match ``DataConfig.garment_resolution``, which
                is what the feature cache is shaped from.

        Returns:
            The configured processor.
        """
        processor = AutoImageProcessor.from_pretrained(model_name)
        processor.do_center_crop = False
        if resolution is not None:
            processor.size = {"shortest_edge": resolution}
        return processor

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters this module contributes to the optimiser."""
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def encode_frozen(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the frozen backbone only, for offline feature caching.

        Returns:
            ``(batch, 1 + num_patches, hidden)`` - a CLS token followed by patch tokens.
            At 224 px with patch size 14 that is 257 tokens.
        """
        return self.backbone(pixel_values).last_hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        cached_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project garment features to prompt-branch tokens.

        Args:
            pixel_values: Preprocessed garment images. Ignored when ``cached_features``
                is supplied.
            cached_features: Precomputed backbone output, as produced by
                :meth:`encode_frozen`.

        Returns:
            ``(batch, num_tokens, cross_attention_dim)``.
        """
        if cached_features is None:
            if pixel_values is None:
                raise ValueError("provide either pixel_values or cached_features")
            cached_features = self.encode_frozen(pixel_values)

        return self.out_norm(self.proj(cached_features))
