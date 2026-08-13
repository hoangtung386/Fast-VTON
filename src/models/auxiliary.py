"""Frozen support modules shared by the generator: VAE, scheduler, encoders."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
)

from src.constants import (
    IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER,
    IP_ADAPTER_REPO,
    SD21_BASE_REPO,
)


class AuxiliaryModel:
    """Bundle of frozen modules the one-step generator depends on.

    Args:
        model_name: Hub id supplying the VAE, tokenizer, text encoder and scheduler.
        image_encoder_path: Hub id supplying the CLIP vision tower.
        device: Torch device for every submodule.
        load_text_encoder: Set ``False`` for virtual try-on, where the prompt branch is
            driven by garment tokens and the text tower is dead weight (~1.4 GB).
        load_vae: Set ``False`` when latents are precomputed. Saves 0.33 GB.
        load_image_encoder: Set ``False`` when CLIP embeddings are precomputed. Saves
            2.53 GB - the single largest saving available to a cache-driven training
            loop. ``clip_projection_dim`` then has to be supplied explicitly, because
            the generator sizes its image projection from it.
        clip_projection_dim: Width of the CLIP image embedding, used in place of the
            tower's own config when ``load_image_encoder`` is ``False``.
    """

    def __init__(
        self,
        model_name: str = SD21_BASE_REPO,
        image_encoder_path: str = IP_ADAPTER_REPO,
        device: str | torch.device = "cuda",
        load_text_encoder: bool = True,
        load_vae: bool = True,
        load_image_encoder: bool = True,
        clip_projection_dim: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name

        if not load_image_encoder and clip_projection_dim is None:
            raise ValueError("clip_projection_dim is required when load_image_encoder=False")

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae: AutoencoderKL | None = None
        if load_vae:
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)

        self.tokenizer: AutoTokenizer | None = None
        self.text_encoder: CLIPTextModel | None = None
        if load_text_encoder:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_name, subfolder="text_encoder"
            ).to(self.device, dtype=torch.float32)

        self.image_encoder: CLIPVisionModelWithProjection | None = None
        self._clip_projection_dim = clip_projection_dim
        if load_image_encoder:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                image_encoder_path, subfolder=IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER
            ).to(self.device, dtype=torch.float32)
            self.image_encoder.requires_grad_(False)
            self._clip_projection_dim = self.image_encoder.config.projection_dim

        self.clip_image_processor = CLIPImageProcessor()

    @property
    def clip_projection_dim(self) -> int:
        """Width of the CLIP image embedding, with or without the tower loaded."""
        if self._clip_projection_dim is None:
            raise RuntimeError("no CLIP projection dim available")
        return self._clip_projection_dim

    @property
    def scaling_factor(self) -> float:
        """VAE latent scaling factor."""
        if self.vae is None:
            raise RuntimeError("VAE was not loaded; construct with load_vae=True")
        return self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_clip_image(self, images: Image.Image | Sequence[Image.Image]) -> torch.Tensor:
        """Encode images to pooled CLIP embeddings, ``(batch, projection_dim)``.

        This is the frozen half of the IP-Adapter conditioning path. The learned
        projection that turns these into tokens lives on the generator, so caches built
        for training must stop here.
        """
        if self.image_encoder is None:
            raise RuntimeError("CLIP tower was not loaded; construct with load_image_encoder=True")
        batch = [images] if isinstance(images, Image.Image) else list(images)
        pixel_values = self.clip_image_processor(images=batch, return_tensors="pt").pixel_values
        return self.image_encoder(pixel_values.to(self.device, dtype=torch.float32)).image_embeds

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode scaled latents to images in ``[0, 1]``."""
        if self.vae is None:
            raise RuntimeError("VAE was not loaded; construct with load_vae=True")
        images = self.vae.decode(
            (latents / self.scaling_factor).to(dtype=self.vae.dtype)
        ).sample.float()
        return (images + 1) / 2
