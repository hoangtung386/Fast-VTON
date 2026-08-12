"""One-step inversion network: image latent -> inverted noise."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel

from src.constants import SD_TURBO_REPO
from src.utils.text import tokenize_captions

_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class InverseModel:
    """Wraps ``F_theta`` together with the VAE and text encoder it was trained against.

    The inversion UNet shares the SwiftBrush v2 architecture and was initialised from
    its weights, but static comparison shows all 686 tensors have since diverged - the
    largest relative drift (0.98) sits in the cross-attention projections. It is
    therefore *not* interchangeable with the generator.

    Args:
        pretrained_model_name_path: Directory holding the ``unet_ema`` subfolder.
        model_name: Hub id supplying the VAE, tokenizer, text encoder and scheduler.
        dtype: One of ``"fp16"``, ``"bf16"`` or ``"fp32"``. The VAE always stays fp32.
        device: Torch device for every submodule.
        load_text_encoder: Set ``False`` to skip the 1.4 GB text tower when a cached
            null embedding is used instead (see ``scripts/make_null_embedding.py``).
    """

    def __init__(
        self,
        pretrained_model_name_path: str | Path,
        model_name: str = SD_TURBO_REPO,
        dtype: str = "fp32",
        device: str | torch.device = "cuda",
        load_text_encoder: bool = True,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")

        self.weight_dtype = _DTYPES[dtype]
        self.device = torch.device(device)
        self.model_name = model_name

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(
            self.device, dtype=torch.float32
        )

        self.unet_inverse = UNet2DConditionModel.from_pretrained(
            str(pretrained_model_name_path), subfolder="unet_ema"
        ).to(self.device, dtype=self.weight_dtype)
        self.unet_inverse.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder: CLIPTextModel | None = None
        if load_text_encoder:
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_name, subfolder="text_encoder"
            ).to(self.device, dtype=self.weight_dtype)

        num_timesteps = self.noise_scheduler.config.num_train_timesteps
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        quarter = (num_timesteps - 1) // 4

        self.corrupt_alpha_t = (alphas_cumprod[quarter] ** 0.5).view(-1, 1, 1, 1)
        self.corrupt_sigma_t = ((1 - alphas_cumprod[quarter]) ** 0.5).view(-1, 1, 1, 1)

        del alphas_cumprod

    @torch.no_grad()
    def encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        """Encode prompts to ``(len(prompts), 77, 1024)`` hidden states."""
        if self.text_encoder is None:
            raise RuntimeError(
                "text encoder was not loaded; construct with load_text_encoder=True or "
                "supply a cached null embedding"
            )
        input_ids = tokenize_captions(self.tokenizer, prompts).to(self.device)
        return self.text_encoder(input_ids)[0].to(dtype=self.weight_dtype)

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a ``[-1, 1]`` image batch to scaled latents.

        Casts to the VAE's own dtype rather than ``weight_dtype``: the VAE is pinned to
        fp32, so feeding it fp16 would fail once a half-precision UNet is requested.
        """
        latents = self.vae.encode(pixel_values.to(self.vae.dtype)).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    @torch.no_grad()
    def invert(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the inverted noise for ``latents`` under the given condition."""
        return self.unet_inverse(latents, timestep, encoder_hidden_states).sample.to(
            self.device, dtype=self.weight_dtype
        )
