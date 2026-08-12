"""Shared constants: Hub identifiers, checkpoint layout and model geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------------------
# Hugging Face Hub identifiers
# --------------------------------------------------------------------------------------

#: Backbone used by the inversion network (VAE, tokenizer, text encoder, scheduler).
SD_TURBO_REPO: Final[str] = "stabilityai/sd-turbo"

#: Backbone used by :class:`~src.models.auxiliary.AuxiliaryModel`.
#:
#: Upstream ``stabilityai/stable-diffusion-2-1-base`` was removed from the Hub and now
#: returns 404 even for authenticated requests. This mirror carries byte-identical
#: weights, verified by sha256:
#:
#: * ``vae/diffusion_pytorch_model.safetensors``  -> ``a1d993488569e928...``
#: * ``text_encoder/model.safetensors``           -> ``cce6febb0b6d876e...``
SD21_BASE_REPO: Final[str] = "Manojb/stable-diffusion-2-1-base"

#: Repository holding the CLIP vision tower used for the IP-Adapter image condition.
IP_ADAPTER_REPO: Final[str] = "h94/IP-Adapter"
IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER: Final[str] = "models/image_encoder"

#: Garment encoder backbone for the virtual try-on adaptation.
DINOV2_REPO: Final[str] = "facebook/dinov2-large"

# --------------------------------------------------------------------------------------
# Local checkpoint layout (see README for download instructions)
# --------------------------------------------------------------------------------------

DEFAULT_WEIGHTS_ROOT: Final[Path] = Path("weights")

INVERSION_CHECKPOINT_DIR: Final[str] = "inverse_ckpt-120k"
INVERSION_UNET_SUBFOLDER: Final[str] = "unet_ema"
GENERATOR_CHECKPOINT_DIR: Final[str] = "sbv2_0.5"
IP_ADAPTER_CHECKPOINT: Final[str] = "ip_adapter_ckpt-90k/ip_adapter.bin"

# --------------------------------------------------------------------------------------
# Model geometry
#
# Verified by static inspection of the released checkpoints; see
# ``scripts/dissect_checkpoints.py`` to reproduce.
# --------------------------------------------------------------------------------------

#: Width of the cross-attention key/value space in both UNets.
CROSS_ATTENTION_DIM: Final[int] = 1024

#: Number of tokens the IP-Adapter image projection emits (``proj: 1024 -> 4 * 1024``).
IP_NUM_TOKENS: Final[int] = 4

#: Latent channel count of the stock UNets.
LATENT_CHANNELS: Final[int] = 4

#: Latent channels once the inpainting condition is attached: noisy + masked + mask.
INPAINTING_LATENT_CHANNELS: Final[int] = 2 * LATENT_CHANNELS + 1

#: Timestep fed to the inversion network when predicting the inverted code.
INVERSION_TIMESTEP: Final[int] = 500

#: VAE downsampling factor (pixel space -> latent space).
VAE_SCALE_FACTOR: Final[int] = 8
