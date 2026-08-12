"""One-step SwiftBrush v2 generator with an IP-Adapter image branch."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from PIL import Image

from src.attention.mask_controller import MaskController
from src.attention.mask_processors import IPAttnProcessor2_0WithIPMaskController
from src.attention.processors import AttnProcessor2_0, IPAttnProcessor2_0
from src.constants import IP_NUM_TOKENS, LATENT_CHANNELS
from src.models.auxiliary import AuxiliaryModel
from src.models.projection import ImageProjModel
from src.utils.text import tokenize_captions

logger = logging.getLogger(__name__)

#: Attention sites ARaM is applied to by default.
DEFAULT_CONTROLLER_BLOCKS: tuple[str, ...] = ("mid_blocks", "up_blocks")


def expand_conv_in(unet: UNet2DConditionModel, new_in_channels: int) -> UNet2DConditionModel:
    """Widen ``unet.conv_in`` in place, zero-initialising the added channels.

    Used to attach an inpainting condition (masked latent + mask) to a UNet that was
    trained on plain latents. Because the new slice starts at zero the network's output
    is bit-identical to the original until those weights receive gradient.

    Must be called *after* loading the pretrained state dict, otherwise the ``conv_in``
    shapes will not match the checkpoint.

    Args:
        unet: Model to modify.
        new_in_channels: Target channel count, must exceed the current one.

    Returns:
        The same ``unet``, modified in place.
    """
    old = unet.conv_in
    if new_in_channels < old.in_channels:
        raise ValueError(
            f"cannot shrink conv_in from {old.in_channels} to {new_in_channels} channels"
        )
    if new_in_channels == old.in_channels:
        return unet

    new = nn.Conv2d(
        new_in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
    ).to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)

    unet.conv_in = new
    unet.register_to_config(in_channels=new_in_channels)
    logger.info("expanded conv_in from %d to %d channels", old.in_channels, new_in_channels)
    return unet


def attention_hidden_size(unet: UNet2DConditionModel, processor_name: str) -> int:
    """Resolve the attention width for a processor from its position in the UNet."""
    channels = unet.config.block_out_channels
    if processor_name.startswith("mid_block"):
        return channels[-1]
    if processor_name.startswith("up_blocks"):
        block_id = int(processor_name[len("up_blocks.")])
        return list(reversed(channels))[block_id]
    if processor_name.startswith("down_blocks"):
        block_id = int(processor_name[len("down_blocks.")])
        return channels[block_id]
    raise ValueError(f"cannot infer hidden size for attention processor {processor_name!r}")


def install_ip_attn_processors(
    unet: UNet2DConditionModel,
    *,
    with_mask_controller: bool,
    device: torch.device | str = "cpu",
    seed_from_unet: bool = True,
) -> nn.ModuleList:
    """Replace the UNet's attention processors with the IP-Adapter variants.

    Self-attention sites get a parameter-free processor; cross-attention sites get one
    carrying ``to_k_ip`` / ``to_v_ip``.

    Args:
        unet: Model to modify in place.
        with_mask_controller: Install ARaM-capable processors instead of plain ones.
        device: Device to place the new processors on.
        seed_from_unet: Initialise the image K/V from the host UNet's prompt K/V. Only
            useful when training an adapter from scratch - when a pretrained
            ``ip_adapter.bin`` is about to be loaded these values are overwritten, so
            the copy is pure overhead.

    Returns:
        A :class:`torch.nn.ModuleList` aliasing the installed processors. The alias is
        what makes the module's ``state_dict`` match the released checkpoint layout.
    """
    processor_cls = (
        IPAttnProcessor2_0WithIPMaskController if with_mask_controller else IPAttnProcessor2_0
    )
    unet_state = unet.state_dict() if seed_from_unet else {}
    attn_procs: dict[str, nn.Module] = {}

    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            # Self-attention: no image condition, no extra parameters.
            attn_procs[name] = AttnProcessor2_0().to(device)
            continue

        processor = processor_cls(
            hidden_size=attention_hidden_size(unet, name),
            cross_attention_dim=unet.config.cross_attention_dim,
        ).to(device)

        if seed_from_unet:
            layer_name = name.split(".processor")[0]
            processor.load_state_dict(
                {
                    "to_k_ip.weight": unet_state[f"{layer_name}.to_k.weight"],
                    "to_v_ip.weight": unet_state[f"{layer_name}.to_v.weight"],
                }
            )
        attn_procs[name] = processor

    unet.set_attn_processor(attn_procs)
    return nn.ModuleList(unet.attn_processors.values())


class IPSBV2Model(nn.Module):
    """SwiftBrush v2 UNet plus the IP-Adapter decoupled cross-attention branch.

    The module's ``state_dict`` deliberately mirrors the released ``ip_adapter.bin``
    layout, which contains three groups: ``unet.*``, ``image_proj_model.*`` and
    ``adapter_modules.*``. The latter is a second reference to the attention processors
    that already live inside ``unet``, so those tensors appear twice - dropping the
    alias would break checkpoint loading.

    Args:
        pretrained_model_name_path: Directory holding the SwiftBrush v2 UNet.
        ip_model_path: Path to ``ip_adapter.bin``.
        aux_model: Frozen VAE / scheduler / encoders bundle.
        device: Torch device for every submodule.
        with_ip_mask_controller: Install ARaM-capable processors instead of plain ones.
        inpainting_channels: When set, widen ``conv_in`` to this many channels after
            loading. Pass ``INPAINTING_LATENT_CHANNELS`` (9) for virtual try-on.
    """

    def __init__(
        self,
        pretrained_model_name_path: str | Path,
        ip_model_path: str | Path,
        aux_model: AuxiliaryModel,
        device: str | torch.device = "cuda",
        with_ip_mask_controller: bool = False,
        inpainting_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.device_ = torch.device(device)
        self.aux_model = aux_model
        self.with_ip_mask_controller = with_ip_mask_controller

        self.unet = UNet2DConditionModel.from_pretrained(str(pretrained_model_name_path)).to(
            self.device_
        )
        self.unet.eval()

        num_timesteps = aux_model.noise_scheduler.config.num_train_timesteps
        self.timestep = torch.ones((1,), dtype=torch.int64, device=self.device_) * (
            num_timesteps - 1
        )

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=aux_model.image_encoder.config.projection_dim,
            clip_extra_context_tokens=IP_NUM_TOKENS,
        ).to(self.device_)

        # Seeding to_k_ip/to_v_ip from the host UNet is redundant here - load_state_dict
        # overwrites every one of them a few lines below - so skip the copy.
        self.adapter_modules = install_ip_attn_processors(
            self.unet,
            with_mask_controller=with_ip_mask_controller,
            device=self.device_,
            seed_from_unet=False,
        )

        alphas_cumprod = aux_model.noise_scheduler.alphas_cumprod.to(self.device_)
        self.alpha_t = (alphas_cumprod[self.timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod

        state_dict = torch.load(ip_model_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
        del state_dict

        if inpainting_channels is not None:
            expand_conv_in(self.unet, inpainting_channels)

    # ------------------------------------------------------------------ setup

    def set_scale(self, scale: float) -> None:
        """Set the image-prompt gain on every IP-Adapter processor."""
        for processor in self.unet.attn_processors.values():
            if isinstance(processor, IPAttnProcessor2_0):
                processor.scale = scale

    def set_controller(
        self,
        controller: MaskController | None,
        where: Sequence[str] = DEFAULT_CONTROLLER_BLOCKS,
    ) -> None:
        """Attach (or clear, with ``None``) an ARaM controller on the selected blocks."""
        for name, processor in self.unet.attn_processors.items():
            if isinstance(processor, IPAttnProcessor2_0WithIPMaskController) and any(
                block in name for block in where
            ):
                processor.controller = controller

    # ------------------------------------------------------------- conditions

    @torch.inference_mode()
    def get_image_embeds(
        self,
        pil_image: Image.Image | Sequence[Image.Image] | None = None,
        clip_image_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode an image (or a precomputed CLIP embedding) to IP-Adapter tokens."""
        if pil_image is not None:
            clip_image_embeds = self.aux_model.encode_clip_image(pil_image)
        elif clip_image_embeds is None:
            raise ValueError("provide either pil_image or clip_image_embeds")
        else:
            clip_image_embeds = clip_image_embeds.to(self.device_, dtype=torch.float32)

        return self.image_proj_model(clip_image_embeds)

    # ---------------------------------------------------------------- forward

    def predict_original_sample(
        self, noise: torch.Tensor, model_pred: torch.Tensor
    ) -> torch.Tensor:
        """Convert an epsilon prediction to the implied clean latent."""
        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

        pred = (noise - self.sigma_t * model_pred) / self.alpha_t

        scheduler_config = self.aux_model.noise_scheduler.config
        if scheduler_config.thresholding:
            pred = self.aux_model.noise_scheduler._threshold_sample(pred)
        elif scheduler_config.clip_sample:
            clip_range = scheduler_config.clip_sample_range
            pred = pred.clamp(-clip_range, clip_range)
        return pred

    def forward_train(
        self,
        noisy_latent: torch.Tensor,
        masked_latent: torch.Tensor,
        mask_latent: torch.Tensor,
        prompt_tokens: torch.Tensor,
        ip_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Single one-step pass returning the predicted clean latent.

        Unlike :meth:`gen_img` this stays entirely in latent space (no VAE decode) and
        keeps gradients, which is what the virtual try-on Stage 1 loop needs.

        Args:
            noisy_latent: ``(B, 4, h, w)`` starting point, typically
                ``alpha_t * z_agnostic + sigma_t * inverted_noise``.
            masked_latent: ``(B, 4, h, w)`` latent of the masked-out person.
            mask_latent: ``(B, 1, h, w)`` binary mask on the latent grid.
            prompt_tokens: ``(B, n, 1024)`` garment tokens for the prompt branch.
            ip_tokens: ``(B, 4, 1024)`` image tokens for the IP branch.

        Returns:
            ``(B, 4, h, w)`` predicted clean latent.
        """
        expected = 2 * LATENT_CHANNELS + 1
        if self.unet.config.in_channels != expected:
            raise RuntimeError(
                f"forward_train needs a {expected}-channel conv_in; the UNet has "
                f"{self.unet.config.in_channels}. Construct the model with "
                f"inpainting_channels={expected}."
            )

        sample = torch.cat([noisy_latent, masked_latent, mask_latent], dim=1)
        condition = torch.cat([prompt_tokens, ip_tokens], dim=1)
        model_pred = self.unet(sample, self.timestep, condition).sample
        return self.predict_original_sample(noisy_latent, model_pred)

    @torch.no_grad()
    def gen_img(
        self,
        pil_image: Image.Image | None = None,
        prompts: Sequence[str] | None = None,
        noise: torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the text-guided editing forward pass.

        Returns:
            ``(image, noise_image)``, both decoded to ``[0, 1]``.
        """
        if noise is None:
            raise ValueError("noise is required")
        if self.aux_model.text_encoder is None:
            raise RuntimeError(
                "gen_img needs the text encoder; construct AuxiliaryModel with "
                "load_text_encoder=True"
            )

        prompts = list(prompts) if prompts else ["best quality, high quality"]
        num_samples = len(prompts)
        self.set_scale(scale)

        image_prompt_embeds = self.get_image_embeds(pil_image=pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        input_ids = tokenize_captions(self.aux_model.tokenizer, prompts).to(self.device_)
        prompt_embeds = self.aux_model.text_encoder(input_ids)[0]
        condition = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)

        noise = torch.cat([noise] * num_samples, dim=0)
        model_pred = self.unet(noise, self.timestep, condition).sample
        pred_original_sample = self.predict_original_sample(noise, model_pred)

        image = self.aux_model.decode_latents(pred_original_sample)
        noise_image = self.aux_model.decode_latents(noise)
        return image, noise_image
