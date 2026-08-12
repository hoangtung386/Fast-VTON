"""Stage 1 training loop: warm up the garment branch with the backbone frozen."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.generator import IPSBV2Model
from src.vton.config import Stage1Config
from src.vton.freezing import TrainableGroups, configure_trainable
from src.vton.garment_encoder import GarmentEncoder

logger = logging.getLogger(__name__)

_AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {
    "no": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def masked_latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mask_weight: float,
) -> torch.Tensor:
    """Mean squared error with the masked region up-weighted.

    The unmasked region is largely handed to the model through the inpainting channels,
    so an unweighted loss lets the easy background dominate the gradient. ``mask_weight``
    restores the balance.

    Args:
        prediction: Predicted clean latent, ``(B, 4, h, w)``.
        target: Ground-truth latent of the person wearing the garment.
        mask: Binary latent mask, ``(B, 1, h, w)``.
        mask_weight: Extra weight applied inside the mask.

    Returns:
        Scalar loss.
    """
    weights = 1.0 + mask_weight * mask
    return (((prediction - target) ** 2) * weights).mean()


@dataclass
class TrainingState:
    """Mutable counters persisted across checkpoint/resume cycles."""

    step: int = 0
    samples_seen: int = 0


class Stage1Trainer:
    """Drive Stage 1 over a cached VITON-HD dataset.

    Args:
        generator: One-step generator, constructed with 9 inpainting channels.
        garment_encoder: Garment branch producing prompt-slot tokens.
        config: Hyper-parameters.
        groups: Which parameter groups to unfreeze.
    """

    def __init__(
        self,
        generator: IPSBV2Model,
        garment_encoder: GarmentEncoder,
        config: Stage1Config,
        groups: TrainableGroups | None = None,
    ) -> None:
        self.generator = generator
        self.garment_encoder = garment_encoder
        self.config = config
        self.state = TrainingState()

        self.parameters = configure_trainable(generator, garment_encoder, groups)
        if not self.parameters:
            raise ValueError("no trainable parameters selected")

        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.max_steps,
            pct_start=config.warmup_fraction,
        )

        self.autocast_dtype = _AUTOCAST_DTYPES[config.mixed_precision]
        # GradScaler is only meaningful for fp16; bf16 has fp32's exponent range.
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision == "fp16")

        if config.gradient_checkpointing:
            generator.unet.enable_gradient_checkpointing()

        self.device = generator.device_
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ steps

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run one forward pass and return the scalar training loss."""
        z_agnostic = batch["z_agnostic"].to(self.device, dtype=torch.float32)
        z_person = batch["z_person"].to(self.device, dtype=torch.float32)
        inverted = batch["inverted_noise"].to(self.device, dtype=torch.float32)
        mask = batch["mask_latent"].to(self.device, dtype=torch.float32)
        clip_embeds = batch["clip_image_embeds"].to(self.device, dtype=torch.float32)
        garment = batch["garment_features"].to(self.device, dtype=torch.float32)

        garment_tokens = self.garment_encoder(cached_features=garment)
        ip_tokens = self.generator.image_proj_model(clip_embeds)

        noisy = self.generator.alpha_t * z_agnostic + self.generator.sigma_t * inverted
        prediction = self.generator.forward_train(
            noisy_latent=noisy,
            masked_latent=z_agnostic,
            mask_latent=mask,
            prompt_tokens=garment_tokens,
            ip_tokens=ip_tokens,
        )
        return masked_latent_loss(prediction, z_person, mask, self.config.mask_loss_weight)

    def training_step(self, batch: dict[str, torch.Tensor], is_accumulating: bool) -> float:
        """Accumulate gradients for one micro-batch; step the optimiser when ready."""
        with torch.autocast(
            "cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None
        ):
            loss = self.compute_loss(batch) / self.config.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        if not is_accumulating:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        return loss.item() * self.config.gradient_accumulation_steps

    # --------------------------------------------------------------- training

    def train(self, dataloader: DataLoader) -> TrainingState:
        """Run until ``config.max_steps`` optimiser steps have been taken."""
        config = self.config
        self.generator.train()
        self.garment_encoder.train()
        self.garment_encoder.backbone.eval()  # frozen; keep its norm statistics fixed

        micro_step = 0
        running_loss = 0.0
        started = time.monotonic()

        while self.state.step < config.max_steps:
            for batch in dataloader:
                is_accumulating = (micro_step + 1) % config.gradient_accumulation_steps != 0
                running_loss += self.training_step(batch, is_accumulating)
                micro_step += 1
                self.state.samples_seen += batch["z_person"].shape[0]

                if is_accumulating:
                    continue

                self.state.step += 1
                if self.state.step % config.log_every == 0:
                    window = config.log_every * config.gradient_accumulation_steps
                    elapsed = time.monotonic() - started
                    logger.info(
                        "step %d/%d | loss %.5f | lr %.2e | %.2f steps/s",
                        self.state.step,
                        config.max_steps,
                        running_loss / window,
                        self.scheduler.get_last_lr()[0],
                        config.log_every / max(elapsed, 1e-6),
                    )
                    running_loss = 0.0
                    started = time.monotonic()

                if self.state.step % config.checkpoint_every == 0:
                    self.save_checkpoint()
                if self.state.step >= config.max_steps:
                    break

        self.save_checkpoint(final=True)
        return self.state

    # ------------------------------------------------------------ checkpoints

    def _trainable_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Collect only the tensors that actually move during Stage 1."""
        return {
            "generator": {
                name: param.detach().to(torch.float16).cpu()
                for name, param in self.generator.named_parameters()
                if param.requires_grad
            },
            "garment_encoder": {
                name: param.detach().to(torch.float16).cpu()
                for name, param in self.garment_encoder.named_parameters()
                if param.requires_grad
            },
        }

    def save_checkpoint(self, final: bool = False) -> Path:
        """Write a resumable checkpoint holding only the trainable tensors.

        The generator and inversion network are frozen, so storing their 1.7 B
        parameters on every save would waste time and space - especially on Colab,
        where the run can be interrupted at any moment.
        """
        name = "final.pt" if final else f"step_{self.state.step:06d}.pt"
        path = self.config.output_dir / name
        torch.save(
            {
                "step": self.state.step,
                "samples_seen": self.state.samples_seen,
                "weights": self._trainable_state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            path,
        )
        logger.info("saved checkpoint %s", path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore weights, optimiser and counters from :meth:`save_checkpoint`."""
        payload = torch.load(path, map_location="cpu")

        for module, key in (
            (self.generator, "generator"),
            (self.garment_encoder, "garment_encoder"),
        ):
            saved = payload["weights"][key]
            named = dict(module.named_parameters())
            for name, tensor in saved.items():
                with torch.no_grad():
                    named[name].copy_(tensor.to(named[name].dtype))

        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.state = TrainingState(
            step=payload["step"], samples_seen=payload.get("samples_seen", 0)
        )
        logger.info("resumed from %s at step %d", path, self.state.step)
