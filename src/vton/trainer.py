"""Stage 1 training loop: warm up the garment branch with the backbone frozen."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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

#: Accepted by :func:`resolve_mixed_precision`; ``"auto"`` never reaches the config.
MIXED_PRECISION_CHOICES: tuple[str, ...] = ("auto", "no", "fp16", "bf16")


def resolve_mixed_precision(requested: str = "auto", device: str | torch.device = "cuda") -> str:
    """Turn a precision request into one the GPU on hand can actually run.

    bfloat16 needs compute capability 8.0 - Ampere and later. A T4 is 7.5, and
    ``torch.autocast`` only notices at the first training step, long after the weights
    have loaded. Resolving up front moves that failure to the command line, where it
    costs seconds instead of minutes.

    Args:
        requested: One of :data:`MIXED_PRECISION_CHOICES`. ``"auto"`` picks bf16 where
            it is available, fp16 on older CUDA hardware, and disables autocast off-GPU.
        device: Device training will run on.

    Returns:
        A concrete precision - ``"no"``, ``"fp16"`` or ``"bf16"``, never ``"auto"``.

    Raises:
        ValueError: The request is unknown, or bf16 was asked for on hardware that has
            no bf16.
    """
    if requested not in MIXED_PRECISION_CHOICES:
        raise ValueError(
            f"mixed precision must be one of {MIXED_PRECISION_CHOICES}, got {requested!r}"
        )

    on_cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    # The same predicate torch.autocast checks, so this cannot disagree with it.
    has_bf16 = on_cuda and torch.cuda.is_bf16_supported()

    if requested == "auto":
        choice = "bf16" if has_bf16 else ("fp16" if on_cuda else "no")
        logger.info("mixed precision auto-selected: %s", choice)
        return choice

    if requested == "bf16" and not has_bf16:
        raise ValueError(
            "this device has no bfloat16 (it needs compute capability 8.0; a T4 is 7.5). "
            "Pass --mixed-precision fp16, or auto to let it choose."
        )
    return requested


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
    best_val_loss: float = float("inf")

    def epochs(self, dataset_size: int) -> float:
        """How many passes over the data the run has actually made.

        Logged every window because it is the number that exposes a mis-set batch size.
        A run can report a healthy steps/s while seeing three epochs where it was meant
        to see fifty.
        """
        return self.samples_seen / max(dataset_size, 1)


class Stage1Trainer:
    """Drive Stage 1 over a cached VITON-HD dataset.

    Args:
        generator: One-step generator, constructed with 9 inpainting channels.
        garment_encoder: Garment branch producing prompt-slot tokens.
        config: Hyper-parameters.
        groups: Which parameter groups to unfreeze.
        vae: Frozen VAE, supplied only to decode preview images. Training never needs
            it - the latents are cached - so leaving it ``None`` saves 0.33 GB.
    """

    def __init__(
        self,
        generator: IPSBV2Model,
        garment_encoder: GarmentEncoder,
        config: Stage1Config,
        groups: TrainableGroups | None = None,
        vae: object | None = None,
    ) -> None:
        self.generator = generator
        self.garment_encoder = garment_encoder
        self.config = config
        self.vae = vae
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

    # ------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Mean loss over a held-out split.

        Training loss alone cannot separate convergence from noise: at small batch
        sizes its window-to-window swing can exceed the improvement of ten thousand
        steps. A validation pass over a fixed split is quiet enough to read.
        """
        self.generator.eval()
        self.garment_encoder.eval()
        total, count = 0.0, 0
        for batch in dataloader:
            with torch.autocast(
                "cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None
            ):
                loss = self.compute_loss(batch)
            size = batch["z_person"].shape[0]
            total += loss.item() * size
            count += size
        self.generator.train()
        self.garment_encoder.train()
        return total / max(count, 1)

    @torch.no_grad()
    def save_preview(self, batch: dict[str, torch.Tensor], name: str) -> Path | None:
        """Decode one batch to a PNG strip: agnostic input, prediction, ground truth.

        The one check no scalar can stand in for. A run can post a respectable loss
        while emitting mush, and without this nobody finds out until the model reaches
        a server. Needs the VAE; returns ``None`` when it was not supplied.
        """
        if self.vae is None:
            return None

        self.generator.eval()
        self.garment_encoder.eval()
        with torch.autocast(
            "cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None
        ):
            z_agnostic = batch["z_agnostic"].to(self.device, dtype=torch.float32)
            mask = batch["mask_latent"].to(self.device, dtype=torch.float32)
            prediction = self.generator.forward_train(
                noisy_latent=(
                    self.generator.alpha_t * z_agnostic
                    + self.generator.sigma_t
                    * batch["inverted_noise"].to(self.device, dtype=torch.float32)
                ),
                masked_latent=z_agnostic,
                mask_latent=mask,
                prompt_tokens=self.garment_encoder(
                    cached_features=batch["garment_features"].to(self.device, torch.float32)
                ),
                ip_tokens=self.generator.image_proj_model(
                    batch["clip_image_embeds"].to(self.device, torch.float32)
                ),
            )
        self.generator.train()
        self.garment_encoder.train()

        scaling = self.vae.config.scaling_factor
        rows = [
            self.vae.decode(
                (latents.float() / scaling).to(self.vae.dtype)
            ).sample.float()
            for latents in (
                z_agnostic,
                prediction,
                batch["z_person"].to(self.device, dtype=torch.float32),
            )
        ]
        grid = torch.cat([((row + 1) / 2).clamp(0, 1).cpu() for row in rows], dim=2)
        path = self.config.output_dir / f"{name}.png"
        save_image(grid, path, nrow=grid.shape[0])
        logger.info("wrote preview %s", path)
        return path

    # --------------------------------------------------------------- training

    def train(
        self,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
    ) -> TrainingState:
        """Run until ``config.max_steps`` optimiser steps have been taken."""
        config = self.config
        dataset_size = len(dataloader.dataset)
        self.generator.train()
        self.garment_encoder.train()
        if self.garment_encoder.backbone is not None:
            self.garment_encoder.backbone.eval()  # frozen; keep its norm statistics fixed

        preview_batch = next(iter(val_dataloader or dataloader))
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
                    elapsed = max(time.monotonic() - started, 1e-6)
                    steps_per_second = config.log_every / elapsed
                    # samples/s, not steps/s: a run at batch 1 posts a healthy steps/s
                    # while moving sixteen times less data than it was meant to.
                    logger.info(
                        "step %d/%d | loss %.5f | lr %.2e | %.1f samples/s "
                        "(%.2f steps/s) | epoch %.2f%s",
                        self.state.step,
                        config.max_steps,
                        running_loss / window,
                        self.scheduler.get_last_lr()[0],
                        steps_per_second * config.effective_batch_size,
                        steps_per_second,
                        self.state.epochs(dataset_size),
                        self._memory_note(),
                    )
                    running_loss = 0.0
                    started = time.monotonic()

                if val_dataloader is not None and self.state.step % config.eval_every == 0:
                    val_loss = self.evaluate(val_dataloader)
                    best = val_loss < self.state.best_val_loss
                    self.state.best_val_loss = min(val_loss, self.state.best_val_loss)
                    logger.info(
                        "step %d | val loss %.5f%s",
                        self.state.step,
                        val_loss,
                        "  <- best so far" if best else "",
                    )
                    started = time.monotonic()

                if config.preview_every and self.state.step % config.preview_every == 0:
                    self.save_preview(preview_batch, f"preview_{self.state.step:06d}")
                    started = time.monotonic()

                if self.state.step % config.checkpoint_every == 0:
                    self.save_checkpoint()
                if self.state.step >= config.max_steps:
                    break

        self.save_preview(preview_batch, "preview_final")
        self.save_checkpoint(final=True)
        return self.state

    def _memory_note(self) -> str:
        """Peak VRAM so far, so an idle GPU is visible in the log rather than in nvidia-smi."""
        if not torch.cuda.is_available():
            return ""
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        return f" | vram {peak:.1f}/{total:.0f} GB"

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
