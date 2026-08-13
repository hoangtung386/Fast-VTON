"""Train Stage 1 of the virtual try-on adaptation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.constants import INPAINTING_LATENT_CHANNELS
from src.models import AuxiliaryModel, IPSBV2Model
from src.vton import CachedVtonDataset, GarmentEncoder, Stage1Config, Stage1Trainer
from src.vton.config import CheckpointConfig, DataConfig
from src.vton.data import split_indices
from src.vton.freezing import TrainableGroups
from src.vton.trainer import MIXED_PRECISION_CHOICES, resolve_mixed_precision

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    defaults = Stage1Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("outputs/vton_cache"))
    parser.add_argument("--weights-root", type=Path, default=CheckpointConfig().root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=defaults.gradient_accumulation_steps
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--mask-loss-weight", type=float, default=defaults.mask_loss_weight)
    parser.add_argument(
        "--mixed-precision",
        default="auto",
        choices=list(MIXED_PRECISION_CHOICES),
        help=(
            "auto picks bf16 on Ampere and newer, fp16 on older CUDA cards. bf16 on a "
            "T4 (compute capability 7.5) has no hardware behind it and is rejected"
        ),
    )
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument(
        "--preview-every",
        type=int,
        default=defaults.preview_every,
        help="steps between preview PNGs; 0 disables them and loads no VAE",
    )
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--train-image-kv",
        action="store_true",
        help="also unfreeze to_k_ip/to_v_ip (25.56 M more parameters)",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Assemble the model, dataset and trainer, then run Stage 1."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    checkpoints = CheckpointConfig(root=args.weights_root)
    checkpoints.validate()

    dataset = CachedVtonDataset(args.cache)
    logger.info("cache holds %d samples", len(dataset))

    config = Stage1Config(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        mask_loss_weight=args.mask_loss_weight,
        mixed_precision=resolve_mixed_precision(args.mixed_precision, args.device),
        log_every=args.log_every,
        eval_every=args.eval_every,
        preview_every=args.preview_every,
        val_fraction=args.val_fraction,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        data=DataConfig(
            height=dataset.meta["height"],
            width=dataset.meta["width"],
        ),
        checkpoints=checkpoints,
    )
    logger.info("effective batch size: %d", config.effective_batch_size)

    # Everything the cache already holds is dead weight on the GPU: the CLIP tower is
    # 2.53 GB, DINOv2 1.22 GB, the VAE 0.33 GB, and the training step touches none of
    # them. Only the scheduler and the CLIP embedding width are still needed.
    sample = dataset[0]
    aux_model = AuxiliaryModel(
        device=args.device,
        load_text_encoder=False,
        load_image_encoder=False,
        load_vae=args.preview_every > 0,  # previews decode latents; nothing else does
        clip_projection_dim=sample["clip_image_embeds"].shape[-1],
    )
    generator = IPSBV2Model(
        checkpoints.generator_dir,
        checkpoints.ip_adapter_path,
        aux_model,
        device=args.device,
        with_ip_mask_controller=True,
        inpainting_channels=INPAINTING_LATENT_CHANNELS,
    )
    garment_encoder = GarmentEncoder.for_cached_features(
        hidden_size=sample["garment_features"].shape[-1]
    ).to(args.device)

    trainer = Stage1Trainer(
        generator,
        garment_encoder,
        config,
        groups=TrainableGroups(image_kv=args.train_image_kv),
        vae=aux_model.vae,
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    train_indices, val_indices = split_indices(len(dataset), config.val_fraction)
    logger.info("split: %d train / %d validation", len(train_indices), len(val_indices))

    def loader(indices: list[int], shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(dataset, indices),
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
            persistent_workers=args.num_workers > 0,
        )

    dataloader = loader(train_indices, shuffle=True)
    val_dataloader = loader(val_indices, shuffle=False) if val_indices else None

    state = trainer.train(dataloader, val_dataloader)
    logger.info(
        "finished at step %d | %d samples | %.1f epochs | best val loss %.5f",
        state.step,
        state.samples_seen,
        state.epochs(len(train_indices)),
        state.best_val_loss,
    )


if __name__ == "__main__":
    main()
