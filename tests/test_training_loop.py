"""Integration tests for the Stage 1 loop, on CPU with stub modules.

These exist because the failure that cost a full A100 run was not a crash: the loop ran
happily at an effective batch size of one, reporting a healthy steps/s, and nobody could
tell from the log that it was seeing three epochs instead of fifty-five.
"""

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from torch.utils.data import DataLoader, Dataset

from src.models.generator import install_ip_attn_processors
from src.models.projection import ImageProjModel
from src.vton.config import Stage1Config
from src.vton.freezing import TrainableGroups
from src.vton.trainer import Stage1Trainer

CROSS_DIM = 8
CLIP_DIM = 12
LATENT = (4, 8, 8)


class _StubGenerator(nn.Module):
    """A UNet small enough to train on CPU, with the attributes the trainer reads."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=9,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(8, 8),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=CROSS_DIM,
            attention_head_dim=2,
            norm_num_groups=4,
        )
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=CROSS_DIM,
            clip_embeddings_dim=CLIP_DIM,
            clip_extra_context_tokens=4,
        )
        self.adapter_modules = install_ip_attn_processors(
            self.unet, with_mask_controller=False, device="cpu", seed_from_unet=False
        )
        self.device_ = torch.device("cpu")
        self.timestep = torch.zeros(1, dtype=torch.int64)
        self.alpha_t = torch.ones(1, 1, 1, 1) * 0.5
        self.sigma_t = torch.ones(1, 1, 1, 1) * 0.5

    def forward_train(self, noisy_latent, masked_latent, mask_latent, prompt_tokens, ip_tokens):
        sample = torch.cat([noisy_latent, masked_latent, mask_latent], dim=1)
        condition = torch.cat([prompt_tokens, ip_tokens], dim=1)
        return self.unet(sample, self.timestep, condition).sample


class _StubDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "z_person": torch.randn(*LATENT, generator=generator),
            "z_agnostic": torch.randn(*LATENT, generator=generator),
            "inverted_noise": torch.randn(*LATENT, generator=generator),
            "mask_latent": torch.ones(1, 8, 8),
            "clip_image_embeds": torch.randn(CLIP_DIM, generator=generator),
            "garment_features": torch.randn(5, CROSS_DIM, generator=generator),
        }


def _trainer(tmp_path: Path, **overrides) -> Stage1Trainer:
    from src.vton.garment_encoder import GarmentEncoder

    settings = {
        "max_steps": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "log_every": 1,
        "eval_every": 2,
        "preview_every": 0,
        "checkpoint_every": 100,
        "mixed_precision": "no",
        "gradient_checkpointing": False,
        "output_dir": tmp_path,
        **overrides,
    }
    return Stage1Trainer(
        _StubGenerator(),
        GarmentEncoder.for_cached_features(hidden_size=CROSS_DIM, cross_attention_dim=CROSS_DIM),
        Stage1Config(**settings),
        groups=TrainableGroups(garment_projection=True),
    )


def test_epoch_counter_exposes_a_mis_set_batch_size(tmp_path: Path) -> None:
    """The number that would have caught the batch-size-1 run in its first minute."""
    trainer = _trainer(tmp_path, max_steps=4, batch_size=2)
    trainer.train(DataLoader(_StubDataset(8), batch_size=2, drop_last=True))

    assert trainer.state.samples_seen == 8
    assert trainer.state.epochs(8) == pytest.approx(1.0)
    # Same step count, a quarter of the data: the epoch figure separates them.
    assert trainer.state.epochs(32) == pytest.approx(0.25)


def test_log_reports_samples_per_second_and_epochs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trainer(tmp_path, max_steps=2, batch_size=2)
    with caplog.at_level(logging.INFO, logger="src.vton.trainer"):
        trainer.train(DataLoader(_StubDataset(8), batch_size=2, drop_last=True))

    step_lines = [r.getMessage() for r in caplog.records if "step 1/" in r.getMessage()]
    assert step_lines, "no step line was logged"
    assert "samples/s" in step_lines[0]
    assert "epoch" in step_lines[0]


def test_validation_loss_is_recorded(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_steps=2, eval_every=2)
    train_loader = DataLoader(_StubDataset(8), batch_size=2, drop_last=True)
    val_loader = DataLoader(_StubDataset(4), batch_size=2)

    trainer.train(train_loader, val_loader)

    assert trainer.state.best_val_loss < float("inf")


def test_evaluate_leaves_the_model_in_train_mode(tmp_path: Path) -> None:
    """A stray eval() would silently freeze dropout and norm updates for the rest of the run."""
    trainer = _trainer(tmp_path)
    trainer.generator.train()
    trainer.evaluate(DataLoader(_StubDataset(4), batch_size=2))
    assert trainer.generator.training


def test_preview_is_skipped_without_a_vae(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    batch = next(iter(DataLoader(_StubDataset(2), batch_size=2)))
    assert trainer.save_preview(batch, "x") is None
    assert not list(tmp_path.glob("*.png"))


def test_runs_without_a_garment_backbone(tmp_path: Path) -> None:
    """Cache-driven training must not need the 306 M DINOv2 tower resident."""
    trainer = _trainer(tmp_path, max_steps=2)
    assert trainer.garment_encoder.backbone is None
    trainer.train(DataLoader(_StubDataset(8), batch_size=2, drop_last=True))
    assert trainer.state.step == 2


def test_checkpoint_round_trip_preserves_counters(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_steps=2)
    trainer.train(DataLoader(_StubDataset(8), batch_size=2, drop_last=True))
    path = tmp_path / "final.pt"
    assert path.exists()

    restored = _trainer(tmp_path, max_steps=2)
    restored.load_checkpoint(path)
    assert restored.state.step == trainer.state.step
    assert restored.state.samples_seen == trainer.state.samples_seen


def test_loss_history_survives_the_checkpoint(tmp_path: Path) -> None:
    """The gate reads these numbers instead of asking a human to retype them."""
    trainer = _trainer(tmp_path, max_steps=4, log_every=1, eval_every=2)
    trainer.train(
        DataLoader(_StubDataset(8), batch_size=2, drop_last=True),
        DataLoader(_StubDataset(4), batch_size=2),
    )
    assert len(trainer.state.loss_history) == 4
    assert [step for step, _ in trainer.state.loss_history] == [1, 2, 3, 4]
    assert [step for step, _ in trainer.state.val_history] == [2, 4]

    restored = _trainer(tmp_path, max_steps=4, log_every=1, eval_every=2)
    restored.load_checkpoint(tmp_path / "final.pt")
    assert restored.state.loss_history == trainer.state.loss_history
    assert restored.state.val_history == trainer.state.val_history
    assert restored.state.best_val_loss == trainer.state.best_val_loss


def test_load_checkpoint_tolerates_a_history_free_payload(tmp_path: Path) -> None:
    """Checkpoints from the previous trainer must still resume, just without a curve."""
    trainer = _trainer(tmp_path, max_steps=2)
    trainer.train(DataLoader(_StubDataset(8), batch_size=2, drop_last=True))

    payload = torch.load(tmp_path / "final.pt", map_location="cpu")
    for key in ("loss_history", "val_history", "best_val_loss", "samples_seen"):
        payload.pop(key)
    torch.save(payload, tmp_path / "legacy.pt")

    restored = _trainer(tmp_path, max_steps=2)
    restored.load_checkpoint(tmp_path / "legacy.pt")
    assert restored.state.loss_history == []
    assert restored.state.best_val_loss == float("inf")
