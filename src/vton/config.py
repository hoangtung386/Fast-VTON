"""Typed configuration for the virtual try-on adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.constants import (
    DEFAULT_WEIGHTS_ROOT,
    DINOV2_REPO,
    GENERATOR_CHECKPOINT_DIR,
    INVERSION_CHECKPOINT_DIR,
    IP_ADAPTER_CHECKPOINT,
    VAE_SCALE_FACTOR,
)

#: Hugging Face dataset holding the official VITON-HD train split (11,647 pairs).
VITON_HD_DATASET: str = "forgeml/viton_hd"


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing settings.

    VITON-HD's standard low-resolution benchmark is 512x384, which also matches the 3:4
    aspect of the source photographs. Both dimensions are divisible by 8, so the
    convolutional UNet handles them without change.
    """

    dataset_id: str = VITON_HD_DATASET
    split: str = "train"
    height: int = 512
    width: int = 384
    mask_diff_threshold: int = 12
    mask_morph_kernel: int = 9
    garment_resolution: int = 224
    horizontal_flip: bool = False

    def __post_init__(self) -> None:
        for name in ("height", "width"):
            value = getattr(self, name)
            if value % VAE_SCALE_FACTOR:
                raise ValueError(f"{name}={value} must be divisible by {VAE_SCALE_FACTOR}")

    @property
    def latent_height(self) -> int:
        """Latent grid height."""
        return self.height // VAE_SCALE_FACTOR

    @property
    def latent_width(self) -> int:
        """Latent grid width."""
        return self.width // VAE_SCALE_FACTOR

    @property
    def pil_size(self) -> tuple[int, int]:
        """``(width, height)`` as PIL expects it."""
        return self.width, self.height


@dataclass(frozen=True)
class CheckpointConfig:
    """Filesystem layout of the pretrained SwiftEdit weights."""

    root: Path = DEFAULT_WEIGHTS_ROOT
    garment_backbone: str = DINOV2_REPO

    @property
    def inversion_dir(self) -> Path:
        """Directory holding the inversion UNet."""
        return self.root / INVERSION_CHECKPOINT_DIR

    @property
    def generator_dir(self) -> Path:
        """Directory holding the SwiftBrush v2 generator UNet."""
        return self.root / GENERATOR_CHECKPOINT_DIR

    @property
    def ip_adapter_path(self) -> Path:
        """Path to ``ip_adapter.bin``."""
        return self.root / IP_ADAPTER_CHECKPOINT

    def validate(self) -> None:
        """Raise :class:`FileNotFoundError` if any required artefact is missing."""
        missing = [
            str(path)
            for path in (self.inversion_dir, self.generator_dir, self.ip_adapter_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "missing SwiftEdit checkpoints: " + ", ".join(missing) + "; see README"
            )


@dataclass(frozen=True)
class Stage1Config:
    """Hyper-parameters for Stage 1 (garment branch warm-up).

    Stage 1 keeps the generator, the inversion network and every frozen encoder fixed,
    training only the garment projection, the prompt-branch K/V matrices, the widened
    ``conv_in`` and the IP projection. That is roughly 33 M of 1.76 B parameters.
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_steps: int = 40_000
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    mask_loss_weight: float = 4.0
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    seed: int = 42

    checkpoint_every: int = 1_000
    log_every: int = 50
    output_dir: Path = Path("outputs/vton_stage1")
    cache_path: Path = Path("outputs/vton_cache.pt")

    data: DataConfig = field(default_factory=DataConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(f"unsupported mixed_precision {self.mixed_precision!r}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie in [0, 1)")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size and gradient_accumulation_steps must be >= 1")

    @property
    def effective_batch_size(self) -> int:
        """Samples per optimiser step."""
        return self.batch_size * self.gradient_accumulation_steps
