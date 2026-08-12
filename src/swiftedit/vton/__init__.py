"""Virtual try-on adaptation of SwiftEdit.

The garment replaces the text prompt in the generator's prompt branch, the agnostic
person image drives the IP-Adapter branch, and an explicit inpainting mask replaces the
self-guided mask. See ``docs/VTON_PLAN.md`` for the reasoning and the evidence behind
each choice.
"""

from swiftedit.vton.config import CheckpointConfig, DataConfig, Stage1Config
from swiftedit.vton.data import CachedVtonDataset
from swiftedit.vton.export import (
    BundleManifest,
    LoadedBundle,
    bundle_summary,
    export_bundle,
    load_bundle,
)
from swiftedit.vton.freezing import TrainableGroups, configure_trainable, count_parameters
from swiftedit.vton.garment_encoder import GarmentEncoder
from swiftedit.vton.masking import build_agnostic_mask, mask_coverage
from swiftedit.vton.precompute import build_cache
from swiftedit.vton.trainer import Stage1Trainer, masked_latent_loss

__all__ = [
    "BundleManifest",
    "CachedVtonDataset",
    "CheckpointConfig",
    "DataConfig",
    "GarmentEncoder",
    "LoadedBundle",
    "Stage1Config",
    "Stage1Trainer",
    "TrainableGroups",
    "build_agnostic_mask",
    "build_cache",
    "bundle_summary",
    "configure_trainable",
    "count_parameters",
    "export_bundle",
    "load_bundle",
    "mask_coverage",
    "masked_latent_loss",
]
