"""Virtual try-on adaptation of SwiftEdit.

The garment replaces the text prompt in the generator's prompt branch, the agnostic
person image drives the IP-Adapter branch, and an explicit inpainting mask replaces the
self-guided mask. See ``docs/VTON_PLAN.md`` for the reasoning and the evidence behind
each choice.
"""

from src.vton.config import CheckpointConfig, DataConfig, Stage1Config
from src.vton.data import CachedVtonDataset
from src.vton.export import (
    BundleManifest,
    LoadedBundle,
    bundle_summary,
    export_bundle,
    load_bundle,
)
from src.vton.freezing import TrainableGroups, configure_trainable, count_parameters
from src.vton.garment_encoder import GarmentEncoder
from src.vton.hub import HUB_ASSETS, prefetch_hub_assets
from src.vton.masking import build_agnostic_mask, mask_coverage, mask_vertical_extent
from src.vton.precompute import build_cache
from src.vton.trainer import Stage1Trainer, masked_latent_loss

__all__ = [
    "HUB_ASSETS",
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
    "mask_vertical_extent",
    "masked_latent_loss",
    "prefetch_hub_assets",
]
