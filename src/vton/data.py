"""Dataset over the precomputed Stage 1 cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.vton.precompute import CACHE_META_FILE

#: Arrays every cache must provide.
REQUIRED_ARRAYS: tuple[str, ...] = (
    "z_person",
    "z_agnostic",
    "inverted_noise",
    "mask_latent",
    "clip_image_embeds",
)


class CachedVtonDataset(Dataset):
    """Serve precomputed latents, masks and conditioning features.

    Arrays are opened as read-only memmaps, so the 6 GB garment-feature file never has
    to fit in RAM and multiple dataloader workers share the page cache.

    Args:
        cache_dir: Directory written by :func:`src.vton.precompute.build_cache`.
        require_garment_features: Fail fast if the optional DINOv2 cache is absent.
    """

    def __init__(self, cache_dir: str | Path, require_garment_features: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / CACHE_META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"no cache metadata at {meta_path}; run the precompute step")

        self.meta = json.loads(meta_path.read_text())
        self.num_samples: int = self.meta["num_samples"]

        available = set(self.meta["arrays"])
        missing = [name for name in REQUIRED_ARRAYS if name not in available]
        if missing:
            raise ValueError(f"cache at {self.cache_dir} is missing arrays: {missing}")

        self.has_garment_features = "garment_features" in available
        if require_garment_features and not self.has_garment_features:
            raise ValueError(
                f"cache at {self.cache_dir} has no garment_features; rebuild it with a "
                "garment encoder or pass require_garment_features=False"
            )

        names = [*REQUIRED_ARRAYS]
        if self.has_garment_features:
            names.append("garment_features")
        self._arrays = {
            name: np.load(self.cache_dir / f"{name}.npy", mmap_mode="r") for name in names
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # np.asarray materialises the slice; .copy() detaches it from the read-only
        # memmap so torch.from_numpy does not emit a non-writable-array warning.
        return {
            name: torch.from_numpy(np.asarray(array[index]).copy())
            for name, array in self._arrays.items()
        }
