"""Precompute the Stage 1 feature cache from VITON-HD."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.models import AuxiliaryModel, InverseModel
from src.vton import GarmentEncoder, build_cache
from src.vton.config import VITON_HD_DATASET, CheckpointConfig, DataConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=VITON_HD_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="cache only the first N samples")
    parser.add_argument("--weights-root", type=Path, default=CheckpointConfig().root)
    parser.add_argument("--null-embedding", type=Path, default=Path("outputs/null_embedding.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vton_cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=DataConfig().height)
    parser.add_argument("--width", type=int, default=DataConfig().width)
    parser.add_argument(
        "--skip-garment-features",
        action="store_true",
        help="omit the ~6 GB DINOv2 cache and run the backbone during training instead",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Load the frozen modules, then materialise the cache."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    from datasets import load_dataset  # imported lazily: only needed for cache building

    checkpoints = CheckpointConfig(root=args.weights_root)
    checkpoints.validate()

    if not args.null_embedding.exists():
        raise FileNotFoundError(
            f"{args.null_embedding} not found; run scripts/make_null_embedding.py first"
        )

    data = DataConfig(
        dataset_id=args.dataset, split=args.split, height=args.height, width=args.width
    )
    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    logger.info("loaded %d samples from %s", len(dataset), args.dataset)

    inverse_model = InverseModel(
        checkpoints.inversion_dir, device=args.device, load_text_encoder=False
    )
    aux_model = AuxiliaryModel(device=args.device, load_text_encoder=False)

    garment_encoder = None
    if not args.skip_garment_features:
        garment_encoder = GarmentEncoder(checkpoints.garment_backbone).to(args.device).eval()

    build_cache(
        dataset=dataset,
        output_dir=args.output,
        inverse_model=inverse_model,
        aux_model=aux_model,
        data=data,
        null_embedding=torch.load(args.null_embedding, map_location="cpu"),
        garment_encoder=garment_encoder,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
