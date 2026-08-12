"""Download every Hugging Face asset Stage 1 needs, with retries.

Run this before ``build_vton_cache.py``: it front-loads ~4 GB of transfers so a dropped
connection costs a retry instead of the whole cache job.
"""

from __future__ import annotations

import argparse
import logging

from src.vton.hub import HUB_ASSETS, prefetch_hub_assets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=5, help="tries per repository")
    parser.add_argument("--initial-backoff", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Prefetch every Hub asset into the local cache."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    prefetch_hub_assets(attempts=args.attempts, initial_backoff=args.initial_backoff)
    print(f"\nall {len(HUB_ASSETS)} repositories are in the local Hub cache")


if __name__ == "__main__":
    main()
