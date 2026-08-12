"""Resilient prefetch of the Hugging Face assets Stage 1 pulls at run time.

Building the cache downloads roughly 4 GB from the Hub before it touches a single
sample: two Stable Diffusion VAEs, the IP-Adapter CLIP vision tower and the DINOv2
backbone. ``from_pretrained`` has no retry, so one dropped connection aborts the whole
job - and Colab drops connections. Fetching everything up front, with backoff, turns a
lost half hour into a few extra seconds.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from huggingface_hub import snapshot_download

from src.constants import (
    DINOV2_REPO,
    IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER,
    IP_ADAPTER_REPO,
    SD21_BASE_REPO,
    SD_TURBO_REPO,
)

logger = logging.getLogger(__name__)

#: Repo id paired with glob patterns covering exactly what the Stage 1 pipeline loads.
#: The patterns are deliberately narrow: ``h94/IP-Adapter`` holds many gigabytes of
#: adapters this project never touches, and most diffusers repos ship both a ``.bin``
#: and a ``.safetensors`` copy of every weight, plus an fp16 variant alongside each.
#: Scheduler config plus the fp32 VAE weights, shared by both Stable Diffusion repos.
_SD_PATTERNS: tuple[str, ...] = (
    "scheduler/*.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)

HUB_ASSETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (SD_TURBO_REPO, _SD_PATTERNS),
    (SD21_BASE_REPO, _SD_PATTERNS),
    (
        IP_ADAPTER_REPO,
        (
            f"{IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER}/*.json",
            f"{IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER}/*.safetensors",
        ),
    ),
    (DINOV2_REPO, ("*.json", "*.safetensors")),
)


def prefetch_hub_assets(
    assets: Sequence[tuple[str, Sequence[str]]] = HUB_ASSETS,
    attempts: int = 5,
    initial_backoff: float = 2.0,
) -> None:
    """Download every asset into the local Hub cache, retrying on transport errors.

    Best effort by design. A pattern that misses a file the loader wants costs nothing,
    because ``from_pretrained`` still falls back to the network for whatever is absent;
    the point is to move the bulk of the transfer to a place where failing is cheap.
    Partial downloads resume, so a retry does not restart from zero.

    Args:
        assets: ``(repo_id, allow_patterns)`` pairs.
        attempts: Tries per repo before giving up.
        initial_backoff: Seconds to wait after the first failure, doubling each retry.

    Raises:
        Exception: Whatever the final attempt raised, if a repo never succeeds.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    for repo_id, patterns in assets:
        backoff = initial_backoff
        for attempt in range(1, attempts + 1):
            try:
                snapshot_download(repo_id, allow_patterns=list(patterns))
            except Exception as error:
                if attempt == attempts:
                    logger.error("giving up on %s after %d attempts", repo_id, attempts)
                    raise
                logger.warning(
                    "%s failed (%s: %s), retrying in %.0fs [attempt %d/%d]",
                    repo_id,
                    type(error).__name__,
                    error,
                    backoff,
                    attempt,
                    attempts,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.info("prefetched %s", repo_id)
                break
