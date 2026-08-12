"""Cache the empty-prompt CLIP embedding the inversion network expects.

Virtual try-on hands the prompt branch to the garment encoder, so the text tower has no
job left except producing this one constant. Precomputing it lets both text encoders be
dropped, saving roughly 1.4 GB at inference.

The embedding *must* come from ``stabilityai/sd-turbo``: that is the text encoder
``InverseModel`` was trained against. Taking it from SD 2.1-base instead produces a
plausible-looking tensor that silently degrades inversion quality.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, CLIPTextModel

from src.constants import SD_TURBO_REPO
from src.utils.text import tokenize_captions

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=SD_TURBO_REPO, help="text encoder source")
    parser.add_argument("--prompt", default="", help="prompt to encode; empty by default")
    parser.add_argument("--output", type=Path, default=Path("outputs/null_embedding.pt"))
    return parser.parse_args()


def main() -> None:
    """Encode the prompt once and save the resulting hidden states."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.model != SD_TURBO_REPO:
        logger.warning(
            "encoding with %s, not %s - the inversion network was trained against the "
            "latter and will silently underperform with a mismatched embedding",
            args.model,
            SD_TURBO_REPO,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").eval()

    input_ids = tokenize_captions(tokenizer, [args.prompt])
    with torch.no_grad():
        embedding = text_encoder(input_ids)[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding, args.output)
    logger.info("saved %s with shape %s", args.output, tuple(embedding.shape))


if __name__ == "__main__":
    main()
