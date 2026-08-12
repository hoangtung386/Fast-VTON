"""Run one-step text-guided editing on a single image."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from torchvision.utils import save_image

from src.constants import DEFAULT_WEIGHTS_ROOT
from src.models import AuxiliaryModel, InverseModel, IPSBV2Model
from src.pipelines import EditConfig, edit_image
from src.vton.config import CheckpointConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("assets/imgs_demo/woman_face.jpg"))
    parser.add_argument("--source-prompt", default="woman", help="may be empty")
    parser.add_argument("--edit-prompt", default="Taylor Swift")
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--output", type=Path, default=Path("result.png"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale-text", type=float, default=EditConfig.scale_text)
    parser.add_argument("--scale-edit", type=float, default=EditConfig.scale_edit)
    parser.add_argument("--scale-non-edit", type=float, default=EditConfig.scale_non_edit)
    parser.add_argument("--mask-threshold", type=float, default=EditConfig.mask_threshold)
    return parser.parse_args()


def main() -> None:
    """Load the models, edit the image and write the result."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    checkpoints = CheckpointConfig(root=args.weights_root)
    checkpoints.validate()

    inverse_model = InverseModel(checkpoints.inversion_dir, device=args.device)
    aux_model = AuxiliaryModel(device=args.device)
    generator = IPSBV2Model(
        checkpoints.generator_dir,
        checkpoints.ip_adapter_path,
        aux_model,
        device=args.device,
        with_ip_mask_controller=True,
    )

    config = EditConfig(
        scale_text=args.scale_text,
        scale_edit=args.scale_edit,
        scale_non_edit=args.scale_non_edit,
        mask_threshold=args.mask_threshold,
    )

    started = time.monotonic()
    result = edit_image(
        args.image, args.source_prompt, args.edit_prompt, inverse_model, generator, config
    )
    logger.info(
        "edited %r -> %r in %.3fs", args.source_prompt, args.edit_prompt, time.monotonic() - started
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, args.output)
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
