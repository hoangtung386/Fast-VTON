"""Tests for the garment preprocessing path feeding DINOv2."""

import numpy as np
import pytest
from PIL import Image

from src.vton.garment_encoder import GARMENT_PAD_COLOUR, GarmentEncoder, pad_to_square

#: VITON-HD ships its product shots at this size.
VITON_CLOTH_SIZE = (768, 1024)


def _garment() -> Image.Image:
    """A red bar on white, standing in for a print near the top of the garment."""
    array = np.full((*VITON_CLOTH_SIZE[::-1], 3), 255, dtype=np.uint8)
    array[40:80, 300:460] = (255, 0, 0)  # near the neckline, first to be cropped away
    array[980:1020, 300:460] = (0, 0, 255)  # near the hem, cropped from the other end
    return Image.fromarray(array)


def test_pad_to_square_keeps_every_pixel() -> None:
    padded = pad_to_square(_garment())

    assert padded.size == (1024, 1024)
    array = np.asarray(padded)
    # Both marks survive, offset by the 128 px bar on the left.
    assert (array[40:80, 428:588] == (255, 0, 0)).all()
    assert (array[980:1020, 428:588] == (0, 0, 255)).all()
    assert tuple(array[0, 0]) == GARMENT_PAD_COLOUR


def test_pad_to_square_is_a_noop_on_squares() -> None:
    square = Image.new("RGB", (256, 256), (1, 2, 3))
    assert pad_to_square(square) is square


def test_pad_to_square_handles_landscape() -> None:
    assert pad_to_square(Image.new("RGB", (300, 100))).size == (300, 300)


@pytest.mark.slow
@pytest.mark.parametrize("resolution", [224, 336, 448])
def test_processor_output_matches_the_cache_token_count(resolution: int) -> None:
    """The cache is shaped from ``garment_resolution``; the processor must agree.

    These drifted apart once already: ``cache_specs`` honoured the config while the
    processor silently kept its own 224 px crop.
    """
    patch = 14
    processor = GarmentEncoder.image_processor(resolution=resolution)

    pixel_values = processor(images=[pad_to_square(_garment())], return_tensors="pt").pixel_values

    assert tuple(pixel_values.shape) == (1, 3, resolution, resolution)
    tokens = 1 + (pixel_values.shape[-1] // patch) * (pixel_values.shape[-2] // patch)
    assert tokens == 1 + (resolution // patch) ** 2
