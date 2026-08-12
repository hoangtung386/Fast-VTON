"""Tests for the agnostic-difference mask builder."""

import numpy as np
import pytest
from PIL import Image

from swiftedit.vton.masking import build_agnostic_mask, mask_coverage

SIZE = (64, 96)  # (width, height)


def _person() -> Image.Image:
    """A textured frame so the difference is not trivially uniform."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (96, 64, 3), dtype=np.uint8))


def _agnostic_from(person: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Return ``person`` with ``box`` (top, bottom, left, right) painted grey."""
    array = np.asarray(person).copy()
    top, bottom, left, right = box
    array[top:bottom, left:right] = 128
    return Image.fromarray(array)


def test_recovers_painted_region() -> None:
    person = _person()
    agnostic = _agnostic_from(person, (20, 70, 10, 50))

    mask = build_agnostic_mask(person, agnostic, SIZE, diff_threshold=12, morph_kernel=3)

    assert mask.shape == (96, 64)
    assert set(np.unique(mask)).issubset({0, 1})
    # The painted rectangle spans 50x40 of 96x64 -> ~0.33 coverage.
    assert 0.25 < mask_coverage(mask) < 0.40
    assert mask[45, 30] == 1, "centre of the painted box must be inside the mask"
    assert mask[5, 5] == 0, "untouched corner must stay outside the mask"


def test_identical_images_produce_empty_mask() -> None:
    person = _person()
    mask = build_agnostic_mask(person, person, SIZE)
    assert mask.sum() == 0


def test_largest_component_filter_drops_speckle() -> None:
    person = _person()
    array = np.asarray(person).copy()
    array[20:70, 10:50] = 128  # main region
    array[2:4, 60:62] = 128  # isolated speckle
    agnostic = Image.fromarray(array)

    kept = build_agnostic_mask(person, agnostic, SIZE, morph_kernel=3, keep_largest_component=True)
    assert kept[3, 61] == 0, "speckle must be removed when filtering is on"


def test_rejects_invalid_kernel() -> None:
    person = _person()
    with pytest.raises(ValueError, match="morph_kernel"):
        build_agnostic_mask(person, person, SIZE, morph_kernel=0)
