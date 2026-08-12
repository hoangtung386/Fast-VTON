"""Derive the try-on inpainting mask from a person / agnostic image pair."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def build_agnostic_mask(
    person: Image.Image,
    agnostic: Image.Image,
    size: tuple[int, int],
    diff_threshold: int = 12,
    morph_kernel: int = 9,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Recover the region the agnostic image removed from the person image.

    VITON-HD's ``agnostic`` view is the person photograph with the garment area painted
    out, so the mask is exactly where the two differ. Recovering it this way avoids
    hard-coding LIP/CIHP label indices, which vary between parsing releases and are a
    common source of silent mislabelling. It also works on mirrors that ship
    ``agnostic`` but omit the parse maps.

    Args:
        person: Original person photograph.
        agnostic: Same photograph with the garment region removed.
        size: Target ``(width, height)``.
        diff_threshold: Per-channel intensity delta above which a pixel counts as
            removed. Raise it if the mask bleeds into unchanged areas.
        morph_kernel: Diameter of the elliptical structuring element used to close
            holes and drop speckle. Raise it if the mask is perforated.
        keep_largest_component: Discard all but the largest connected region.

    Returns:
        ``uint8`` array of shape ``(height, width)`` with values in ``{0, 1}``.
    """
    if morph_kernel < 1:
        raise ValueError("morph_kernel must be >= 1")

    left = np.asarray(person.convert("RGB").resize(size, Image.BILINEAR), dtype=np.int16)
    right = np.asarray(agnostic.convert("RGB").resize(size, Image.BILINEAR), dtype=np.int16)
    mask = (np.abs(left - right).max(axis=2) > diff_threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if keep_largest_component:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)

    return mask


def mask_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame the mask occupies, useful as a sanity check.

    Healthy VITON-HD upper-body masks land roughly in ``0.10 - 0.35``. Values far
    outside that band usually mean ``diff_threshold`` needs adjusting.
    """
    return float(mask.mean())
