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
            removed. The reduction over channels is a max, so a single noisy channel is
            enough to light a pixel up - the effective noise floor is higher than the
            number suggests. Raise it if the mask bleeds into unchanged areas.
        morph_kernel: Diameter of the elliptical structuring element used to drop
            speckle and then close holes. Raise it if the mask is perforated, but note
            that an opening this wide can also sever a thin forearm, which
            ``keep_largest_component`` would then discard entirely.
        keep_largest_component: Discard all but the largest connected region.

    Returns:
        ``uint8`` array of shape ``(height, width)`` with values in ``{0, 1}``.
    """
    if morph_kernel < 1:
        raise ValueError("morph_kernel must be >= 1")

    left = np.asarray(person.convert("RGB").resize(size, Image.BILINEAR), dtype=np.int16)
    right = np.asarray(agnostic.convert("RGB").resize(size, Image.BILINEAR), dtype=np.int16)
    mask = (np.abs(left - right).max(axis=2) > diff_threshold).astype(np.uint8)

    # Open before close. Closing first dilates the speckle that JPEG re-encoding leaves
    # around the garment and welds it onto the real region; the later open cannot undo a
    # merge, and `keep_largest_component` then preserves the inflated blob. Removing the
    # speckle first and only then filling holes keeps the mask on the garment.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if keep_largest_component:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)

    return mask


def mask_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame the mask occupies.

    A weak signal on its own: it cannot tell a legitimately large mask (baggy garment,
    outstretched arms) from one that has crept down over the trousers. Pair it with
    :func:`mask_vertical_extent`, which answers that question directly.
    """
    return float(mask.mean())


def mask_vertical_extent(mask: np.ndarray) -> tuple[float, float]:
    """Topmost and bottommost mask row, as fractions of the frame height.

    This is the check that matters for try-on. The garment region must stop at the
    waist: a mask reaching the bottom of the frame is covering trousers, and Stage 1
    would then be asking the generator to invent legs from a token sequence that only
    ever saw a shirt.

    On VITON-HD's frontal upper-body crops a healthy mask starts below the head and
    ends around the waist. An empty mask reports ``(0.0, 0.0)``.

    Returns:
        ``(top, bottom)`` in ``[0, 1]``, measured from the top of the frame.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return 0.0, 0.0
    height = mask.shape[0]
    return float(rows[0]) / height, float(rows[-1] + 1) / height
