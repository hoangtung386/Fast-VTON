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

    Measured over 200 ``forgeml/viton_hd`` train pairs, healthy masks land at a median
    of 0.35 and span roughly ``0.20 - 0.63``. The band is this wide, and this high,
    because VITON-HD's ``agnostic`` view is ``agnostic-v3.2``: it paints out a generous
    bounding region - garment, both arms, hair falling on the shoulders, plus a halo of
    plain backdrop around the silhouette - rather than a tight body mask. Inspecting the
    highest-coverage samples confirms the trousers stay outside it.

    Coverage alone cannot separate a legitimately large mask from one creeping down over
    the trousers; both show up as a big number. :func:`mask_vertical_extent` is the
    measurement that tells them apart.
    """
    return float(mask.mean())


def mask_vertical_extent(
    mask: np.ndarray, min_row_fraction: float = 0.10
) -> tuple[float, float]:
    """Topmost and bottommost *substantial* mask row, as fractions of frame height.

    This is the check that matters for try-on: the region must stop around the
    waistband. A mask running to the bottom of the frame is covering trousers, and
    Stage 1 would then be asking the generator to invent them from a token sequence that
    only ever saw a shirt.

    A row counts only once at least ``min_row_fraction`` of it is masked, so a few stray
    pixels cannot set the answer on their own.

    Read the bottom figure with care. Measured over 30 VITON-HD pairs at the default
    threshold, ``top`` sits at 0.20 (p05 0.14) and ``bottom`` at 0.90 (p05 0.79), with
    20% of masks reaching 1.00. That is not the mask swallowing the trousers: the
    agnostic halo runs down the plain backdrop either side of the legs, while the
    trousers themselves - the middle third of the lower quarter of the frame - stay only
    9% covered at the median. A mask that has genuinely crept onto the trousers shows up
    there, not in this number.

    Args:
        mask: Binary mask, ``(height, width)``.
        min_row_fraction: Share of a row that must be masked for it to count.

    Returns:
        ``(top, bottom)`` in ``[0, 1]`` measured from the top of the frame, or
        ``(0.0, 0.0)`` when no row qualifies.
    """
    if not 0.0 < min_row_fraction <= 1.0:
        raise ValueError("min_row_fraction must lie in (0, 1]")

    rows = np.flatnonzero(mask.mean(axis=1) >= min_row_fraction)
    if rows.size == 0:
        return 0.0, 0.0
    height = mask.shape[0]
    return float(rows[0]) / height, float(rows[-1] + 1) / height
