"""Attention Rescaling for Mask-aware editing (ARaM).

Implements Eq. 9 of the SwiftEdit paper: the attention output is split into an edit
region and a non-edit region by a binary mask, and each region is rescaled independently
so edit strength can be traded against background preservation.

Two deliberate deviations from the released implementation, both behaviour-preserving:

* ``attn_batch`` no longer takes a pre-computed ``sim`` argument. The original signature
  accepted one but overwrote it on the first line, so it was dead.
* Grid shapes are derived from the mask's aspect ratio instead of ``int(sqrt(n))``.
  Identical for square latents, and correct for the 3:4 latents used by virtual try-on.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange


class MaskController:
    """Rescales attention per region using a binary editing mask.

    Args:
        mask: Binary mask over the *latent* grid, shape ``(height, width)``. Ones mark
            the region to edit.
        scale_text_hiddenstate: ``s_y`` in Eq. 9 - gain on the prompt branch.
        scale_ip_fg: ``s_edit`` - gain on the image branch inside the edit region.
        scale_ip_bg: ``s_non-edit`` - gain on the image branch outside it.

    Note:
        The mask is broadcast along the *query* axis of the similarity matrix, not the
        key axis. Query positions outside the selected region receive a constant
        ``finfo.min`` row, which softmax turns into a uniform distribution. That is the
        behaviour of the released implementation and is preserved verbatim.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        scale_text_hiddenstate: float | None = None,
        scale_ip_fg: float = 0.0,
        scale_ip_bg: float = 1.0,
    ) -> None:
        if mask.ndim != 2:
            raise ValueError(f"mask must be a 2-D latent grid, got shape {tuple(mask.shape)}")
        self.mask_s = mask
        self.scale_text_hiddenstate = scale_text_hiddenstate
        self.scale_ip_fg = scale_ip_fg
        self.scale_ip_bg = scale_ip_bg

    # ----------------------------------------------------------------- helpers

    def spatial_shape(self, num_tokens: int) -> tuple[int, int]:
        """Infer the ``(height, width)`` of an attention grid holding ``num_tokens``."""
        base_h, base_w = self.mask_s.shape[-2:]
        scale = math.sqrt(num_tokens / (base_h * base_w))
        height, width = round(base_h * scale), round(base_w * scale)
        if height * width != num_tokens:
            raise ValueError(
                f"cannot map {num_tokens} attention tokens onto a grid with the aspect "
                f"ratio of the {base_h}x{base_w} mask"
            )
        return height, width

    def resized_mask(self, num_tokens: int) -> torch.Tensor:
        """Return the mask resampled to ``num_tokens`` positions, shape ``(n, 1)``."""
        height, width = self.spatial_shape(num_tokens)
        mask = F.interpolate(self.mask_s[None, None], (height, width), mode="nearest")
        return mask.reshape(-1, 1)

    @staticmethod
    def _head_slice(tensor: torch.Tensor, index: int, num_heads: int) -> torch.Tensor:
        """Select the ``index``-th batch element from a head-flattened tensor."""
        return tensor[index * num_heads : (index + 1) * num_heads]

    # --------------------------------------------------------------- attention

    def attn_batch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        *,
        scale: float,
        is_mask_attn: bool = False,
    ) -> torch.Tensor:
        """Run one attention batch, optionally splitting it into two masked halves.

        When ``is_mask_attn`` is set the output is twice as long along dim 0: the first
        half attends with the edit region suppressed, the second with it retained.
        """
        batch = q.shape[0] // num_heads
        num_tokens = q.shape[1]

        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        if is_mask_attn:
            height, width = self.spatial_shape(num_tokens)
            mask = F.interpolate(self.mask_s[None, None], (height, width), mode="nearest")
            mask = mask.flatten().unsqueeze(0).unsqueeze(-1)

            neg_inf = torch.finfo(sim.dtype).min
            sim_fg = sim + mask.masked_fill(mask == 1, neg_inf)
            sim_bg = sim + mask.masked_fill(mask == 0, neg_inf)
            sim = torch.cat([sim_fg, sim_bg], dim=0)

        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=batch, h=num_heads)

    def fwd_no_ip(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Prompt branch: ``s_y * Attn(Q, K_y, V_y)``.

        Note:
            For a two-element batch the released code applies ``s_y`` globally rather
            than only inside the mask as written in Eq. 9. That divergence is preserved
            here; with the default ``scale_text_hiddenstate=1`` it is a no-op anyway.
            The three-element branch does apply the masked form.
        """
        batch = q.shape[0] // num_heads
        out_source = self.attn_batch(
            self._head_slice(q, 0, num_heads),
            self._head_slice(k, 0, num_heads),
            self._head_slice(v, 0, num_heads),
            num_heads,
            scale=scale,
        )

        if batch <= 2:
            out_target = self.attn_batch(
                q[-num_heads:], k[-num_heads:], v[-num_heads:], num_heads, scale=scale
            )
            if self.scale_text_hiddenstate:
                out_target = self.scale_text_hiddenstate * out_target
            return torch.cat([out_source, out_target], dim=0)

        if batch == 3:
            mask = self.resized_mask(q.shape[1])
            out_target1 = self.attn_batch(
                self._head_slice(q, 1, num_heads),
                self._head_slice(k, 1, num_heads),
                self._head_slice(v, 1, num_heads),
                num_heads,
                scale=scale,
            )
            out_target2 = self.attn_batch(
                q[-num_heads:], k[-num_heads:], v[-num_heads:], num_heads, scale=scale
            )
            if self.scale_text_hiddenstate:
                out_target1 = self.scale_text_hiddenstate * out_target1 * mask + out_target1 * (
                    1 - mask
                )
            return torch.cat([out_source, out_target1, out_target2], dim=0)

        raise ValueError(f"unsupported batch size {batch}; expected 1, 2 or 3")

    def fwd_ip(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Image branch: ``s_edit * M * Attn + s_non-edit * (1 - M) * Attn``."""
        batch = q.shape[0] // num_heads
        mask = self.resized_mask(q.shape[1])

        out_source = self.attn_batch(
            self._head_slice(q, 0, num_heads),
            self._head_slice(k, 0, num_heads),
            self._head_slice(v, 0, num_heads),
            num_heads,
            scale=scale,
        )

        if batch <= 2:
            out_target = self.attn_batch(
                q[-num_heads:],
                k[-num_heads:],
                v[-num_heads:],
                num_heads,
                scale=scale,
                is_mask_attn=True,
            )
            out_fg, _ = out_target.chunk(2, 0)
            blended = self.scale_ip_fg * out_fg * mask + self.scale_ip_bg * out_source * (1 - mask)
            return torch.cat([out_source, blended], dim=0)

        if batch == 3:
            out_target1 = self.attn_batch(
                self._head_slice(q, 1, num_heads),
                self._head_slice(k, 1, num_heads),
                self._head_slice(v, 1, num_heads),
                num_heads,
                scale=scale,
                is_mask_attn=True,
            )
            out_target2 = self.attn_batch(
                q[-num_heads:],
                k[-num_heads:],
                v[-num_heads:],
                num_heads,
                scale=scale,
                is_mask_attn=True,
            )
            out_fg1, _ = out_target1.chunk(2, 0)
            out_fg2, out_bg2 = out_target2.chunk(2, 0)

            blended1 = self.scale_ip_fg * ((out_fg1 + out_fg2) / 2) * mask + (
                self.scale_ip_bg * out_source * (1 - mask)
            )
            blended2 = out_fg2 * mask + out_bg2 * (1 - mask)
            return torch.cat([out_source, blended1, blended2], dim=0)

        raise ValueError(f"unsupported batch size {batch}; expected 1, 2 or 3")
