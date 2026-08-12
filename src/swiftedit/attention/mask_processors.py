"""IP-Adapter attention processor wired to a :class:`MaskController`."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from einops import rearrange

from swiftedit.attention.mask_controller import MaskController
from swiftedit.attention.processors import (
    IPAttnProcessor2_0,
    finalize_attention,
    merge_heads,
    prepare_hidden_states,
    split_heads,
)


class IPAttnProcessor2_0WithIPMaskController(IPAttnProcessor2_0):  # noqa: N801
    """IP-Adapter processor that routes attention through ARaM when a controller is set.

    With ``controller`` unset this behaves exactly like :class:`IPAttnProcessor2_0`, so
    the module is safe both during training (where ARaM does not apply) and at inference
    (where it does).

    The parameter layout - ``to_k_ip`` and ``to_v_ip`` only - matches the released
    checkpoint, so instances load ``ip_adapter.bin`` without key remapping.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int | None = None,
        scale: float = 1.0,
        num_tokens: int = 4,
    ) -> None:
        super().__init__(hidden_size, cross_attention_dim, scale, num_tokens)
        self.controller: MaskController | None = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Attend with ARaM region rescaling, or fall back to the plain path."""
        if self.controller is None:
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )

        residual = hidden_states
        hidden_states, attention_mask, spatial_shape, batch_size = prepare_hidden_states(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            encoder_hidden_states, ip_hidden_states = self.split_conditions(encoder_hidden_states)
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // attn.heads
        query = split_heads(query, batch_size, attn.heads, head_dim)
        key = split_heads(key, batch_size, attn.heads, head_dim)
        value = split_heads(value, batch_size, attn.heads, head_dim)

        if ip_hidden_states is None:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = merge_heads(hidden_states, batch_size, attn.heads, head_dim)
            return finalize_attention(attn, hidden_states.to(query.dtype), residual, spatial_shape)

        ip_key = split_heads(self.to_k_ip(ip_hidden_states), batch_size, attn.heads, head_dim)
        ip_value = split_heads(self.to_v_ip(ip_hidden_states), batch_size, attn.heads, head_dim)

        def flatten(tensor: torch.Tensor) -> torch.Tensor:
            """Collapse heads into the batch axis: ``(b, h, n, d) -> (b * h, n, d)``."""
            return rearrange(tensor, "b h n d -> (b h) n d", h=attn.heads)

        prompt_states = self.controller.fwd_no_ip(
            flatten(query), flatten(key), flatten(value), attn.heads, scale=attn.scale
        )
        image_states = self.controller.fwd_ip(
            flatten(query), flatten(ip_key), flatten(ip_value), attn.heads, scale=attn.scale
        )

        # The released implementation adds the two branches without applying self.scale
        # here; ARaM's own gains already control the image contribution.
        return finalize_attention(attn, prompt_states + image_states, residual, spatial_shape)
