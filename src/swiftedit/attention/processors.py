# The code in this file originates from
# https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py
# and is released under an Apache 2.0 license, itself modified from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
#
# The code is modified as follows:
# - non relevant classes have been removed for brevity
# - type annotations, docstrings and PEP 8 formatting have been applied
# - the shared pre/post-processing steps were factored into module-level helpers
#
# -- original code follows this line --

"""Baseline attention processors: plain SDPA and the IP-Adapter decoupled variant."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention

from swiftedit.constants import IP_NUM_TOKENS


def require_sdpa() -> None:
    """Raise if the running PyTorch lacks ``scaled_dot_product_attention``."""
    if not hasattr(F, "scaled_dot_product_attention"):
        raise ImportError(
            "This attention processor requires PyTorch 2.0 or newer for "
            "torch.nn.functional.scaled_dot_product_attention."
        )


def prepare_hidden_states(
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    temb: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, ...] | None, int]:
    """Apply the standard pre-attention normalisation and flattening.

    Returns:
        A tuple of ``(hidden_states, attention_mask, spatial_shape, batch_size)`` where
        ``spatial_shape`` is ``(channel, height, width)`` for 4-D inputs and ``None``
        otherwise. The caller must restore that shape after attention.
    """
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    spatial_shape: tuple[int, ...] | None = None
    if hidden_states.ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        spatial_shape = (channel, height, width)
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects (batch, heads, source_length, target_length).
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    return hidden_states, attention_mask, spatial_shape, batch_size


def finalize_attention(
    attn: Attention,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    spatial_shape: tuple[int, ...] | None,
) -> torch.Tensor:
    """Apply the output projection, restore the spatial layout and add the residual."""
    hidden_states = attn.to_out[0](hidden_states)  # linear projection
    hidden_states = attn.to_out[1](hidden_states)  # dropout

    if spatial_shape is not None:
        channel, height, width = spatial_shape
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    return hidden_states / attn.rescale_output_factor


def split_heads(tensor: torch.Tensor, batch_size: int, heads: int, head_dim: int) -> torch.Tensor:
    """Reshape ``(batch, seq, inner)`` to ``(batch, heads, seq, head_dim)``."""
    return tensor.view(batch_size, -1, heads, head_dim).transpose(1, 2)


def merge_heads(tensor: torch.Tensor, batch_size: int, heads: int, head_dim: int) -> torch.Tensor:
    """Reshape ``(batch, heads, seq, head_dim)`` back to ``(batch, seq, inner)``."""
    return tensor.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)


class AttnProcessor2_0(nn.Module):  # noqa: N801 - name kept for checkpoint compatibility
    """Scaled dot-product attention processor (PyTorch 2.0 fast path).

    Used for the self-attention (``attn1``) sites, which take no image condition.
    """

    def __init__(
        self,
        hidden_size: int | None = None,
        cross_attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        require_sdpa()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

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
        """Apply self- or cross-attention and return the projected hidden states."""
        residual = hidden_states
        hidden_states, attention_mask, spatial_shape, batch_size = prepare_hidden_states(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // attn.heads
        query = split_heads(query, batch_size, attn.heads, head_dim)
        key = split_heads(key, batch_size, attn.heads, head_dim)
        value = split_heads(value, batch_size, attn.heads, head_dim)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = merge_heads(hidden_states, batch_size, attn.heads, head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return finalize_attention(attn, hidden_states, residual, spatial_shape)


class IPAttnProcessor2_0(nn.Module):  # noqa: N801 - name kept for checkpoint compatibility
    """IP-Adapter attention processor with decoupled text and image cross-attention.

    ``encoder_hidden_states`` is expected to be the concatenation of the prompt tokens
    and exactly ``num_tokens`` image tokens along the sequence axis. The prompt part is
    routed through the frozen ``to_k``/``to_v`` of the host UNet while the image part
    goes through the learned ``to_k_ip``/``to_v_ip``.

    Args:
        hidden_size: Width of the attention layer.
        cross_attention_dim: Channel count of ``encoder_hidden_states``.
        scale: Weight applied to the image-prompt contribution.
        num_tokens: Number of trailing image tokens (16 for the ``ip_adapter_plus``
            variant, 4 for the released SwiftEdit checkpoint).
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int | None = None,
        scale: float = 1.0,
        num_tokens: int = IP_NUM_TOKENS,
    ) -> None:
        super().__init__()
        require_sdpa()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def split_conditions(
        self, encoder_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ``encoder_hidden_states`` into the prompt part and the image part.

        The split point is ``sequence_length - num_tokens``, so the prompt branch accepts
        an arbitrary sequence length. That is what lets a dense garment-token sequence be
        substituted for the 77 CLIP text tokens without touching this class.
        """
        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        return encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]

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
        """Attend over the prompt and image conditions, then sum the two branches."""
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

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = merge_heads(hidden_states, batch_size, attn.heads, head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_hidden_states is not None:
            ip_key = split_heads(self.to_k_ip(ip_hidden_states), batch_size, attn.heads, head_dim)
            ip_value = split_heads(self.to_v_ip(ip_hidden_states), batch_size, attn.heads, head_dim)
            ip_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            with torch.no_grad():
                self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)

            ip_states = merge_heads(ip_states, batch_size, attn.heads, head_dim)
            hidden_states = hidden_states + self.scale * ip_states.to(query.dtype)

        return finalize_attention(attn, hidden_states, residual, spatial_shape)
