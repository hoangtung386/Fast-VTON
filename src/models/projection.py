"""Projection from a pooled CLIP image embedding to cross-attention tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import CROSS_ATTENTION_DIM, IP_NUM_TOKENS


class ImageProjModel(nn.Module):
    """Map one pooled image embedding to ``clip_extra_context_tokens`` prompt tokens.

    The released checkpoint stores ``proj.weight`` with shape ``(4096, 1024)``, i.e.
    4 tokens of width 1024.

    Args:
        cross_attention_dim: Width of each emitted token.
        clip_embeddings_dim: Width of the pooled CLIP image embedding.
        clip_extra_context_tokens: Number of tokens to emit.
    """

    def __init__(
        self,
        cross_attention_dim: int = CROSS_ATTENTION_DIM,
        clip_embeddings_dim: int = CROSS_ATTENTION_DIM,
        clip_extra_context_tokens: int = IP_NUM_TOKENS,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Project ``(batch, clip_dim)`` to ``(batch, num_tokens, cross_attention_dim)``."""
        tokens = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        return self.norm(tokens)
