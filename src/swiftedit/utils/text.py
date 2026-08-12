"""Prompt tokenisation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_captions(tokenizer: PreTrainedTokenizerBase, captions: Sequence[str]) -> torch.Tensor:
    """Tokenise ``captions`` to a padded batch of input ids.

    Args:
        tokenizer: Any CLIP-style tokenizer exposing ``model_max_length``.
        captions: Prompts to encode. An empty string yields the null embedding.

    Returns:
        Integer tensor of shape ``(len(captions), tokenizer.model_max_length)``.
    """
    inputs = tokenizer(
        list(captions),
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids
