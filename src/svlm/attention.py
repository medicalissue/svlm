from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class AttentionProjections:
    """
    Container for attention tensors collected from the model.

    Shapes:
        cross_modal: (num_heads, num_visual_tokens, num_text_tokens)
        self_attn: (num_heads, num_text_tokens, num_text_tokens)
    """

    cross_modal: torch.Tensor
    self_attn: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.cross_modal.dim() != 3:
            raise ValueError(
                "cross_modal attention must be a 3D tensor [H, N_v, N_t], "
                f"got shape {tuple(self.cross_modal.shape)}"
            )
        if self.self_attn is not None and self.self_attn.dim() != 3:
            raise ValueError(
                "self_attn attention must be a 3D tensor [H, N_t, N_t], "
                f"got shape {tuple(self.self_attn.shape)}"
            )


def _avg_over_heads(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 1:
        raise ValueError("tensor must have at least one dimension for head averaging")
    return tensor.mean(dim=0)


def compute_visual_grounding(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute g_i(t) from cross-modal attention.

    Args:
        attn: Tensor of shape [H, N_v, N_t].

    Returns:
        Tensor of shape [N_t] with visual grounding scores.
    """
    if attn.dim() != 3:
        raise ValueError("Expected attention tensor of shape [H, N_v, N_t]")
    head_avg = _avg_over_heads(attn)
    if head_avg.dim() != 2:
        raise ValueError("Head averaged tensor must have shape [N_v, N_t]")
    return head_avg.mean(dim=0)


def compute_language_evidence(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute l_i(t) from self-attention maps.

    Args:
        attn: Tensor of shape [H, N_t, N_t].

    Returns:
        Tensor of shape [N_t] with language evidence scores.
    """
    if attn.dim() != 3:
        raise ValueError("Expected self-attention tensor of shape [H, N_t, N_t]")
    head_avg = _avg_over_heads(attn)
    if head_avg.dim() != 2:
        raise ValueError("Head averaged tensor must have shape [N_t, N_t]")
    return head_avg.mean(dim=0)


def compute_visual_persistence(
    grounding: torch.Tensor,
    prev_persistence: Optional[torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """
    Exponentially weighted moving average of grounding signal.

    Args:
        grounding: Current g_i(t) scores, shape [N_t].
        prev_persistence: Previous g~_i(t-1) scores, shape [N_t] or None.
        beta: EMA coefficient in [0, 1).
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if prev_persistence is None:
        return grounding
    if grounding.shape != prev_persistence.shape:
        raise ValueError("grounding and prev_persistence must have identical shapes")
    return beta * prev_persistence + (1.0 - beta) * grounding


def compute_visual_ratio(
    grounding: torch.Tensor,
    language: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute r_i(t) = g_i(t) / (g_i(t) + l_i(t) + eps).
    """
    if grounding.shape != language.shape:
        raise ValueError("grounding and language tensors must have identical shapes")
    denom = grounding + language + eps
    return grounding / denom


def normalize_signal(signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center signal by subtracting mean, returning centered signal and mean.
    """
    mean = signal.mean()
    return signal - mean, mean
