"""A small scalar reward head for externally encoded sequence representations."""
import torch
from torch import nn


class RewardModelHead(nn.Module):
    """Map pooled sequence embeddings to one scalar reward per sequence.

    A tokenizer/transformer encoder is injected by the caller; this module
    defines masked mean pooling and the scalar reward projection.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        if hidden_size <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("hidden_size must be positive and dropout in [0,1)")
        self.scorer = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, hidden_states, attention_mask):
        """hidden_states [B,T,H], attention_mask [B,T] -> rewards [B]."""
        if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("hidden_states and attention_mask must align on [B,T]")
        mask = attention_mask.to(device=hidden_states.device,
                                 dtype=hidden_states.dtype)
        if not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("attention_mask must be finite and nonnegative")
        denominator = mask.sum(dim=1)
        if (denominator <= 0).any():
            raise ValueError("each sequence must contain at least one valid token")
        pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / denominator.unsqueeze(-1)
        return self.scorer(pooled).squeeze(-1)


__all__ = ["RewardModelHead"]
