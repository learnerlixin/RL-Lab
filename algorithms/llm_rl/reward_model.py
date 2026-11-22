"""A small scalar reward head for externally encoded sequence representations."""
import torch
from torch import nn


class RewardModelHead(nn.Module):
    """Map pooled sequence embeddings to one scalar reward per sequence.

    A tokenizer/transformer encoder is intentionally injected by the caller;
    this module only defines masked pooling and the reward projection.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, hidden_states, attention_mask):
        """hidden_states [B,T,H], attention_mask [B,T] -> rewards [B]."""
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden_states * mask).sum(dim=1) / denominator
        return self.scorer(pooled).squeeze(-1)
