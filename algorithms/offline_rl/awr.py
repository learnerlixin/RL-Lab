"""Advantage-Weighted Regression (AWR) objective components."""
import torch
import torch.nn.functional as F


def awr_weights(advantages, temperature=1.0, max_weight=20.0):
    """Exponentiate and cap advantages to form stable imitation weights."""
    if temperature <= 0 or max_weight <= 0:
        raise ValueError("temperature and max_weight must be positive")
    return torch.exp(advantages.detach() / temperature).clamp(max=max_weight)


def awr_policy_loss(policy_log_probs, advantages, temperature=1.0,
                    max_weight=20.0, mask=None):
    """Weighted maximum-likelihood loss for dataset actions."""
    if policy_log_probs.shape != advantages.shape:
        raise ValueError("policy_log_probs and advantages must have identical shapes")
    weights = awr_weights(advantages, temperature, max_weight)
    losses = -weights * policy_log_probs
    if mask is None:
        return losses.mean()
    mask = mask.to(dtype=losses.dtype)
    if mask.shape != losses.shape or mask.sum() <= 0:
        raise ValueError("mask must match losses and contain at least one valid item")
    return (losses * mask).sum() / mask.sum()


def awr_value_loss(values, returns, mask=None):
    """Regress the value network onto empirical/bootstrapped dataset returns."""
    if values.shape != returns.shape:
        raise ValueError("values and returns must have identical shapes")
    losses = F.mse_loss(values, returns.detach(), reduction="none")
    if mask is None:
        return losses.mean()
    mask = mask.to(dtype=losses.dtype)
    if mask.shape != losses.shape or mask.sum() <= 0:
        raise ValueError("mask must match values and contain at least one valid item")
    return (losses * mask).sum() / mask.sum()


__all__ = ["awr_weights", "awr_policy_loss", "awr_value_loss"]
