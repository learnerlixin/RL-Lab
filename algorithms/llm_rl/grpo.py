"""GRPO-style clipped policy objective over token log-probabilities.

The caller is responsible for generation, reward scoring, token masks and rollout storage.
"""
import torch


def grpo_clipped_loss(new_logps, old_logps, advantages, attention_mask,
                      clip_range=0.2, reference_logps=None, kl_coef=0.0):
    """Compute masked token-level clipped surrogate with optional reference KL penalty.

    new_logps/old_logps/attention_mask: [batch, sequence]
    advantages: [batch] or [batch, 1]
    reference_logps: optional [batch, sequence]
    """
    if clip_range <= 0 or kl_coef < 0:
        raise ValueError("clip_range must be positive and kl_coef nonnegative")
    mask = attention_mask.to(dtype=new_logps.dtype)
    if mask.shape != new_logps.shape or mask.sum() <= 0:
        raise ValueError("attention_mask must match log-probs and include valid tokens")
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    ratio = torch.exp(new_logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    per_token = -torch.minimum(unclipped, clipped)
    if reference_logps is not None and kl_coef:
        log_ratio_ref = reference_logps - new_logps
        per_token = per_token + kl_coef * (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0)
    return (per_token * mask).sum() / mask.sum()


def group_relative_advantages(rewards, eps=1e-8):
    """Normalize rewards over completions for each prompt; rewards shape [B, G]."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [batch, group_size>=2]")
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)
