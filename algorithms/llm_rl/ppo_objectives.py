"""Clipped policy-gradient objectives over already-computed token log-probabilities."""
import torch


def clipped_policy_loss(new_logps, old_logps, advantages, clip_range=0.2):
    """PPO clipped surrogate; tensors must have identical shape and valid-token masking."""
    if clip_range <= 0:
        raise ValueError("clip_range must be positive")
    ratio = torch.exp(new_logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def approximate_kl(new_logps, old_logps):
    """Simple nonnegative-ish sampled KL diagnostic; inputs are aligned token log-probs."""
    log_ratio = new_logps - old_logps
    return ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
