"""Preference objectives over precomputed rewards/log-probabilities, not a full trainer."""
import torch
import torch.nn.functional as F


def bradley_terry_loss(chosen_reward, rejected_reward):
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """Inputs are sequence-level log probabilities, one scalar per preference pair."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits)
    chosen = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return loss.mean(), chosen, rejected


def group_relative_advantages(rewards, eps=1e-8):
    """Normalize rewards within each prompt group along the last dimension."""
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("each group needs at least two completions")
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)
