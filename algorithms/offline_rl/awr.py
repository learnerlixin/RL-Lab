"""Advantage-Weighted Regression (AWR) objective and policy/value update."""
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
    mask = mask.to(device=losses.device, dtype=losses.dtype)
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
    mask = mask.to(device=losses.device, dtype=losses.dtype)
    if mask.shape != losses.shape or mask.sum() <= 0:
        raise ValueError("mask must match values and contain at least one valid item")
    return (losses * mask).sum() / mask.sum()


def awr_train_step(policy, value_network, actor_optimizer, value_optimizer,
                   observations, actions, returns, temperature=1.0,
                   max_weight=20.0, mask=None, max_grad_norm=1.0):
    """Update a stochastic actor and value network from offline returns.

    The policy must expose log_prob(observations, actions) returning one
    log-probability per sample. Return/advantage estimation is caller-owned.
    """
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    values = value_network(observations)
    if values.shape != returns.shape:
        if values.numel() != returns.numel():
            raise ValueError("value predictions and returns must have matching elements")
        values = values.reshape_as(returns)
    advantages = (returns - values).detach()
    value_loss = awr_value_loss(values, returns, mask)
    value_optimizer.zero_grad(set_to_none=True)
    value_loss.backward()
    value_grad_norm = torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_grad_norm)
    value_optimizer.step()

    log_probs = policy.log_prob(observations, actions)
    if log_probs.shape != advantages.shape:
        if log_probs.numel() != advantages.numel():
            raise ValueError("policy log-probabilities and returns must have matching elements")
        log_probs = log_probs.reshape_as(advantages)
    actor_loss = awr_policy_loss(log_probs, advantages, temperature, max_weight, mask)
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    actor_optimizer.step()
    return {"actor_loss": float(actor_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "advantage_mean": float(advantages.mean()),
            "weight_mean": float(awr_weights(advantages, temperature, max_weight).mean()),
            "actor_grad_norm": float(actor_grad_norm.detach()),
            "value_grad_norm": float(value_grad_norm.detach())}


__all__ = ["awr_weights", "awr_policy_loss", "awr_value_loss", "awr_train_step"]
