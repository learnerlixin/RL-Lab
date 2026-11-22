"""Supervised behavior-cloning objectives and an optimizer step."""
import torch
import torch.nn.functional as F


def discrete_behavior_cloning_loss(logits, actions, mask=None):
    """Cross-entropy imitation loss for discrete actions."""
    if logits.ndim < 2 or logits.shape[:-1] != actions.shape:
        raise ValueError("actions must match all leading logits dimensions")
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             actions.long().reshape(-1), reduction="none")
    losses = losses.reshape(actions.shape)
    if mask is None:
        return losses.mean()
    mask = mask.to(device=losses.device, dtype=losses.dtype)
    if mask.shape != losses.shape or mask.sum() <= 0:
        raise ValueError("mask must match actions and contain a valid item")
    return (losses * mask).sum() / mask.sum()


def continuous_behavior_cloning_loss(predicted_actions, dataset_actions, mask=None):
    """Mean-squared imitation loss for deterministic continuous policies."""
    if predicted_actions.shape != dataset_actions.shape:
        raise ValueError("predicted_actions and dataset_actions must have identical shapes")
    losses = F.mse_loss(predicted_actions, dataset_actions, reduction="none")
    if losses.ndim > 1:
        losses = losses.mean(dim=-1)
    if mask is None:
        return losses.mean()
    mask = mask.to(device=losses.device, dtype=losses.dtype)
    if mask.shape != losses.shape or mask.sum() <= 0:
        raise ValueError("mask must match per-sample losses and contain a valid item")
    return (losses * mask).sum() / mask.sum()


def behavior_cloning_train_step(policy, optimizer, observations, actions,
                                action_type="discrete", mask=None,
                                max_grad_norm=1.0):
    """Perform one supervised policy update on an offline action dataset."""
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    predictions = policy(observations)
    if action_type == "discrete":
        loss = discrete_behavior_cloning_loss(predictions, actions, mask)
        with torch.no_grad():
            correct = (predictions.argmax(dim=-1) == actions.long()).float()
            if mask is None:
                accuracy = correct.mean()
            else:
                weights = mask.to(device=correct.device, dtype=correct.dtype)
                accuracy = (correct * weights).sum() / weights.sum().clamp_min(1.0)
        metrics = {"accuracy": float(accuracy.detach())}
    elif action_type == "continuous":
        loss = continuous_behavior_cloning_loss(predictions, actions, mask)
        metrics = {"mse": float(loss.detach())}
    else:
        raise ValueError("action_type must be 'discrete' or 'continuous'")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm.detach()), **metrics}


__all__ = ["discrete_behavior_cloning_loss", "continuous_behavior_cloning_loss",
           "behavior_cloning_train_step"]
