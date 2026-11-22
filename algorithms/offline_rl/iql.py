"""Implicit Q-Learning (IQL) and behavior-cloning objective components."""
import torch
import torch.nn.functional as F


def expectile_regression_loss(predicted, target, expectile=0.7):
    """Asymmetric squared error used to fit IQL's state-value function."""
    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be in (0, 1)")
    residual = target - predicted
    weight = torch.where(residual > 0, expectile, 1.0 - expectile)
    return (weight * residual.pow(2)).mean()


def iql_value_loss(value, q1, q2, expectile=0.7):
    """Fit V(s) toward the lower of the two data-action Q estimates."""
    return expectile_regression_loss(value, torch.minimum(q1, q2).detach(), expectile)


def iql_q_target(rewards, next_values, terminals, gamma=0.99):
    return rewards + gamma * (1.0 - terminals) * next_values


def advantage_weighted_bc_loss(policy_log_probs, advantages, temperature=3.0,
                               max_weight=100.0):
    """AWR-style weighted behavior cloning used by IQL's policy extraction."""
    if temperature <= 0 or max_weight <= 0:
        raise ValueError("temperature and max_weight must be positive")
    weights = torch.exp(advantages.detach() / temperature).clamp(max=max_weight)
    return -(weights * policy_log_probs).mean()


def behavior_cloning_loss(predicted_actions, dataset_actions):
    """MSE behavior cloning for continuous actions."""
    return F.mse_loss(predicted_actions, dataset_actions)
