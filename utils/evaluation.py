"""Gymnasium-compatible episodic policy evaluation."""
import inspect

import numpy as np
import torch


def _to_action(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return value


def _policy_action(policy, observation, deterministic):
    method = policy.act if hasattr(policy, "act") else policy
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "deterministic" in parameters:
        return _to_action(method(observation, deterministic=deterministic))
    return _to_action(method(observation))


def evaluate_policy(env, policy, episodes=5, deterministic=True, seed=None):
    """Evaluate a callable or agent without taking ownership of/closing env.

    Returns per-episode returns and lengths plus their means. Both Gymnasium's
    five-value step API and the legacy four-value API are accepted.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    returns, lengths = [], []
    for episode in range(episodes):
        reset_result = env.reset(seed=seed + episode) if seed is not None else env.reset()
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        total_reward, length, done = 0.0, 0, False
        while not done:
            action = _policy_action(policy, observation, deterministic)
            result = env.step(action)
            if len(result) == 5:
                observation, reward, terminated, truncated, _ = result
                done = bool(terminated or truncated)
            else:
                observation, reward, done, _ = result
                done = bool(done)
            total_reward += float(reward)
            length += 1
        returns.append(total_reward)
        lengths.append(length)
    return {"returns": returns, "lengths": lengths,
            "mean_return": float(np.mean(returns)),
            "mean_length": float(np.mean(lengths))}
