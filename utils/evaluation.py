"""Gymnasium-compatible episodic evaluation for policies and actor-critic agents."""
import inspect

import numpy as np
import torch


def _to_action(value):
    """Extract the environment action from tensors and common agent return tuples."""
    if isinstance(value, (tuple, list)) and value:
        # Actor-critic APIs commonly return (action, log_prob, value).
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value.squeeze(axis=0)
    if value.ndim == 0:
        return value.item()
    return value


def _policy_action(policy, observation, deterministic):
    method = policy.act if hasattr(policy, "act") else policy
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs = {"deterministic": deterministic} if "deterministic" in parameters else {}

    if isinstance(policy, torch.nn.Module):
        argument = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        if argument.ndim == 1:
            argument = argument.unsqueeze(0)
    else:
        argument = observation

    with torch.inference_mode():
        result = method(argument, **kwargs)
    return _to_action(result)


def evaluate_policy(env, policy, episodes=5, deterministic=True, seed=None):
    """Evaluate a callable or agent without taking ownership of/closing env.

    Supports NumPy-based agents, torch.nn.Module policies, and actor-critic
    methods returning tuples such as (action, log_prob, value). The evaluator
    preserves a module's previous train/eval mode and never records gradients.

    Both Gymnasium's five-value step API and the legacy four-value API are
    accepted. Returns per-episode returns and lengths plus their means.
    """
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    returns, lengths = [], []
    was_training = policy.training if isinstance(policy, torch.nn.Module) else None
    if was_training is not None:
        policy.eval()
    try:
        for episode in range(episodes):
            reset_result = (env.reset(seed=seed + episode) if seed is not None
                            else env.reset())
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
    finally:
        if was_training is not None:
            policy.train(was_training)
    return {"returns": returns, "lengths": lengths,
            "mean_return": float(np.mean(returns)),
            "mean_length": float(np.mean(lengths))}


__all__ = ["evaluate_policy"]
