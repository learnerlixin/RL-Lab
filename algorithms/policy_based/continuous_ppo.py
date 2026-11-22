"""PPO for bounded continuous actions using a tanh-squashed Gaussian policy."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class ContinuousActorCritic(nn.Module):
    """Gaussian actor + state-value critic for vector observations."""
    def __init__(self, observation_dim, action_dim, action_low=-1.0,
                 action_high=1.0, hidden_dim=256):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if torch.any(high <= low):
            raise ValueError("action_high must be greater than action_low")
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def distribution(self, states):
        features = self.actor_body(states)
        return Normal(self.mean(features),
                      self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX).exp())

    def value(self, states):
        return self.critic(states).squeeze(-1)

    def _log_prob(self, distribution, raw_actions):
        squashed = torch.tanh(raw_actions)
        log_prob = distribution.log_prob(raw_actions)
        log_prob = log_prob - torch.log(1.0 - squashed.square() + 1e-6)
        log_prob = log_prob - torch.log(self.action_scale.abs().clamp_min(1e-6))
        return log_prob.sum(dim=-1)

    def act(self, states, deterministic=False):
        distribution = self.distribution(states)
        raw = distribution.mean if deterministic else distribution.rsample()
        squashed = torch.tanh(raw)
        actions = squashed * self.action_scale + self.action_bias
        return actions, self._log_prob(distribution, raw), self.value(states)

    def evaluate_actions(self, states, actions):
        normalized = ((actions - self.action_bias) / self.action_scale).clamp(
            -1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(normalized)
        distribution = self.distribution(states)
        log_prob = self._log_prob(distribution, raw)
        # -log pi(a|s) is a one-sample entropy estimator.
        return log_prob, -log_prob, self.value(states)


def continuous_ppo_update(model, optimizer, states, actions, old_log_probs,
                          returns, advantages, clip_range=0.2,
                          value_coef=0.5, entropy_coef=0.0, epochs=4,
                          batch_size=64, max_grad_norm=0.5):
    """Optimize a collected continuous-action rollout with PPO clipping."""
    if clip_range <= 0 or epochs <= 0 or batch_size <= 0 or max_grad_norm <= 0:
        raise ValueError("clip_range, epochs, batch_size and max_grad_norm must be positive")
    if value_coef < 0 or entropy_coef < 0:
        raise ValueError("value_coef and entropy_coef must be nonnegative")
    size = states.shape[0]
    if size == 0:
        raise ValueError("rollout must contain at least one transition")
    if not (actions.shape[0] == old_log_probs.shape[0] == returns.shape[0]
            == advantages.shape[0] == size):
        raise ValueError("all rollout tensors must share the leading dimension")
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0
    for _ in range(epochs):
        indices = torch.randperm(size, device=states.device)
        for start in range(0, size, batch_size):
            idx = indices[start:start + batch_size]
            new_log_probs, entropy, values = model.evaluate_actions(states[idx], actions[idx])
            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            unclipped = ratio * advantages[idx]
            clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages[idx]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[idx])
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(entropy_mean.detach())
            updates += 1
    return {key: value / max(updates, 1) for key, value in metrics.items()}


__all__ = ["ContinuousActorCritic", "continuous_ppo_update"]
