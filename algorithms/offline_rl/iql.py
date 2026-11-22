"""Implicit Q-Learning objectives and a continuous-control IQL agent."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


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
    """One-step TD target; terminals should exclude time-limit truncations."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    return rewards + gamma * (1.0 - terminals) * next_values


def advantage_weighted_bc_loss(policy_log_probs, advantages, temperature=3.0,
                               max_weight=100.0):
    """AWR-style weighted behavior cloning used by IQL policy extraction."""
    if temperature <= 0 or max_weight <= 0:
        raise ValueError("temperature and max_weight must be positive")
    if policy_log_probs.shape != advantages.shape:
        raise ValueError("policy_log_probs and advantages must have identical shapes")
    weights = torch.exp(advantages.detach() / temperature).clamp(max=max_weight)
    return -(weights * policy_log_probs).mean()


def behavior_cloning_loss(predicted_actions, dataset_actions):
    """MSE behavior cloning for continuous actions."""
    return F.mse_loss(predicted_actions, dataset_actions)


class _IQLValue(nn.Module):
    def __init__(self, observation_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, states):
        return self.net(states)


class _IQLGaussianActor(nn.Module):
    def __init__(self, observation_dim, action_dim, action_low, action_high, hidden_dim):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if torch.any(high <= low):
            raise ValueError("action_high must be greater than action_low")
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def distribution(self, states):
        features = self.body(states)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20.0, 2.0)
        return Normal(mean, log_std.exp())

    def log_prob(self, states, actions):
        normalized = (actions - self.action_bias) / self.action_scale
        normalized = normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(normalized)
        distribution = self.distribution(states)
        log_prob = distribution.log_prob(raw) - torch.log(1.0 - normalized.square() + 1e-6)
        log_prob = log_prob - torch.log(self.action_scale.abs().clamp_min(1e-6))
        return log_prob.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act(self, states, deterministic=False):
        distribution = self.distribution(states)
        raw = distribution.mean if deterministic else distribution.sample()
        return torch.tanh(raw) * self.action_scale + self.action_bias


class IQLAgent:
    """Implicit Q-Learning for offline continuous control.

    The caller owns dataset loading and minibatch sampling. Each update expects
    arrays/tensors for (s, a, r, s', terminal). Time-limit timeouts must be
    represented as non-terminal transitions to preserve bootstrap values.
    """
    def __init__(self, observation_dim, action_dim, action_low=-1.0,
                 action_high=1.0, hidden_dim=256, learning_rate=3e-4,
                 gamma=0.99, tau=0.005, expectile=0.7, temperature=3.0,
                 max_weight=100.0, device="cpu"):
        if not 0.0 <= gamma <= 1.0 or not 0.0 < tau <= 1.0:
            raise ValueError("gamma must be in [0,1] and tau in (0,1]")
        if not 0.0 < expectile < 1.0 or temperature <= 0 or max_weight <= 0:
            raise ValueError("invalid expectile/temperature/max_weight")
        self.device = torch.device(device)
        self.gamma, self.tau = float(gamma), float(tau)
        self.expectile, self.temperature = float(expectile), float(temperature)
        self.max_weight = float(max_weight)
        from algorithms.policy_optimization.sac import QNetwork
        self.actor = _IQLGaussianActor(
            observation_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.q1 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_q1 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_q2 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.value = _IQLValue(observation_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=learning_rate)

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        return self.actor.act(state, deterministic).squeeze(0).cpu().numpy()

    def update(self, batch):
        """Update value, twin Q critics and policy from a transition minibatch."""
        states = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32,
                                  device=self.device).reshape(-1, 1)
        next_states = torch.as_tensor(batch["next_observations"], dtype=torch.float32,
                                      device=self.device)
        terminals = torch.as_tensor(batch["terminals"], dtype=torch.float32,
                                    device=self.device).reshape(-1, 1)
        if states.ndim != 2 or actions.ndim != 2 or next_states.shape != states.shape:
            raise ValueError("IQL expects vector observations and vector continuous actions")
        if len(states) == 0 or actions.shape[0] != len(states) or rewards.shape[0] != len(states):
            raise ValueError("all minibatch fields must have the same nonzero batch dimension")
        with torch.no_grad():
            target_q = torch.minimum(self.target_q1(states, actions),
                                     self.target_q2(states, actions))
        value_prediction = self.value(states)
        value_loss = expectile_regression_loss(value_prediction, target_q, self.expectile)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            td_target = iql_q_target(rewards, self.value(next_states), terminals, self.gamma)
        q1_prediction, q2_prediction = self.q1(states, actions), self.q2(states, actions)
        critic_loss = F.mse_loss(q1_prediction, td_target) + F.mse_loss(q2_prediction, td_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            advantage = target_q - self.value(states)
            weights = torch.exp(advantage / self.temperature).clamp(max=self.max_weight)
        log_probs = self.actor.log_prob(states, actions)
        actor_loss = -(weights * log_probs).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for target, source in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.mul_(1.0 - self.tau).add_(self.tau * param)
        return {
            "value_loss": float(value_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "advantage": float(advantage.mean()),
            "weight_mean": float(weights.mean()),
        }


__all__ = ["expectile_regression_loss", "iql_value_loss", "iql_q_target",
           "advantage_weighted_bc_loss", "behavior_cloning_loss", "IQLAgent"]
