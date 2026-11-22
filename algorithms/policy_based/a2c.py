"""Synchronous Advantage Actor-Critic (A2C) baseline."""
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .actor_critic import ActorCritic


def discounted_returns(rewards, dones, bootstrap_value, gamma=0.99):
    """Compute time-major returns; dones marks true MDP terminals."""
    if rewards.ndim != 1 or dones.shape != rewards.shape:
        raise ValueError("rewards and dones must be matching [T] tensors")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    returns = torch.empty_like(rewards)
    running = torch.as_tensor(bootstrap_value, dtype=rewards.dtype,
                              device=rewards.device).reshape(())
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * (1.0 - dones[t]) * running
        returns[t] = running
    return returns


def a2c_loss(logits, values, actions, returns, value_coef=0.5,
             entropy_coef=0.01):
    """Combined policy-gradient, value regression, and entropy objective."""
    if value_coef < 0 or entropy_coef < 0:
        raise ValueError("loss coefficients must be nonnegative")
    distribution = Categorical(logits=logits)
    advantages = returns - values
    policy_loss = -(distribution.log_prob(actions) * advantages.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = distribution.entropy().mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss,
                   "entropy": entropy}


class A2CAgent:
    """On-policy synchronous A2C optimizer for a collected rollout batch."""
    def __init__(self, observation_dim, action_dim, learning_rate=3e-4,
                 hidden_dim=128, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, device="cpu"):
        self.device = torch.device(device)
        self.network = ActorCritic(observation_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.value_coef, self.entropy_coef = value_coef, entropy_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def act(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.network(state)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return int(action.item()), float(value.item())

    def update(self, states, actions, returns):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        logits, values = self.network(states)
        loss, parts = a2c_loss(logits, values, actions, returns,
                               self.value_coef, self.entropy_coef)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                   self.max_grad_norm)
        self.optimizer.step()
        return {"loss": float(loss.detach().item()),
                "policy_loss": float(parts["policy_loss"].detach().item()),
                "value_loss": float(parts["value_loss"].detach().item()),
                "entropy": float(parts["entropy"].detach().item()),
                "grad_norm": float(grad_norm.detach().item())}
