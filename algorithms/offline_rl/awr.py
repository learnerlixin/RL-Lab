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
    if mask is not None and mask.numel() == advantages.numel() and mask.shape != advantages.shape:
        mask = mask.reshape_as(advantages)
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



def monte_carlo_returns(rewards, terminals, timeouts=None, gamma=0.99):
    """Compute episode-bounded discounted returns for sequential transitions.

    Both true termination and time-limit timeout reset the return accumulator.
    This is a pure Monte-Carlo target (no timeout bootstrap); use custom
    bootstrapped targets when timeout value estimates are available.
    """
    import numpy as np
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)
    timeouts = (np.zeros_like(terminals) if timeouts is None else
                np.asarray(timeouts, dtype=np.float32).reshape(-1))
    if len(rewards) == 0 or terminals.shape != rewards.shape or timeouts.shape != rewards.shape:
        raise ValueError("rewards, terminals and timeouts must have the same nonzero length")
    if not np.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    if not np.isin(terminals, [0.0, 1.0]).all() or not np.isin(timeouts, [0.0, 1.0]).all():
        raise ValueError("terminals and timeouts must be binary")
    returns = np.zeros_like(rewards)
    running = 0.0
    for index in reversed(range(len(rewards))):
        if terminals[index] or timeouts[index]:
            running = 0.0
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return torch.as_tensor(returns, dtype=torch.float32)


class AWRAgent:
    """Continuous-action AWR agent with Gaussian policy and state-value network.

    Caller supplies precomputed return-to-go targets in batch["returns"]. The
    agent handles weighted policy likelihood, value regression and action bounds.
    """
    def __init__(self, observation_dim, action_dim, action_low=-1.0,
                 action_high=1.0, hidden_dim=256, learning_rate=3e-4,
                 temperature=1.0, max_weight=20.0, device="cpu"):
        if temperature <= 0 or max_weight <= 0:
            raise ValueError("temperature and max_weight must be positive")
        from .iql import _IQLGaussianActor, _IQLValue
        self.device = torch.device(device)
        self.temperature, self.max_weight = float(temperature), float(max_weight)
        self.actor = _IQLGaussianActor(
            observation_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.value = _IQLValue(observation_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=learning_rate)

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        import numpy as np
        states = torch.as_tensor(np.asarray(observation), dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        return self.actor.act(states, deterministic).squeeze(0).cpu().numpy()

    def update(self, batch):
        states = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32,
                                  device=self.device).reshape(-1, 1)
        if states.ndim != 2 or actions.ndim != 2 or len(states) == 0:
            raise ValueError("AWR expects non-empty vector observations and continuous actions")
        if actions.shape[0] != len(states) or returns.shape[0] != len(states):
            raise ValueError("all AWR batch fields must have the same batch dimension")
        values = self.value(states)
        advantages = (returns - values).detach()
        value_loss = awr_value_loss(values, returns)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_optimizer.step()

        log_probs = self.actor.log_prob(states, actions)
        weights = awr_weights(advantages, self.temperature, self.max_weight)
        actor_loss = -(weights * log_probs).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        return {
            "actor_loss": float(actor_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "advantage_mean": float(advantages.mean()),
            "weight_mean": float(weights.mean()),
            "actor_grad_norm": float(actor_grad_norm.detach()),
            "value_grad_norm": float(value_grad_norm.detach()),
        }


__all__ = ["awr_weights", "awr_policy_loss", "awr_value_loss", "awr_train_step",
           "monte_carlo_returns", "AWRAgent"]
