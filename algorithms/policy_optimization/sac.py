"""Soft Actor-Critic for continuous vector-observation/action tasks."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def _validated_action_bounds(action_dim, action_low, action_high):
    """Return finite per-dimension action bounds, broadcasting scalar inputs."""
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    low = torch.as_tensor(action_low, dtype=torch.float32)
    high = torch.as_tensor(action_high, dtype=torch.float32)
    try:
        low = torch.broadcast_to(low, (action_dim,)).clone()
        high = torch.broadcast_to(high, (action_dim,)).clone()
    except RuntimeError as error:
        raise ValueError("action bounds must be scalars or match action_dim") from error
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError("action bounds must be finite")
    if torch.any(high <= low):
        raise ValueError("action_high must be greater than action_low")
    return low, high


class GaussianActor(nn.Module):
    def __init__(self, observation_dim, action_dim, action_low=-1.0, action_high=1.0, hidden_dim=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        low, high = _validated_action_bounds(action_dim, action_low, action_high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, states):
        features = self.body(states)
        return self.mean(features), self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, states):
        mean, log_std = self(states)
        distribution = Normal(mean, log_std.exp())
        raw = distribution.rsample()
        squashed = torch.tanh(raw)
        actions = squashed * self.action_scale + self.action_bias
        log_prob = distribution.log_prob(raw)
        log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(self.action_scale.abs().clamp_min(1e-6)).sum()
        deterministic = torch.tanh(mean) * self.action_scale + self.action_bias
        return actions, log_prob, deterministic


class QNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim + action_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))


class ContinuousReplayBuffer:
    """Replay buffer preserving vector-valued continuous actions."""
    def __init__(self, capacity=1_000_000):
        from collections import deque
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        reward = float(reward)
        terminal = float(terminated)
        if state.ndim != 1 or next_state.shape != state.shape or action.ndim != 1:
            raise ValueError("replay transitions require vector state/action and matching next_state")
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise ValueError("replay observations must be finite")
        if not np.isfinite(action).all() or not np.isfinite(reward):
            raise ValueError("replay actions and rewards must be finite")
        if terminal not in (0.0, 1.0):
            raise ValueError("terminated must be a boolean/binary value")
        if self.data:
            old_state, old_action = self.data[0][0], self.data[0][1]
            if state.shape != old_state.shape or action.shape != old_action.shape:
                raise ValueError("replay observation/action shapes must remain consistent")
        self.data.append((state.copy(), action.copy(), reward,
                          next_state.copy(), terminal))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)
        return (np.stack(states), np.stack(actions), np.asarray(rewards, dtype=np.float32),
                np.stack(next_states), np.asarray(terminals, dtype=np.float32))

    def __len__(self):
        return len(self.data)


class SACAgent:
    """Twin-critic SAC with entropy-temperature tuning."""
    def __init__(self, observation_dim, action_dim, action_low=-1.0, action_high=1.0,
                 learning_rate=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 automatic_entropy=True, device="cpu"):
        if learning_rate <= 0 or not 0.0 <= gamma <= 1.0 or not 0.0 < tau <= 1.0:
            raise ValueError("learning_rate must be positive, gamma in [0,1], tau in (0,1]")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.device = torch.device(device)
        self.actor = GaussianActor(observation_dim, action_dim, action_low, action_high).to(self.device)
        self.q1 = QNetwork(observation_dim, action_dim).to(self.device)
        self.q2 = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_q1 = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_q2 = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)
        self.gamma, self.tau = gamma, tau
        self.automatic_entropy = automatic_entropy
        self.target_entropy = -float(action_dim)
        if automatic_entropy:
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32,
                                          device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = float(alpha)

    @property
    def current_alpha(self):
        return self.log_alpha.exp().detach() if self.automatic_entropy else self.alpha

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        if deterministic:
            return self.actor.sample(state)[2].squeeze(0).cpu().numpy()
        return self.actor.sample(state)[0].squeeze(0).cpu().numpy()

    def update(self, replay, batch_size=256):
        if len(replay) < batch_size:
            return None
        states, actions, rewards, next_states, terminals = replay.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(terminals, dtype=torch.float32, device=self.device).unsqueeze(-1)
        alpha = self.current_alpha

        with torch.no_grad():
            next_actions, next_logp, _ = self.actor.sample(next_states)
            target_q = torch.minimum(self.target_q1(next_states, next_actions),
                                     self.target_q2(next_states, next_actions))
            target = rewards + self.gamma * (1.0 - terminals) * (target_q - alpha * next_logp)
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            sampled_actions, logp, _ = self.actor.sample(states)
            actor_q = torch.minimum(self.q1(states, sampled_actions), self.q2(states, sampled_actions))
            actor_loss = (alpha * logp - actor_q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)

        alpha_loss_value = 0.0
        if self.automatic_entropy:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach())

        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * param)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * param)
        return {"critic_loss": float(critic_loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "alpha": float(self.current_alpha), "alpha_loss": alpha_loss_value}
