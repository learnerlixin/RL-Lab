"""Deep Deterministic Policy Gradient (DDPG) for continuous control."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from algorithms.policy_optimization.sac import ContinuousReplayBuffer, QNetwork
from algorithms.policy_optimization.td3 import DeterministicActor


class DDPGAgent:
    """Deterministic actor-critic with target networks and replay."""

    def __init__(self, observation_dim, action_dim, action_low=-1.0,
                 action_high=1.0, learning_rate=1e-3, gamma=0.99,
                 tau=0.005, device="cpu"):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        self.device = torch.device(device)
        self.actor = DeterministicActor(
            observation_dim, action_dim, action_low, action_high
        ).to(self.device)
        self.target_actor = DeterministicActor(
            observation_dim, action_dim, action_low, action_high
        ).to(self.device)
        self.critic = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_critic = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma, self.tau = gamma, tau
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, observation, noise_std=0.0):
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        action = self.actor(state).squeeze(0)
        if noise_std > 0:
            action = action + noise_std * torch.randn_like(action)
        return action.clamp(self.action_low, self.action_high).cpu().numpy()

    @torch.no_grad()
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - self.tau).add_(self.tau * param)

    def update(self, replay, batch_size=256):
        if len(replay) < batch_size:
            return None
        states, actions, rewards, next_states, terminals = replay.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(terminals, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            targets = rewards + self.gamma * (1.0 - terminals) * target_q

        predicted_q = self.critic(states, actions)
        critic_loss = F.mse_loss(predicted_q, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        try:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
        }


__all__ = ["DDPGAgent"]
