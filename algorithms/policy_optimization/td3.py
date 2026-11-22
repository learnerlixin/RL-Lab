"""Twin Delayed Deep Deterministic Policy Gradient for continuous control."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from algorithms.policy_optimization.sac import (
    ContinuousReplayBuffer, QNetwork, _validated_action_bounds,
)


class DeterministicActor(nn.Module):
    def __init__(self, observation_dim, action_dim, action_low=-1.0, action_high=1.0, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh())
        low, high = _validated_action_bounds(action_dim, action_low, action_high)
        self.register_buffer("scale", (high - low) / 2.0)
        self.register_buffer("bias", (high + low) / 2.0)

    def forward(self, states):
        return self.net(states) * self.scale + self.bias


class TD3Agent:
    def __init__(self, observation_dim, action_dim, action_low=-1.0, action_high=1.0,
                 learning_rate=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, device="cpu"):
        if learning_rate <= 0 or not 0.0 <= gamma <= 1.0 or not 0.0 < tau <= 1.0:
            raise ValueError("learning_rate must be positive, gamma in [0,1], tau in (0,1]")
        if policy_noise < 0 or noise_clip < 0 or policy_delay <= 0:
            raise ValueError("policy_noise/noise_clip must be nonnegative and policy_delay positive")
        self.device = torch.device(device)
        self.actor = DeterministicActor(observation_dim, action_dim, action_low, action_high).to(self.device)
        self.target_actor = DeterministicActor(observation_dim, action_dim, action_low, action_high).to(self.device)
        self.q1 = QNetwork(observation_dim, action_dim).to(self.device)
        self.q2 = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_q1 = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_q2 = QNetwork(observation_dim, action_dim).to(self.device)
        for target, source in ((self.target_actor, self.actor), (self.target_q1, self.q1), (self.target_q2, self.q2)):
            target.load_state_dict(source.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                                  lr=learning_rate)
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip, self.policy_delay = policy_noise, noise_clip, policy_delay
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.updates = 0

    @torch.no_grad()
    def act(self, observation, noise_std=0.0):
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        action = self.actor(state).squeeze(0)
        if noise_std:
            action = action + noise_std * torch.randn_like(action)
        return action.clamp(self.action_low, self.action_high).cpu().numpy()

    def _soft_update(self, target, source):
        with torch.no_grad():
            for tp, p in zip(target.parameters(), source.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

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
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(self.action_low, self.action_high)
            target_q = torch.minimum(self.target_q1(next_states, next_actions),
                                     self.target_q2(next_states, next_actions))
            target = rewards + self.gamma * (1.0 - terminals) * target_q
        q1, q2 = self.q1(states, actions), self.q2(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        actor_loss_value = 0.0
        if self.updates % self.policy_delay == 0:
            critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            try:
                actor_loss = -self.q1(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.detach())
            finally:
                for parameter in critic_parameters:
                    parameter.requires_grad_(True)
            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_q1, self.q1)
            self._soft_update(self.target_q2, self.q2)
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": actor_loss_value}
