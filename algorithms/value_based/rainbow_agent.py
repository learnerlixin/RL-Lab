"""Trainable Rainbow-style categorical DQN with prioritized replay."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .advanced_dqn import PrioritizedReplayBuffer
from .rainbow import NoisyLinear, categorical_projection


class RainbowNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, atom_count=51,
                 value_min=-10.0, value_max=10.0, hidden_dim=128,
                 noisy=True, dueling=True):
        super().__init__()
        if atom_count < 2 or value_min >= value_max:
            raise ValueError("invalid categorical support")
        self.action_dim, self.atom_count = action_dim, atom_count
        self.value_min, self.value_max = float(value_min), float(value_max)
        self.noisy, self.dueling = noisy, dueling
        self.register_buffer("support", torch.linspace(value_min, value_max, atom_count))
        self.feature = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        linear = NoisyLinear if noisy else nn.Linear
        if dueling:
            self.value_head = nn.Sequential(linear(hidden_dim, hidden_dim), nn.ReLU(),
                                            linear(hidden_dim, atom_count))
            self.advantage_head = nn.Sequential(linear(hidden_dim, hidden_dim), nn.ReLU(),
                                                linear(hidden_dim, action_dim * atom_count))
        else:
            self.head = nn.Sequential(linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      linear(hidden_dim, action_dim * atom_count))

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, states):
        features = self.feature(states)
        if self.dueling:
            value = self.value_head(features).view(-1, 1, self.atom_count)
            advantage = self.advantage_head(features).view(-1, self.action_dim, self.atom_count)
            logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            logits = self.head(features).view(-1, self.action_dim, self.atom_count)
        return torch.softmax(logits, dim=-1)

    def q_values(self, states):
        return (self(states) * self.support).sum(dim=-1)


class RainbowAgent:
    """C51 + Double-DQN + Dueling + NoisyNet + importance-weighted PER."""
    def __init__(self, observation_dim, action_dim, learning_rate=1e-4,
                 gamma=0.99, target_update_interval=1000, atom_count=51,
                 value_min=-10.0, value_max=10.0, hidden_dim=128,
                 noisy=True, dueling=True, device="cpu"):
        self.device = torch.device(device)
        kwargs = dict(observation_dim=observation_dim, action_dim=action_dim,
                      atom_count=atom_count, value_min=value_min, value_max=value_max,
                      hidden_dim=hidden_dim, noisy=noisy, dueling=dueling)
        self.online, self.target = RainbowNetwork(**kwargs).to(self.device), RainbowNetwork(**kwargs).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma, self.target_update_interval = float(gamma), int(target_update_interval)
        self.steps, self.action_dim = 0, action_dim

    @torch.no_grad()
    def act(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_dim))
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online.q_values(state).argmax(dim=1).item())

    def update(self, replay, batch_size=64, beta=0.4):
        if len(replay) < batch_size:
            return None
        states, actions, rewards, next_states, terminals, discounts, indices, weights = replay.sample(batch_size, beta=beta, include_discounts=True)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(terminals, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(discounts, dtype=torch.float32, device=self.device)
        discounts = torch.where(torch.isnan(discounts), torch.full_like(discounts, self.gamma), discounts)

        self.online.reset_noise()
        self.target.reset_noise()
        probs = self.online(states)
        chosen = probs[torch.arange(batch_size, device=self.device), actions]
        with torch.no_grad():
            next_actions = self.online.q_values(next_states).argmax(dim=1)
            next_probs = self.target(next_states)[torch.arange(batch_size, device=self.device), next_actions]
            projected = categorical_projection(next_probs, rewards, terminals, discounts,
                                               self.online.support, self.online.value_min, self.online.value_max)
        per_sample_loss = -(projected * chosen.clamp_min(1e-6).log()).sum(dim=1)
        loss = (weights * per_sample_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        priorities = per_sample_loss.detach().cpu().numpy()
        replay.update_priorities(indices, priorities)
        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.item()), "mean_priority": float(np.mean(priorities))}
