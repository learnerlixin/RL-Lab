"""Minimal DQN components for discrete-action environments."""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated):
        self.data.append((np.asarray(state, dtype=np.float32),
                          int(action), float(reward),
                          np.asarray(next_state, dtype=np.float32),
                          float(terminated)))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)
        return (np.stack(states), np.asarray(actions), np.asarray(rewards, dtype=np.float32),
                np.stack(next_states), np.asarray(terminals, dtype=np.float32))

    def __len__(self):
        return len(self.data)


class QNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))

    def forward(self, states):
        return self.net(states)


class DQNAgent:
    """DQN with replay memory and a periodically synchronized target network."""
    def __init__(self, observation_dim, action_dim, learning_rate=1e-3,
                 gamma=0.99, target_update_interval=500, device="cpu"):
        self.device = torch.device(device)
        self.online = QNetwork(observation_dim, action_dim).to(self.device)
        self.target = QNetwork(observation_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.action_dim = action_dim
        self.steps = 0

    @torch.no_grad()
    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    def update(self, replay, batch_size=64):
        if len(replay) < batch_size:
            return None
        states, actions, rewards, next_states, terminals = replay.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(terminals, dtype=torch.float32, device=self.device)

        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = self.target(next_states).max(dim=1).values
            target = rewards + self.gamma * (1.0 - terminals) * next_values
        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
