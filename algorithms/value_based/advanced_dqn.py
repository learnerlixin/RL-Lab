"""Double DQN, dueling heads, and proportional prioritized replay primitives."""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Separate state-value and action-advantage streams."""
    def __init__(self, observation_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        features = self.feature(states)
        values = self.value(features)
        advantages = self.advantage(features)
        return values + advantages - advantages.mean(dim=-1, keepdim=True)


def double_dqn_targets(rewards, terminals, next_online_q, next_target_q, gamma):
    """Select next actions online, but evaluate them with the target network."""
    next_actions = next_online_q.argmax(dim=1, keepdim=True)
    next_values = next_target_q.gather(1, next_actions).squeeze(1)
    return rewards + gamma * (1.0 - terminals) * next_values


class PrioritizedReplayBuffer:
    """Simple O(N) proportional PER reference implementation.

    For large-scale use, replace the linear probability calculation with a sum tree.
    """
    def __init__(self, capacity=100_000, alpha=0.6, eps=1e-6):
        if capacity <= 0 or alpha < 0 or eps <= 0:
            raise ValueError("invalid replay buffer configuration")
        self.capacity, self.alpha, self.eps = capacity, alpha, eps
        self.data = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0

    def add(self, state, action, reward, next_state, terminated):
        max_priority = self.priorities[:len(self.data)].max() if self.data else 1.0
        transition = (np.asarray(state, dtype=np.float32), int(action), float(reward),
                      np.asarray(next_state, dtype=np.float32), float(terminated))
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if not 0 <= beta <= 1 or not 0 < batch_size <= len(self.data):
            raise ValueError("invalid batch_size or beta")
        scaled = self.priorities[:len(self.data)] ** self.alpha
        probabilities = scaled / scaled.sum()
        indices = np.random.choice(len(self.data), batch_size, p=probabilities)
        weights = (len(self.data) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        batch = [self.data[i] for i in indices]
        states, actions, rewards, next_states, terminals = zip(*batch)
        return (np.stack(states), np.asarray(actions), np.asarray(rewards, dtype=np.float32),
                np.stack(next_states), np.asarray(terminals, dtype=np.float32),
                indices, weights.astype(np.float32))

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[int(index)] = abs(float(priority)) + self.eps

    def __len__(self):
        return len(self.data)


class DoubleDQNAgent:
    """DQN agent using online argmax and target-network evaluation."""
    def __init__(self, observation_dim, action_dim, learning_rate=1e-3,
                 gamma=0.99, target_update_interval=500, device="cpu"):
        self.device = torch.device(device)
        self.online = nn.Sequential(nn.Linear(observation_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(128, action_dim)).to(self.device)
        self.target = nn.Sequential(nn.Linear(observation_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(128, action_dim)).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma, self.target_update_interval = gamma, target_update_interval
        self.steps = 0

    def update(self, replay, batch_size=64):
        if len(replay) < batch_size:
            return None
        states, actions, rewards, next_states, terminals = replay.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(terminals, dtype=torch.float32, device=self.device)
        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = double_dqn_targets(rewards, terminals, self.online(next_states),
                                         self.target(next_states), self.gamma)
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
