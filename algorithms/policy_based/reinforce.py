"""Compact REINFORCE for discrete-action Gymnasium environments."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, action_dim))

    def forward(self, observation):
        return self.net(observation)


def reinforce(env, episodes=1000, learning_rate=1e-3, gamma=0.99, hidden_dim=128, seed=0, device="cpu"):
    """Train episodic policy gradient. Observations must be flat vectors."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    policy = PolicyNetwork(obs_dim, env.action_space.n, hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        log_probs, rewards = [], []
        while True:
            state = torch.as_tensor(np.asarray(obs).reshape(-1), dtype=torch.float32, device=device)
            dist = Categorical(logits=policy(state))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(float(reward))
            if terminated or truncated:
                break
        returns, running = [], 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.append(running)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=device)
        if returns.numel() > 1 and returns.std(unbiased=False) > 1e-8:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        loss = -(torch.stack(log_probs) * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(sum(rewards))
    return policy, history
