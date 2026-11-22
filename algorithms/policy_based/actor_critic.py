"""Actor-Critic network and generalized advantage estimation utilities."""
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared MLP trunk with categorical policy and scalar value heads."""
    def __init__(self, observation_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states):
        features = self.trunk(states)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def act(self, states):
        logits, values = self(states)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values


def generalized_advantage_estimate(rewards, values, next_values, dones,
                                   gamma=0.99, lam=0.95):
    """Compute GAE(lambda) over a rollout; tensors are time-major and shape [T]."""
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0, 1]")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns
