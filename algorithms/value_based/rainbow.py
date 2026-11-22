"""Rainbow DQN building blocks: noisy layers, C51 and n-step returns."""
import torch
from torch import nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer."""
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma0 / in_features ** 0.5))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma0 / out_features ** 0.5))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = self.in_features ** -0.5
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @staticmethod
    def _scale_noise(size, device):
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        device = self.weight_mu.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon)


class CategoricalQNetwork(nn.Module):
    """C51 distributional Q network; outputs action x atom probabilities."""
    def __init__(self, observation_dim, action_dim, atom_count=51,
                 value_min=-10.0, value_max=10.0, hidden_dim=128):
        super().__init__()
        if atom_count < 2 or value_min >= value_max:
            raise ValueError("invalid categorical support")
        self.action_dim, self.atom_count = action_dim, atom_count
        self.value_min, self.value_max = value_min, value_max
        self.register_buffer("support", torch.linspace(value_min, value_max, atom_count))
        self.feature = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.head = nn.Linear(hidden_dim, action_dim * atom_count)

    def forward(self, states):
        logits = self.head(self.feature(states)).view(-1, self.action_dim, self.atom_count)
        return torch.softmax(logits, dim=-1)

    def q_values(self, states):
        return (self(states) * self.support).sum(dim=-1)


def categorical_projection(next_probabilities, rewards, terminals, gamma,
                           support, value_min, value_max):
    """Project Bellman-shifted distributions; gamma may be scalar or [batch]."""
    atom_count = support.numel()
    delta_z = (value_max - value_min) / (atom_count - 1)
    batch_size = rewards.shape[0]
    if torch.is_tensor(gamma):
        discounts = gamma.to(device=rewards.device, dtype=rewards.dtype).reshape(-1, 1)
        if discounts.shape[0] != batch_size:
            raise ValueError("batched gamma must have one value per transition")
    else:
        discounts = float(gamma)
    target_atoms = rewards.unsqueeze(1) + discounts * (1.0 - terminals.unsqueeze(1)) * support.unsqueeze(0)
    target_atoms = target_atoms.clamp(value_min, value_max)
    positions = (target_atoms - value_min) / delta_z
    lower, upper = positions.floor().long(), positions.ceil().long()
    projected = torch.zeros_like(next_probabilities)
    offset = (torch.arange(batch_size, device=rewards.device) * atom_count).unsqueeze(1)
    flat = projected.view(-1)
    lower_weight = upper.to(positions.dtype) - positions
    upper_weight = positions - lower.to(positions.dtype)
    same = lower == upper
    lower_weight = torch.where(same, torch.ones_like(lower_weight), lower_weight)
    upper_weight = torch.where(same, torch.zeros_like(upper_weight), upper_weight)
    flat.index_add_(0, (lower + offset).reshape(-1), (next_probabilities * lower_weight).reshape(-1))
    flat.index_add_(0, (upper + offset).reshape(-1), (next_probabilities * upper_weight).reshape(-1))
    return projected


def n_step_return(rewards, gamma, bootstrap_value=0.0, terminal=False):
    """Compute discounted n-step return for one transition sequence."""
    result = 0.0
    for reward in reversed(rewards):
        result = float(reward) + gamma * result
    if not terminal:
        result += (gamma ** len(rewards)) * float(bootstrap_value)
    return result
