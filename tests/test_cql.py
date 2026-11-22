import torch
from torch import nn

from algorithms.offline_rl.cql import (
    continuous_cql_critic_step, continuous_cql_penalty, discrete_cql_penalty,
    discrete_cql_critic_step,
)


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))


def test_discrete_cql_penalty_is_finite_and_conservative():
    q_all = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    q_data = torch.tensor([1.0, 0.0])
    assert torch.isfinite(discrete_cql_penalty(q_all, q_data))
    low = discrete_cql_penalty(torch.tensor([[1.0, 1.0]]), torch.tensor([1.0]))
    high = discrete_cql_penalty(torch.tensor([[1.0, 5.0]]), torch.tensor([1.0]))
    assert high > low


def test_continuous_cql_penalty_and_log_density_correction():
    critic = TinyCritic()
    states = torch.randn(4, 2)
    data_actions = torch.zeros(4, 2)
    random_actions = torch.randn(4, 5, 2)
    policy_actions = torch.randn(4, 5, 2)
    random_logps = torch.zeros(4, 5)
    policy_logps = torch.zeros(4, 5)
    loss = continuous_cql_penalty(
        critic, states, data_actions, random_actions, policy_actions,
        random_logps, policy_logps)
    corrected = continuous_cql_penalty(
        critic, states, data_actions, random_actions, policy_actions,
        random_logps + 2.0, policy_logps + 2.0)
    assert torch.isfinite(loss)
    assert corrected < loss


def test_continuous_cql_critic_step_updates_parameters():
    critic = TinyCritic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    states = torch.randn(8, 2)
    actions = torch.randn(8, 2)
    targets = torch.randn(8, 1)
    random_actions = torch.randn(8, 4, 2)
    policy_actions = torch.randn(8, 4, 2)
    before = next(critic.parameters()).detach().clone()
    metrics = continuous_cql_critic_step(
        critic, optimizer, states, actions, targets, random_actions, policy_actions)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, next(critic.parameters()))


def test_continuous_cql_rejects_bad_action_shapes():
    critic = TinyCritic()
    states = torch.randn(2, 2)
    actions = torch.randn(2, 2)
    try:
        continuous_cql_penalty(
            critic, states, actions, torch.randn(2, 3), torch.randn(2, 3, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid candidate-action shape should fail")


class TinyDiscreteQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 4)

    def forward(self, states):
        return self.net(states)


def test_discrete_cql_critic_step_updates_parameters():
    q_network = TinyDiscreteQ()
    optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3)
    states = torch.randn(8, 3)
    actions = torch.randint(0, 4, (8,))
    targets = torch.randn(8)
    before = q_network.net.weight.detach().clone()
    metrics = discrete_cql_critic_step(q_network, optimizer, states, actions, targets)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, q_network.net.weight)


def test_continuous_cql_agent_performs_offline_update_and_bounds_action():
    from algorithms.offline_rl.cql import ContinuousCQLAgent
    agent = ContinuousCQLAgent(
        3, 2, action_low=[-2.0, -1.0], action_high=[2.0, 1.0],
        hidden_dim=32, num_action_samples=4)
    batch = {
        "observations": torch.randn(8, 3),
        "actions": torch.empty(8, 2).uniform_(-0.8, 0.8),
        "rewards": torch.randn(8),
        "next_observations": torch.randn(8, 3),
        "terminals": torch.randint(0, 2, (8,)).float(),
    }
    metrics = agent.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    action = agent.act([0.0, 0.1, -0.1])
    assert action.shape == (2,)
    assert -2.0 <= action[0] <= 2.0
    assert -1.0 <= action[1] <= 1.0
