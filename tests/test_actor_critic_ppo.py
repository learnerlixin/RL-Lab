import torch
from algorithms.policy_based.actor_critic import ActorCritic, generalized_advantage_estimate
from algorithms.policy_based.ppo import ppo_update


def test_gae_one_step_terminal():
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([100.0])
    dones = torch.tensor([1.0])
    advantages, returns = generalized_advantage_estimate(rewards, values, next_values, dones)
    assert torch.allclose(advantages, torch.tensor([0.5]))
    assert torch.allclose(returns, torch.tensor([1.0]))


def test_ppo_update_runs():
    model = ActorCritic(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    states = torch.randn(16, 4)
    actions, old_logps, values = model.act(states)
    advantages = torch.randn(16)
    returns = values.detach() + advantages
    metrics = ppo_update(model, optimizer, states, actions, old_logps.detach(),
                         returns, advantages, epochs=2, batch_size=8)
    assert set(metrics) == {"policy_loss", "value_loss", "entropy"}
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
