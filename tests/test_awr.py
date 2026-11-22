import pytest
import torch

from algorithms.offline_rl.awr import awr_policy_loss, awr_value_loss, awr_weights


def test_awr_weights_are_detached_and_capped():
    advantages = torch.tensor([-1.0, 0.0, 100.0], requires_grad=True)
    weights = awr_weights(advantages, temperature=1.0, max_weight=5.0)
    expected = torch.tensor([0.36787945, 1.0, 5.0])
    assert torch.allclose(weights, expected)
    assert not weights.requires_grad


def test_awr_policy_loss_masked_weighted_likelihood():
    logps = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    advantages = torch.tensor([0.0, 0.0, 100.0])
    loss = awr_policy_loss(logps, advantages, max_weight=2.0,
                           mask=torch.tensor([1.0, 1.0, 0.0]))
    assert torch.allclose(loss, torch.tensor(1.5))
    loss.backward()
    assert logps.grad is not None


def test_awr_value_loss_and_invalid_shapes():
    values = torch.tensor([1.0, 2.0])
    returns = torch.tensor([2.0, 4.0])
    assert torch.allclose(awr_value_loss(values, returns), torch.tensor(2.5))
    with pytest.raises(ValueError):
        awr_value_loss(values, torch.ones(3))


class TinyGaussianPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Linear(3, 2)

    def log_prob(self, states, actions):
        mean = self.logit(states)
        return -0.5 * (actions - mean).square().sum(dim=-1)


def test_awr_train_step_updates_actor_and_value():
    from algorithms.offline_rl.awr import awr_train_step
    policy = TinyGaussianPolicy()
    value = nn.Linear(3, 1)
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    observations = torch.randn(8, 3)
    actions = torch.randn(8, 2)
    returns = torch.randn(8, 1)
    actor_before = policy.logit.weight.detach().clone()
    metrics = awr_train_step(policy, value, actor_optimizer, value_optimizer,
                             observations, actions, returns)
    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    assert torch.isfinite(torch.tensor(metrics["value_loss"]))
    assert not torch.equal(actor_before, policy.logit.weight)
