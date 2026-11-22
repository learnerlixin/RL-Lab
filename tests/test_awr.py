import pytest
import torch

from algorithms.offline_rl.awr import awr_policy_loss, awr_value_loss, awr_weights


def test_awr_weights_are_detached_and_capped():
    advantages = torch.tensor([-1.0, 0.0, 100.0], requires_grad=True)
    weights = awr_weights(advantages, temperature=1.0, max_weight=5.0)
    assert torch.allclose(weights, torch.tensor([torch.exp(torch.tensor(-1.0)), 1.0, 5.0]))
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
