import torch

from algorithms.offline_rl.iql import (
    IQLAgent, expectile_regression_loss, iql_q_target, advantage_weighted_bc_loss
)


def test_expectile_weights_positive_residual_more_when_expectile_high():
    pred = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, -1.0])
    loss = expectile_regression_loss(pred, target, expectile=0.8)
    assert torch.allclose(loss, torch.tensor(0.5))


def test_iql_target_masks_terminal_bootstrap():
    target = iql_q_target(torch.tensor([1.0, 2.0]), torch.tensor([10.0, 10.0]),
                          torch.tensor([1.0, 0.0]), gamma=0.9)
    assert torch.allclose(target, torch.tensor([1.0, 11.0]))


def test_advantage_weighted_bc_is_finite():
    logps = torch.tensor([-0.5, -1.0])
    advantages = torch.tensor([1.0, -1.0])
    assert torch.isfinite(advantage_weighted_bc_loss(logps, advantages))


def test_iql_agent_update_and_bounded_actions():
    agent = IQLAgent(3, 2, action_low=[-2.0, -1.0], action_high=[2.0, 1.0],
                     hidden_dim=32, learning_rate=1e-3)
    batch = {
        "observations": torch.randn(16, 3),
        "actions": torch.empty(16, 2).uniform_(-0.8, 0.8),
        "rewards": torch.randn(16),
        "next_observations": torch.randn(16, 3),
        "terminals": torch.randint(0, 2, (16,)).float(),
    }
    before = next(agent.actor.parameters()).detach().clone()
    metrics = agent.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(before, next(agent.actor.parameters()))
    action = agent.act([0.0, 0.1, -0.1])
    assert action.shape == (2,)
    assert action[0] >= -2.0 and action[0] <= 2.0
    assert action[1] >= -1.0 and action[1] <= 1.0


def test_iql_agent_rejects_invalid_hyperparameters():
    import pytest
    with pytest.raises(ValueError):
        IQLAgent(3, 2, gamma=1.1)
    with pytest.raises(ValueError):
        IQLAgent(3, 2, expectile=1.0)

def test_iql_agent_rejects_nonpositive_learning_rate():
    import pytest
    with pytest.raises(ValueError):
        IQLAgent(3, 2, learning_rate=0.0)
