import torch
from algorithms.offline_rl.iql import (
    expectile_regression_loss, iql_q_target, advantage_weighted_bc_loss
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
