import pytest
import torch
from torch import nn

from algorithms.offline_rl.behavior_cloning import (
    behavior_cloning_train_step, continuous_behavior_cloning_loss,
    discrete_behavior_cloning_loss,
)


def test_discrete_bc_loss_supports_mask():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    actions = torch.tensor([0, 0, 1])
    loss = discrete_behavior_cloning_loss(logits, actions, torch.tensor([1.0, 0.0, 1.0]))
    assert loss < torch.tensor(0.1)


def test_continuous_bc_loss_and_mask():
    predicted = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
    target = torch.zeros_like(predicted)
    loss = continuous_behavior_cloning_loss(predicted, target, torch.tensor([1.0, 0.0]))
    assert torch.allclose(loss, torch.tensor(1.0))


def test_bc_train_step_updates_discrete_policy():
    policy = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    observations = torch.randn(16, 3)
    actions = torch.zeros(16, dtype=torch.long)
    before = policy.weight.detach().clone()
    metrics = behavior_cloning_train_step(policy, optimizer, observations, actions)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, policy.weight)


def test_bc_rejects_invalid_action_kind_and_shape():
    with pytest.raises(ValueError):
        discrete_behavior_cloning_loss(torch.zeros(2, 3), torch.zeros(3, dtype=torch.long))
    policy = nn.Linear(2, 2)
    with pytest.raises(ValueError):
        behavior_cloning_train_step(policy, torch.optim.SGD(policy.parameters(), lr=0.1),
                                    torch.ones(1, 2), torch.zeros(1), action_type="other")
