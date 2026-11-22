import torch
from algorithms.llm_rl.ppo_objectives import clipped_policy_loss, approximate_kl


def test_ppo_loss_is_finite():
    new = torch.tensor([-0.2, -0.4])
    old = torch.tensor([-0.3, -0.3])
    advantages = torch.tensor([1.0, -1.0])
    loss = clipped_policy_loss(new, old, advantages)
    assert torch.isfinite(loss)


def test_kl_is_zero_for_identical_logps():
    values = torch.tensor([-0.1, -0.5])
    assert torch.allclose(approximate_kl(values, values), torch.tensor(0.0))
