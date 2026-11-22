import torch
from algorithms.llm_rl.preference_objectives import bradley_terry_loss, dpo_loss, group_relative_advantages


def test_bradley_terry_loss_ranks_chosen_higher():
    good = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    bad = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([2.0]))
    assert good < bad


def test_dpo_outputs_finite_loss():
    loss, chosen, rejected = dpo_loss(torch.tensor([-1.0]), torch.tensor([-2.0]),
                                     torch.tensor([-1.5]), torch.tensor([-2.0]), beta=0.1)
    assert torch.isfinite(loss)
    assert chosen.item() > rejected.item()


def test_group_advantages_centered():
    adv = group_relative_advantages(torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(adv.mean(-1), torch.zeros(1), atol=1e-6)
