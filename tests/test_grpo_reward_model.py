import torch
from algorithms.llm_rl.grpo import grpo_clipped_loss, group_relative_advantages
from algorithms.llm_rl.reward_model import RewardModelHead


def test_grpo_group_advantages_are_centered_per_prompt():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    adv = group_relative_advantages(rewards)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_grpo_mask_ignores_padding_tokens():
    new = torch.tensor([[-0.1, -0.2, -9.0]])
    old = torch.tensor([[-0.2, -0.3, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv = torch.tensor([1.0])
    loss = grpo_clipped_loss(new, old, adv, mask)
    changed_new = new.clone()
    changed_old = old.clone()
    changed_new[0, 2] = 100.0
    changed_old[0, 2] = -100.0
    changed_padding = grpo_clipped_loss(changed_new, changed_old, adv, mask)
    assert torch.allclose(loss, changed_padding)


def test_reward_model_head_masked_pooling_shape():
    model = RewardModelHead(hidden_size=8)
    hidden = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    assert model(hidden, mask).shape == (2,)
