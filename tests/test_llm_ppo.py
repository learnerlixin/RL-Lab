import torch
from torch import nn
from types import SimpleNamespace
from algorithms.llm_rl.llm_ppo import (\n    causal_token_logps, kl_shaped_rewards, llm_ppo_policy_step,\n    token_level_gae, llm_ppo_value_step,\n)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 12)
        self.head = nn.Linear(12, 13)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def test_kl_shaped_reward_places_terminal_reward_on_last_response_token():
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    logps = torch.zeros(1, 3)
    rewards = kl_shaped_rewards(torch.tensor([2.0]), logps, logps, mask, kl_coef=0.0)
    assert torch.allclose(rewards, torch.tensor([[0.0, 0.0, 2.0]]))


def test_llm_ppo_policy_step_runs():
    model = TinyLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = torch.tensor([[1, 2, 3, 4], [1, 3, 5, 6]])
    attention = torch.ones_like(ids)
    completion = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    old_logps = causal_token_logps(model, ids, attention).detach() - 0.1
    metrics = llm_ppo_policy_step(model, optimizer, ids, attention, completion,
                                 old_logps, torch.tensor([1.0, -1.0]),
                                 reference_token_logps=old_logps, kl_coef=0.01)
    assert torch.isfinite(torch.tensor(metrics["loss"]))


class TinyValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 8)
        self.head = nn.Linear(8, 1)

    def forward(self, input_ids, attention_mask):
        values = self.head(self.embedding(input_ids)).squeeze(-1)
        return SimpleNamespace(values=values)


def test_token_level_gae_bootstraps_time_limit_and_masks_padding():
    rewards = torch.tensor([[1.0, 1.0, 0.0]])
    values = torch.zeros(1, 4)
    terminals = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    advantages, returns = token_level_gae(
        rewards, values, terminals, mask, gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.tensor([[2.0, 1.0, 0.0]]))
    assert torch.allclose(returns, advantages)


def test_token_level_gae_stops_at_terminal():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[0.0, 5.0, 10.0]])
    terminals = torch.tensor([[0.0, 1.0]])
    mask = torch.ones_like(rewards)
    advantages, _ = token_level_gae(rewards, values, terminals, mask,
                                    gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.tensor([[8.0, -3.0]]))


def test_llm_ppo_value_step_updates_value_head():
    model = TinyValueModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = torch.tensor([[1, 2, 3], [2, 4, 5]])
    attention = torch.ones_like(ids)
    targets = torch.randn(2, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    before = model.head.weight.detach().clone()
    metrics = llm_ppo_value_step(model, optimizer, ids, attention, targets, mask)
    assert torch.isfinite(torch.tensor(metrics["value_loss"]))
    assert not torch.equal(before, model.head.weight)
