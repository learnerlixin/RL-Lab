import torch
from torch import nn
from types import SimpleNamespace
from algorithms.llm_rl.llm_ppo import causal_token_logps, kl_shaped_rewards, llm_ppo_policy_step


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 12)
        self.head = nn.Linear(12, 13)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def test_kl_shaped_reward_places_terminal_reward_on_last_response_token():
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
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
