from types import SimpleNamespace
import torch
from torch import nn

from algorithms.llm_rl.grpo_trainer import (
    causal_lm_completion_logps, grpo_train_step
)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=13, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_completion_logps_excludes_prompt_and_padding():
    logits = torch.randn(2, 5, 13)
    ids = torch.randint(0, 13, (2, 5))
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]])
    logps, shifted_mask = causal_lm_completion_logps(logits, ids, mask)
    assert logps.shape == (2, 4)
    assert torch.equal(shifted_mask, mask[:, 1:].float())


def test_grpo_train_step_updates_model_parameters():
    torch.manual_seed(4)
    model = TinyCausalLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    ids = torch.randint(0, 13, (3, 6))
    attention = torch.ones_like(ids)
    completion = torch.tensor([[0, 0, 1, 1, 1, 0]] * 3)
    old_logps = torch.full((3, 5), -2.5)
    advantages = torch.tensor([1.0, -0.5, 0.2])
    before = model.lm_head.weight.detach().clone()
    metrics = grpo_train_step(
        model, optimizer, ids, attention, completion,
        old_logps, advantages, clip_range=0.2
    )
    assert metrics["loss"] == metrics["loss"]
    assert metrics["grad_norm"] >= 0
    assert not torch.equal(before, model.lm_head.weight)


def test_grpo_train_step_supports_prompt_group_shape():
    model = TinyCausalLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 13, (2, 3, 5))
    attention = torch.ones_like(ids)
    completion = torch.tensor([0, 0, 1, 1, 1]).view(1, 1, 5).expand(2, 3, 5)
    old_logps = torch.full((2, 3, 4), -2.0)
    advantages = torch.randn(2, 3)
    metrics = grpo_train_step(model, optimizer, ids, attention, completion,
                              old_logps, advantages)
    assert "approx_kl" in metrics
