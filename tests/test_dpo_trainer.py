import copy
import torch
from torch import nn
from types import SimpleNamespace
from algorithms.llm_rl.dpo_trainer import causal_completion_logps, dpo_train_step


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=11, hidden=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


def test_completion_logps_excludes_prompt_tokens():
    model = TinyCausalLM()
    ids = torch.tensor([[1, 2, 3, 4]])
    attention = torch.ones_like(ids)
    completion = torch.tensor([[0, 0, 1, 1]])
    value = causal_completion_logps(model, ids, attention, completion)
    changed_prompt_mask = torch.tensor([[1, 1, 1, 1]])
    full = causal_completion_logps(model, ids, attention, changed_prompt_mask)
    assert value.shape == (1,)
    assert not torch.allclose(value, full)


def test_dpo_train_step_runs_and_reference_is_frozen():
    policy = TinyCausalLM()
    reference = copy.deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    chosen = torch.tensor([[1, 2, 3, 4], [1, 3, 4, 5]])
    rejected = torch.tensor([[1, 2, 5, 6], [1, 3, 6, 7]])
    mask = torch.ones_like(chosen)
    completion = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    metrics = dpo_train_step(policy, reference, optimizer, 0.1,
                             chosen, mask, completion, rejected, mask, completion)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert all(not p.requires_grad or p.grad is None for p in reference.parameters())
    assert reference.training  # original mode is restored after reference scoring


def test_dpo_train_step_rejects_nonpositive_beta():
    import pytest
    policy, reference = TinyCausalLM(), TinyCausalLM()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    completion = torch.tensor([[0, 0, 1]])
    with pytest.raises(ValueError):
        dpo_train_step(policy, reference, optimizer, 0.0,
                       ids, mask, completion, ids, mask, completion)
