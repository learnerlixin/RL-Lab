from types import SimpleNamespace

import torch
from torch import nn

from algorithms.llm_rl import GRPOTrainer


class DropoutCausalLM(nn.Module):
    def __init__(self, vocab_size=17, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        hidden = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_grpo_trainer_connects_generation_packing_and_update():
    policy = DropoutCausalLM()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    generation_modes = []

    def generate_fn(model, prompt, count):
        generation_modes.append(model.training)
        assert not torch.is_grad_enabled()
        return [[5 + index, 6] for index in range(count)]

    trainer = GRPOTrainer(
        policy, optimizer, generate_fn,
        reward_fn=lambda prompt, completion: float(sum(completion)),
        pad_token_id=0, clip_range=0.2,
    )
    policy.train()
    before = policy.lm_head.weight.detach().clone()
    metrics = trainer.train_batch([[1, 2], [3, 4]], group_size=2)
    assert generation_modes == [False, False]
    assert policy.training is True
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["prompt_count"] == 2
    assert metrics["group_size"] == 2
    assert metrics["reward_std"] > 0
    assert not torch.equal(before, policy.lm_head.weight)


def test_grpo_trainer_restores_eval_mode_when_already_evaluating():
    policy = DropoutCausalLM()
    policy.eval()
    trainer = GRPOTrainer(
        policy, torch.optim.Adam(policy.parameters(), lr=1e-3),
        generate_fn=lambda model, prompt, count: [[7 + i] for i in range(count)],
        reward_fn=lambda prompt, completion: float(completion[0]),
        pad_token_id=0,
    )
    trainer.train_batch([[1, 2]], group_size=2)
    assert policy.training is False
