from copy import deepcopy

import torch
from torch import nn
from types import SimpleNamespace

from algorithms.llm_rl.rollouts import collect_grouped_rollouts, prepare_grpo_batch
from algorithms.llm_rl.grpo_trainer import grpo_train_step


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=17, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


def test_collect_grouped_rollouts_calls_generation_and_reward_callbacks():
    rollouts = collect_grouped_rollouts(
        [[1, 2], [3]],
        generate_fn=lambda prompt, n: [[4 + i, 5] for i in range(n)],
        reward_fn=lambda prompt, completion: float(sum(completion)),
        group_size=2,
    )
    assert len(rollouts["completion_token_ids"]) == 2
    assert rollouts["rewards"].shape == (2, 2)
    assert rollouts["rewards"][0, 0] != rollouts["rewards"][0, 1]


def test_prepare_grpo_batch_builds_masks_and_old_logps():
    model = TinyCausalLM()
    rollouts = {
        "prompt_token_ids": [[1, 2], [3]],
        "completion_token_ids": [[[4], [5, 6]], [[7, 8], [9]]],
        "rewards": torch.tensor([[1.0, 2.0], [4.0, 1.0]]),
    }
    batch = prepare_grpo_batch(rollouts, model, pad_token_id=0,
                               reference_model=deepcopy(model))
    assert batch["input_ids"].shape == (2, 2, 4)
    assert batch["old_logps"].shape == (2, 2, 3)
    assert batch["reference_logps"].shape == (2, 2, 3)
    assert batch["completion_mask"][0, 0].sum().item() == 1
    assert batch["completion_mask"][0, 1].sum().item() == 2
    assert batch["attention_mask"][1, 0].sum().item() == 3


def test_prepared_rollout_batch_runs_grpo_update():
    model = TinyCausalLM()
    old_model = deepcopy(model)
    rollouts = collect_grouped_rollouts(
        [[1, 2], [3, 4]],
        generate_fn=lambda prompt, n: [[5 + i] for i in range(n)],
        reward_fn=lambda prompt, completion: float(completion[0]),
        group_size=2,
    )
    batch = prepare_grpo_batch(rollouts, old_model, pad_token_id=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = grpo_train_step(
        model, optimizer, batch["input_ids"], batch["attention_mask"],
        batch["completion_mask"], batch["old_logps"], batch["advantages"])
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_collect_grouped_rollouts_rejects_single_completion_group():
    import pytest
    with pytest.raises(ValueError):
        collect_grouped_rollouts([[1]], lambda prompt, n: [[2]], lambda p, c: 1.0,
                                 group_size=1)
