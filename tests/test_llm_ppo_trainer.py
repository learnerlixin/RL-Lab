from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from algorithms.llm_rl import LLMPPOTrainer


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=20, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        hidden = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


class TinyValueModel(nn.Module):
    def __init__(self, vocab_size=20, hidden_size=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(values=self.value_head(hidden).squeeze(-1))


def test_llm_ppo_trainer_runs_generation_reward_gae_and_ppo_epochs():
    policy = TinyCausalLM()
    value_model = TinyValueModel()
    reference = deepcopy(policy)
    reference.eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-3)
    generation_modes = []

    def generate_fn(model, prompt, max_new_tokens):
        generation_modes.append(model.training)
        assert not torch.is_grad_enabled()
        assert max_new_tokens == 3
        return [5 + len(prompt), 9]  # 9 is the EOS token

    trainer = LLMPPOTrainer(
        policy, value_model, optimizer, value_optimizer,
        generate_fn=generate_fn,
        reward_fn=lambda prompt, completion: float(completion[0]),
        pad_token_id=0, eos_token_id=9,
        reference_model=reference,
        ppo_epochs=2, minibatch_size=1, kl_coef=0.02,
    )
    policy.train()
    value_model.train()
    before_policy = policy.lm_head.weight.detach().clone()
    before_value = value_model.value_head.weight.detach().clone()
    metrics = trainer.train_batch([[1, 2], [3, 4, 5]], max_new_tokens=3)

    assert generation_modes == [False, False]
    assert policy.training is True
    assert value_model.training is True
    assert reference.training is False
    assert metrics["prompt_count"] == 2
    assert metrics["ppo_epochs"] == 2
    assert metrics["optimizer_steps"] == 4
    assert metrics["completion_tokens"] == 4
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["value_loss"]))
    assert torch.isfinite(torch.tensor(metrics["approx_kl"]))
    assert not torch.equal(before_policy, policy.lm_head.weight)
    assert not torch.equal(before_value, value_model.value_head.weight)


def test_llm_ppo_trainer_rejects_bad_ppo_schedule_and_empty_prompts():
    policy, value_model = TinyCausalLM(), TinyValueModel()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-3)
    generate = lambda model, prompt, limit: [7, 9]
    reward = lambda prompt, completion: 1.0

    with pytest.raises(ValueError):
        LLMPPOTrainer(policy, value_model, optimizer, value_optimizer,
                      generate, reward, pad_token_id=0, ppo_epochs=0)
    trainer = LLMPPOTrainer(
        policy, value_model, optimizer, value_optimizer,
        generate, reward, pad_token_id=0)
    with pytest.raises(ValueError):
        trainer.train_batch([])


def test_llm_ppo_trainer_uses_reference_kl_only_once():
    from unittest.mock import patch
    policy = TinyCausalLM()
    value_model = TinyValueModel()
    reference = deepcopy(policy)
    trainer = LLMPPOTrainer(
        policy, value_model,
        torch.optim.Adam(policy.parameters(), lr=1e-3),
        torch.optim.Adam(value_model.parameters(), lr=1e-3),
        generate_fn=lambda model, prompt, limit: [7, 9],
        reward_fn=lambda prompt, completion: 1.0,
        pad_token_id=0, eos_token_id=9,
        reference_model=reference, kl_coef=0.05,
    )
    seen = []
    original_step = __import__(
        "algorithms.llm_rl.llm_ppo", fromlist=["llm_ppo_policy_step"]
    ).llm_ppo_policy_step

    def wrapped_step(*args, **kwargs):
        seen.append(kwargs["kl_coef"])
        return original_step(*args, **kwargs)

    with patch("algorithms.llm_rl.llm_ppo_trainer.llm_ppo_policy_step",
               side_effect=wrapped_step):
        metrics = trainer.train_batch([[1, 2]], max_new_tokens=2)
    assert seen == [0.0]
    assert torch.isfinite(torch.tensor(metrics["approx_kl"]))
    assert metrics["approx_kl"] >= 0.0
