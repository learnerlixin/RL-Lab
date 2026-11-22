import pytest
import torch
from torch import nn

from algorithms.llm_rl import generate_causal_lm_completions


class FakeGenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.seen = None

    def generate(self, input_ids, attention_mask, num_return_sequences,
                 max_new_tokens, pad_token_id, eos_token_id, **kwargs):
        assert not torch.is_grad_enabled()
        self.seen = {
            "input_shape": tuple(input_ids.shape),
            "attention": attention_mask.clone(),
            "num_return_sequences": num_return_sequences,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "kwargs": kwargs,
        }
        continuations = torch.tensor([
            [4, 9, 0],
            [5, 6, 7],
            [8, 9, 0],
        ], device=input_ids.device)
        expanded_inputs = input_ids.repeat(num_return_sequences, 1)
        return torch.cat([expanded_inputs, continuations], dim=1)


def test_generation_adapter_batches_prompt_and_stops_at_eos():
    model = FakeGenerateModel()
    completions = generate_causal_lm_completions(
        model, [1, 2], group_size=3, pad_token_id=0,
        max_new_tokens=3, eos_token_id=9,
        generation_kwargs={"do_sample": True, "temperature": 0.8},
    )
    assert completions == [[4, 9], [5, 6, 7], [8, 9]]
    # generate expands this single prompt to num_return_sequences outputs.
    assert model.seen["input_shape"] == (1, 2)
    assert model.seen["attention"].tolist() == [[1, 1]]
    assert model.seen["kwargs"] == {"do_sample": True, "temperature": 0.8}


def test_generation_adapter_rejects_invalid_generation_settings():
    model = FakeGenerateModel()
    with pytest.raises(ValueError):
        generate_causal_lm_completions(
            model, [1], group_size=1, pad_token_id=0, max_new_tokens=2)
    with pytest.raises(ValueError):
        generate_causal_lm_completions(
            model, [1], group_size=2, pad_token_id=0, max_new_tokens=2,
            generation_kwargs={"max_new_tokens": 10})
    with pytest.raises(TypeError):
        generate_causal_lm_completions(
            nn.Linear(2, 2), [1], group_size=2,
            pad_token_id=0, max_new_tokens=2)
