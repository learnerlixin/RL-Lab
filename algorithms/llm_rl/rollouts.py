"""Model-agnostic GRPO rollout collection and batch preparation.

Token generation and reward scoring are injected as callables so this module
does not impose a tokenizer, Transformers version, or decoding policy.
"""
import torch

from .grpo import group_relative_advantages
from .grpo_trainer import causal_lm_completion_logps


def collect_grouped_rollouts(prompt_token_ids, generate_fn, reward_fn, group_size=4):
    """Generate completion groups and scalar rewards through user callbacks.

    generate_fn(prompt_ids, group_size) must return exactly group_size token
    sequences. reward_fn(prompt_ids, completion_ids) returns one finite scalar
    reward per completion.
    """
    if group_size < 2:
        raise ValueError("GRPO requires at least two completions per prompt")
    all_completions, all_rewards = [], []
    normalized_prompts = []
    for prompt in prompt_token_ids:
        prompt = [int(token) for token in prompt]
        if not prompt:
            raise ValueError("prompt token sequences must be non-empty")
        completions = generate_fn(prompt, group_size)
        if len(completions) != group_size:
            raise ValueError("generate_fn must return exactly group_size completions")
        normalized = [[int(token) for token in completion] for completion in completions]
        if any(not completion for completion in normalized):
            raise ValueError("generated completion sequences must be non-empty")
        rewards = [float(reward_fn(prompt, completion)) for completion in normalized]
        if not torch.isfinite(torch.tensor(rewards)).all():
            raise ValueError("reward_fn must return finite scalar rewards")
        normalized_prompts.append(prompt)
        all_completions.append(normalized)
        all_rewards.append(rewards)
    if not all_completions:
        raise ValueError("at least one prompt is required")
    return {"prompt_token_ids": normalized_prompts,
            "completion_token_ids": all_completions,
            "rewards": torch.tensor(all_rewards, dtype=torch.float32)}


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def _sequence_logps(model, input_ids, attention_mask, completion_mask):
    device = _model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    completion_mask = completion_mask.to(device)
    was_training = model.training
    model.eval()
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output[0]
        return causal_lm_completion_logps(logits, input_ids, completion_mask)[0]
    finally:
        model.train(was_training)



def generate_causal_lm_completions(model, prompt_token_ids, group_size,
                                   pad_token_id, max_new_tokens,
                                   eos_token_id=None, generation_kwargs=None):
    """Generate one completion group with a Hugging Face-style causal LM.

    This adapter only relies on model.generate and does not import Transformers.
    generation_kwargs can include sampling settings such as do_sample,
    temperature and top_p. Special arguments controlled by this helper cannot
    be overridden there.
    """
    if not hasattr(model, "generate"):
        raise TypeError("model must expose a generate method")
    if group_size < 2 or max_new_tokens < 1:
        raise ValueError("group_size must be >=2 and max_new_tokens positive")
    prompt = [int(token) for token in prompt_token_ids]
    if not prompt:
        raise ValueError("prompt_token_ids must be non-empty")
    options = dict(generation_kwargs or {})
    reserved = {
        "input_ids", "attention_mask", "num_return_sequences",
        "max_new_tokens", "pad_token_id", "eos_token_id",
    }
    conflicts = reserved.intersection(options)
    if conflicts:
        raise ValueError(f"generation_kwargs cannot override: {sorted(conflicts)}")
    device = _model_device(model)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=group_size,
            max_new_tokens=max_new_tokens,
            pad_token_id=int(pad_token_id),
            eos_token_id=None if eos_token_id is None else int(eos_token_id),
            **options,
        )
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    generated = torch.as_tensor(generated)
    if generated.ndim != 2 or generated.shape[0] != group_size:
        raise ValueError("generate must return token IDs with shape [group_size, sequence_length]")
    if generated.shape[1] < len(prompt):
        raise ValueError("generated sequences cannot be shorter than the prompt")
    completions = []
    for sequence in generated[:, len(prompt):].detach().cpu().tolist():
        sequence = [int(token) for token in sequence]
        if eos_token_id is not None and int(eos_token_id) in sequence:
            sequence = sequence[:sequence.index(int(eos_token_id)) + 1]
        else:
            while sequence and sequence[-1] == int(pad_token_id):
                sequence.pop()
        if not sequence:
            raise ValueError("model generated an empty completion")
        completions.append(sequence)
    return completions


def prepare_grpo_batch(rollouts, old_policy_model, pad_token_id,
                       reference_model=None, device=None):
    """Pad prompt/completion groups and calculate behavior/reference token logps.

    Returns tensors accepted by grpo_train_step. All prompt and completion
    token sequences are unpadded integer lists; completion_mask marks only
    generated tokens, while attention_mask marks both prompt and completion.
    """
    prompts = rollouts["prompt_token_ids"]
    completions = rollouts["completion_token_ids"]
    rewards = torch.as_tensor(rollouts["rewards"], dtype=torch.float32)
    if len(prompts) == 0 or len(prompts) != len(completions):
        raise ValueError("prompts and completion groups must have the same nonzero length")
    group_size = len(completions[0])
    if group_size < 2 or any(len(group) != group_size for group in completions):
        raise ValueError("all prompts must have the same group size >= 2")
    if rewards.shape != (len(prompts), group_size):
        raise ValueError("rewards must have shape [batch, group_size]")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    if any(not prompt for prompt in prompts):
        raise ValueError("prompt sequences must be non-empty")
    if any(not completion for group in completions for completion in group):
        raise ValueError("completion sequences must be non-empty")

    max_length = max(len(prompt) + len(completion)
                     for prompt, group in zip(prompts, completions)
                     for completion in group)
    shape = (len(prompts), group_size, max_length)
    input_ids = torch.full(shape, int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros(shape, dtype=torch.long)
    completion_mask = torch.zeros(shape, dtype=torch.float32)
    for batch_index, (prompt, group) in enumerate(zip(prompts, completions)):
        for group_index, completion in enumerate(group):
            sequence = [int(t) for t in prompt] + [int(t) for t in completion]
            length = len(sequence)
            input_ids[batch_index, group_index, :length] = torch.tensor(sequence)
            attention_mask[batch_index, group_index, :length] = 1
            completion_mask[batch_index, group_index, len(prompt):length] = 1

    old_device = _model_device(old_policy_model)
    flat_ids = input_ids.reshape(-1, max_length).to(old_device)
    flat_attention = attention_mask.reshape(-1, max_length).to(old_device)
    flat_completion = completion_mask.reshape(-1, max_length).to(old_device)
    old_logps = _sequence_logps(old_policy_model, flat_ids, flat_attention, flat_completion)
    old_logps = old_logps.reshape(len(prompts), group_size, max_length - 1)
    batch_device = torch.device(device) if device is not None else old_device
    batch = {
        "input_ids": input_ids.to(batch_device),
        "attention_mask": attention_mask.to(batch_device),
        "completion_mask": completion_mask.to(batch_device),
        "old_logps": old_logps.to(batch_device),
        "rewards": rewards.to(batch_device),
        "advantages": group_relative_advantages(rewards).to(batch_device),
    }
    if reference_model is not None:
        ref_device = _model_device(reference_model)
        ref_logps = _sequence_logps(
            reference_model,
            input_ids.reshape(-1, max_length).to(ref_device),
            attention_mask.reshape(-1, max_length).to(ref_device),
            completion_mask.reshape(-1, max_length).to(ref_device),
        )
        batch["reference_logps"] = ref_logps.reshape(
            len(prompts), group_size, max_length - 1).to(batch_device)
    return batch


__all__ = ["collect_grouped_rollouts", "generate_causal_lm_completions", "prepare_grpo_batch"]
