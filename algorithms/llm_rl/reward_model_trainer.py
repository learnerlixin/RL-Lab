"""Pairwise reward-model training step for transformer-style encoders."""
import torch
from .preference_objectives import bradley_terry_loss


def _last_hidden_state(encoder, input_ids, attention_mask):
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise TypeError("encoder output must expose last_hidden_state or tuple[0]")


def reward_model_train_step(encoder, reward_head, optimizer,
                            chosen_input_ids, chosen_attention_mask,
                            rejected_input_ids, rejected_attention_mask,
                            max_grad_norm=1.0):
    """Train encoder + scalar reward head from chosen/rejected sequence pairs."""
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    chosen_hidden = _last_hidden_state(encoder, chosen_input_ids, chosen_attention_mask)
    rejected_hidden = _last_hidden_state(encoder, rejected_input_ids, rejected_attention_mask)
    chosen_rewards = reward_head(chosen_hidden, chosen_attention_mask)
    rejected_rewards = reward_head(rejected_hidden, rejected_attention_mask)
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError("chosen and rejected batches must have matching batch sizes")
    loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    optimizer.step()
    with torch.no_grad():
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
    return {"loss": float(loss.detach()), "preference_accuracy": float(accuracy),
            "reward_margin": float(reward_margin), "grad_norm": float(grad_norm)}


__all__ = ["reward_model_train_step"]
