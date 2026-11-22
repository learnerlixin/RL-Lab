"""A model-agnostic GRPO optimization step for causal language models.

Generation and reward scoring intentionally remain caller-owned. This module
handles causal token log-probabilities, group-shaped rollout tensors, the
clipped GRPO objective, optional reference KL, and one optimizer step.
"""
import torch
import torch.nn.functional as F

from .grpo import grpo_clipped_loss, group_relative_advantages


def causal_lm_completion_logps(logits, input_ids, completion_mask):
    """Return next-token log-probs and mask, excluding prompt/padding positions.

    logits/input_ids/completion_mask have shape [batch, sequence]. The mask marks
    completion tokens at their input_ids positions; causal shifting is applied
    consistently to logits, labels, and mask.
    """
    if logits.ndim != 3 or input_ids.ndim != 2 or completion_mask.shape != input_ids.shape:
        raise ValueError("expected logits [B,L,V] and ids/mask [B,L]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids sequence shapes must match")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = input_ids[:, 1:]
    shifted_mask = completion_mask[:, 1:].to(dtype=shifted_logits.dtype)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, shifted_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logps, shifted_mask


def _model_logits(model, input_ids, attention_mask):
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("model output must expose .logits or tuple[0]")


def grpo_train_step(model, optimizer, input_ids, attention_mask,
                    completion_mask, old_logps, advantages,
                    reference_logps=None, clip_range=0.2, kl_coef=0.0,
                    max_grad_norm=1.0):
    """Run one GRPO update.

    Accepts [B,L] or [B,G,L] token batches. In the latter case, old/reference
    log-probs must be [B,G,L-1], and advantages [B,G].
    """
    grouped = input_ids.ndim == 3
    if grouped:
        batch, group, length = input_ids.shape
        input_ids = input_ids.reshape(batch * group, length)
        attention_mask = attention_mask.reshape(batch * group, length)
        completion_mask = completion_mask.reshape(batch * group, length)
        old_logps = old_logps.reshape(batch * group, -1)
        advantages = advantages.reshape(batch * group)
        if reference_logps is not None:
            reference_logps = reference_logps.reshape(batch * group, -1)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,L] or [B,G,L]")

    logits = _model_logits(model, input_ids, attention_mask)
    new_logps, shifted_mask = causal_lm_completion_logps(logits, input_ids, completion_mask)
    if old_logps.shape != new_logps.shape:
        raise ValueError("old_logps must match shifted completion log-prob shape")
    if reference_logps is not None and reference_logps.shape != new_logps.shape:
        raise ValueError("reference_logps must match shifted completion log-prob shape")

    loss = grpo_clipped_loss(
        new_logps, old_logps.detach(), advantages,
        shifted_mask, clip_range=clip_range,
        reference_logps=None if reference_logps is None else reference_logps.detach(),
        kl_coef=kl_coef,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    with torch.no_grad():
        mask_sum = shifted_mask.sum().clamp_min(1.0)
        approx_kl = (((old_logps - new_logps) * shifted_mask).sum() / mask_sum).item()
    return {"loss": float(loss.detach().item()),
            "approx_kl": float(approx_kl),
            "grad_norm": float(grad_norm.detach().item())}


__all__ = ["causal_lm_completion_logps", "grpo_train_step",
           "group_relative_advantages"]
