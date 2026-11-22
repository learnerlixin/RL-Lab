"""A minimal DPO optimization step for causal language models."""
import torch
import torch.nn.functional as F


def causal_completion_logps(model, input_ids, attention_mask, completion_mask):
    """Sum next-token log-probabilities over completion tokens only.

    All masks align with input_ids [B, L]. The first token has no preceding
    causal position and is therefore excluded after shifting.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,L]")
    if attention_mask.shape != input_ids.shape or completion_mask.shape != input_ids.shape:
        raise ValueError("attention_mask and completion_mask must match input_ids")
    shift_mask = (attention_mask[:, 1:] * completion_mask[:, 1:]).to(torch.float32)
    if (shift_mask.sum(dim=-1) <= 0).any():
        raise ValueError("each example must contain at least one valid completion token")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(
        -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * shift_mask.to(token_logps.dtype)).sum(dim=-1)


def dpo_train_step(policy, reference, optimizer, beta,
                   chosen_input_ids, chosen_attention_mask, chosen_completion_mask,
                   rejected_input_ids, rejected_attention_mask, rejected_completion_mask,
                   max_grad_norm=1.0):
    """One optimizer step on paired chosen/rejected completion batches.

    The reference model is temporarily put in evaluation mode and always run
    under no_grad; its original train/eval mode is restored after scoring.
    """
    if beta <= 0 or max_grad_norm <= 0:
        raise ValueError("beta and max_grad_norm must be positive")
    if chosen_input_ids.shape[0] != rejected_input_ids.shape[0]:
        raise ValueError("chosen and rejected batches must have the same size")
    policy_chosen = causal_completion_logps(
        policy, chosen_input_ids, chosen_attention_mask, chosen_completion_mask)
    policy_rejected = causal_completion_logps(
        policy, rejected_input_ids, rejected_attention_mask, rejected_completion_mask)
    reference_was_training = reference.training
    reference.eval()
    try:
        with torch.no_grad():
            reference_chosen = causal_completion_logps(
                reference, chosen_input_ids, chosen_attention_mask, chosen_completion_mask)
            reference_rejected = causal_completion_logps(
                reference, rejected_input_ids, rejected_attention_mask, rejected_completion_mask)
    finally:
        reference.train(reference_was_training)
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()),
            "chosen_logp": float(policy_chosen.detach().mean()),
            "rejected_logp": float(policy_rejected.detach().mean()),
            "preference_accuracy": float((logits.detach() > 0).float().mean()),
            "grad_norm": float(grad_norm.detach())}


__all__ = ["causal_completion_logps", "dpo_train_step"]
