"""A minimal DPO optimization step for causal language models."""
import torch
import torch.nn.functional as F


def causal_completion_logps(model, input_ids, attention_mask, completion_mask):
    """Sum next-token log-probabilities over completion tokens only.

    All masks align with input_ids [B, L]. The first token has no preceding
    causal position and is therefore excluded after shifting.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = (attention_mask[:, 1:] * completion_mask[:, 1:]).to(logits.dtype)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(
        -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * shift_mask).sum(dim=-1)


def dpo_train_step(policy, reference, optimizer, beta,
                   chosen_input_ids, chosen_attention_mask, chosen_completion_mask,
                   rejected_input_ids, rejected_attention_mask, rejected_completion_mask):
    """One optimizer step on paired chosen/rejected completion batches."""
    policy_chosen = causal_completion_logps(
        policy, chosen_input_ids, chosen_attention_mask, chosen_completion_mask)
    policy_rejected = causal_completion_logps(
        policy, rejected_input_ids, rejected_attention_mask, rejected_completion_mask)
    with torch.no_grad():
        reference_chosen = causal_completion_logps(
            reference, chosen_input_ids, chosen_attention_mask, chosen_completion_mask)
        reference_rejected = causal_completion_logps(
            reference, rejected_input_ids, rejected_attention_mask, rejected_completion_mask)
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.detach()),
            "chosen_logp": float(policy_chosen.detach().mean()),
            "rejected_logp": float(policy_rejected.detach().mean()),
            "preference_accuracy": float((logits.detach() > 0).float().mean())}
