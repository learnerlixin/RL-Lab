# GRPO Rollout 管线与批次构造

GRPO 训练除了 clipped objective，还需要同一 prompt 下的 completion 分组、奖励计算、padding、completion mask 与旧策略 log-probability。本仓库提供轻量的模型无关 rollout 适配层；生成和 reward model 仍由调用方提供。

## 1. 收集 completion groups

```python
from algorithms.llm_rl import collect_grouped_rollouts

rollouts = collect_grouped_rollouts(
    prompt_token_ids=[[1, 2, 3], [4, 5]],
    generate_fn=lambda prompt, n: generator(prompt, num_return_sequences=n),
    reward_fn=lambda prompt, completion: reward_model_score(prompt, completion),
    group_size=4,
)
```

generate callback 接收 prompt token IDs 和 group size，返回对应数量的 completion token IDs。reward callback 为每条 completion 返回一个有限标量。

## 2. Padding、掩码与 old log-probs

`prepare_grpo_batch` 将变长 prompt/completion 组合成 [B,G,L] tensor，生成：

- input_ids：prompt + completion + padding；
- attention_mask：prompt 与 completion 的有效 token；
- completion_mask：只标识 completion token；
- old_logps：旧策略对 shifted completion tokens 的 log-prob；
- rewards / advantages：按 prompt 组归一化的相对优势；
- 可选 reference_logps：参考模型对 completion token 的 log-prob。

```python
from algorithms.llm_rl import prepare_grpo_batch, grpo_train_step

batch = prepare_grpo_batch(rollouts, old_policy, pad_token_id=0,
                           reference_model=reference_model)
metrics = grpo_train_step(
    policy, optimizer, batch["input_ids"], batch["attention_mask"],
    batch["completion_mask"], batch["old_logps"], batch["advantages"],
    reference_logps=batch.get("reference_logps"), kl_coef=0.02,
)
```

批次构造使用模型 forward 计算 old/reference log-prob，不负责调用 generate、不负责 tokenizer，也不负责 reward model 推理。调用方需确保 completion IDs 已经包含希望训练的 EOS token（如适用），并固定旧策略模型直到这一批 rollout 被消费。

## 3. 对齐约定

completion_mask 的位置与 input_ids 一致；因果语言模型的 next-token log-prob 会对 logits/labels/mask 做一次 shift，因此 trainer 接收的 old_logps 形状为 [B,G,L-1]。padding 和 prompt token 的掩码值为 0，只有生成的 completion 目标 token 参与策略损失。

`collect_grouped_rollouts` 与 `prepare_grpo_batch` 是 rollout/packing 工具，不是完整 RLHF 系统。批次更新之后如何刷新旧策略、采样调度、奖励归一化策略、KL 控制器和分布式生成仍由训练循环负责。
