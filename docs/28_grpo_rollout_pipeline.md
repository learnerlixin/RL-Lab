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


## 4. 一步式 GRPOTrainer

若不需要自定义 rollout 迭代器，可以使用 `GRPOTrainer` 将 callback generation、reward scoring、packing 和单次策略更新组合：

```python
from algorithms.llm_rl import GRPOTrainer

trainer = GRPOTrainer(
    policy, optimizer,
    generate_fn=lambda policy, prompt_ids, group_size: generate(
        policy, prompt_ids, num_return_sequences=group_size),
    reward_fn=lambda prompt_ids, completion_ids: reward(
        prompt_ids, completion_ids),
    pad_token_id=tokenizer.pad_token_id,
    reference_model=reference_model,
    kl_coef=0.02,
    ppo_epochs=4, minibatch_size=2,
)
metrics = trainer.train_batch(prompt_token_ids, group_size=4)
```

Trainer 在生成、old-policy log-prob 计算和本次更新期间统一关闭 dropout，保证 PPO ratio 的 old/new log-prob 来自相同 dropout 设置；更新结束后恢复 policy 的原始 train/eval 模式。Trainer 默认执行一个 PPO epoch，也支持通过 `ppo_epochs` 和 `minibatch_size` 在固定 rollout 上重复更新。old/reference log-prob 与 group advantages 每个 rollout 只计算一次，在所有 epoch/minibatch 之间保持固定；模型参数更新后，不能把当前策略重新算出的 log-prob 替换原有 old log-prob。


## 5. Hugging Face-style generate adapter

若模型暴露标准 `generate(input_ids, attention_mask, ...)` 方法，可以直接使用 `generate_causal_lm_completions`：

```python
from algorithms.llm_rl import GRPOTrainer, generate_causal_lm_completions

trainer = GRPOTrainer(
    policy, optimizer,
    generate_fn=lambda model, prompt, group_size:
        generate_causal_lm_completions(
            model, prompt, group_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128,
            generation_kwargs={"do_sample": True, "temperature": 0.8, "top_p": 0.95},
        ),
    reward_fn=score_completion,
    pad_token_id=tokenizer.pad_token_id,
)
```

Adapter 将单条 prompt 以 batch size 1 传入，由 `num_return_sequences=group_size` 让模型生成接口扩展返回组；随后在返回序列中移除 prompt，只保留 completion，并在首次 EOS 处截断。这样不会出现输入 batch 和 `num_return_sequences` 双重扩展导致的 group_size² 输出；不依赖 Transformers import，但 model 必须提供兼容的 `generate` 方法。tokenizer 管理、EOS/PAD ID 和 reward callback 仍由调用方提供。
