# LLM-PPO 端到端训练编排

`LLMPPOTrainer` 把原本需要调用方手动串联的 token-level PPO 训练步骤组合起来：

1. 按 prompt 调用 generation callback，产生 completion token IDs；
2. 用 reward callback 为每条 completion 计算一个标量奖励；
3. 统一 padding，建立 attention/completion masks；
4. 计算 old-policy 和可选 reference-policy token log-prob；
5. 计算旧 value predictions、KL-shaped token rewards 和 token-level GAE；
6. 固定本批次的 old log-prob、reference log-prob、advantages 和 value targets；
7. 在固定 rollout 上执行多个 PPO epoch 与 prompt minibatch 的 value/actor 更新。

## 使用示例

```python
from algorithms.llm_rl import LLMPPOTrainer, generate_causal_lm_completions

trainer = LLMPPOTrainer(
    policy=policy,
    value_model=value_model,
    policy_optimizer=policy_optimizer,
    value_optimizer=value_optimizer,
    generate_fn=lambda model, prompt_ids, max_new_tokens:
        generate_causal_lm_completions(
            model, prompt_ids, group_size=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            generation_kwargs={"do_sample": True, "temperature": 0.8},
        )[0],
    reward_fn=score_completion,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    reference_model=reference_model,
    clip_range=0.2,
    kl_coef=0.02,
    ppo_epochs=4,
    minibatch_size=2,
)
metrics = trainer.train_batch(prompt_token_ids, max_new_tokens=128)
```

## 关键对齐约定

- callback 返回 completion token IDs，不包含 prompt。
- 如果 completion 含 EOS，EOS 会包含在 completion mask 内，并在 token GAE 中作为真正 terminal；达到 max_new_tokens 但未生成 EOS 时保留 bootstrap。
- value model 必须对每个 input token 返回一个标量 state value，输出形状为 [B,L] 或 [B,L,1]。
- policy/reference 采用 causal next-token log-prob，形状为 [B,L-1]；completion mask 会同步右移。
- 每个 rollout 的 old/reference log-prob 与 advantages 只计算一次，多个 PPO epoch 之间保持不变。
- generation、旧策略评分、value target 计算和策略更新期间 dropout 关闭，训练前的 train/eval 状态会在结束后恢复。
- 支持 policy/reference/value model 位于不同 device；trainer 会按模型 device 转移输入与 target tensor。

该实现是单机 callback-driven 教学 trainer；它不负责 tokenizer 加载、reward model 服务部署、异步/分布式 rollout、动态 KL controller、packed sequence attention 或完整的 production RLHF scheduler。
