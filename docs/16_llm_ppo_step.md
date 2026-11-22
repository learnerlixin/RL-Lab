# 16. LLM PPO：token log-prob、KL shaping 与策略更新

LLM PPO 的动作是生成 token。对 completion token 序列，策略概率比在 token 级计算：
\[
\rho_{i,t}=\exp(\log\pi_\theta(y_{i,t}|x_i,y_{i,<t})
-\log\pi_{old}(y_{i,t}|x_i,y_{i,<t})).
\]
将序列优势广播到有效 completion token，构造 clipped surrogate，并用 attention/completion mask 排除 prompt 和 padding。

## KL shaping
常见 sampled KL 估计可用于将策略偏离参考模型的代价纳入奖励：
\[
\tilde r_t=r_t-\beta\hat{KL}_t.
\]
如果奖励模型给出序列级分数，可将终端奖励放在最后一个有效 completion token，再结合 token KL shaping。不同实现会选择不同 KL estimator 和 reward placement，必须保持训练与评估定义一致。

## 代码边界
`llm_ppo_policy_step` 实现 actor 的 token-level PPO clipped 更新；critic、GAE、rollout generation、reward-model inference、分布式采样和 checkpoint 由外部训练系统提供。它是可读的核心训练步骤，不是完整 RLHF pipeline。
