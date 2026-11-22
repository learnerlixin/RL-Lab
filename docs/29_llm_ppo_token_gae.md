# LLM-PPO：Token-level GAE 与 Value Head 更新

LLM-PPO 的 actor 使用 completion token 上的 clipped surrogate；critic 需要 token-level value prediction，并用 shaped rewards 计算 GAE。

## 1. Token-level GAE

`token_level_gae` 接收：

- rewards: [B,T]，已经按 token 分配的 environment/KL-shaped reward；
- values: [B,T+1]，最后一列是 rollout 结束位置的 bootstrap value；
- terminals: [B,T]，真正 MDP terminal 为 1；
- mask: [B,T]，有效 completion token 为 1。

反向递推：

\[
\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t
\]
\[
A_t=\left(\delta_t+\gamma\lambda(1-d_t)A_{t+1}\right)m_t,
\qquad R_t=(A_t+V_t)m_t.
\]

时间限制截断不应自动设为 terminal；只要 bootstrap value 可用，最后一步就应继续 bootstrap。padding mask 会在递推中屏蔽无效 token。

## 2. Value head 更新

`llm_ppo_value_step` 接收 value model、optimizer、token IDs、attention mask、GAE returns 和 value mask，执行 masked MSE、梯度裁剪和 optimizer step。value model 可返回 [B,L]、[B,L,1] tensor，或暴露同形状的 values 字段。

## 3. Actor 更新与完整循环边界

`llm_ppo_policy_step` 使用 shifted completion mask、old token log-probs 和 token-level advantages 执行 PPO clipped actor update，可选 reference KL。一个完整循环还需调用方负责：

1. 从当前策略生成 completion；
2. 获取 reward model / environment 分数；
3. 用 KL shaping 将终端分数映射到 token rewards；
4. 从旧策略/value head 计算 old log-probs 和 values；
5. 调用 token GAE、critic step 和 actor step；
6. 按 PPO epoch/minibatch 调度并刷新 rollout。

这些函数是训练内核组件，不负责 tokenizer、生成调度或分布式执行。
