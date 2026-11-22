# 08. PPO clipped objective

PPO 使用新旧策略概率比：
\[
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
=\exp(\log\pi_\theta-\log\pi_{old}).
\]

Clipped surrogate：
\[
L^{CLIP}=\mathbb{E}_t[
\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)].
\]
训练时通常最小化其负值。裁剪限制概率比对目标的贡献，降低一次更新过大的风险；它并不保证策略变化绝对受限。

本仓库的 `clipped_policy_loss` 是目标函数级实现，输入必须已经对齐并处理好 padding/valid-token mask。完整 PPO 还需要 advantage/return 估计、old log-prob 缓存、mini-batch epoch、value loss、entropy bonus 和 KL 监控。LLM PPO 还要处理 token-level log-prob 与序列奖励分配。
