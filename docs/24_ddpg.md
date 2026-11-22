# 24. Deep Deterministic Policy Gradient（DDPG）

DDPG 将确定性策略梯度与 off-policy critic 学习结合，适用于连续动作空间。

## Critic 目标

使用目标 Actor 和目标 Critic 构造 bootstrap target：

\[
y_t=r_t+\gamma(1-d_t)Q_{\phi'}(s_{t+1},
\mu_{\theta'}(s_{t+1}))
\]

Critic 最小化：

\[
L_Q(\phi)=\mathbb E[(Q_\phi(s_t,a_t)-y_t)^2]
\]

## Actor 目标

确定性策略通过 Critic 对动作的梯度更新：

\[
L_\mu(\theta)=-\mathbb E[Q_\phi(s,\mu_\theta(s))]
\]

## Target network

在线网络更新后，对目标网络做 Polyak averaging：

\[
\theta'\leftarrow(1-\tau)\theta'+\tau\theta
\]

Critic 参数同理。与 TD3 相比，基础 DDPG 使用单 Critic、每步更新 Actor，且没有 target policy smoothing，因此更容易受到 Q 过估计影响。

## 实现边界

- DDPGAgent 提供 deterministic actor、单 Q critic、target networks 与 replay update。
- 探索噪声由 act 的 noise_std 参数控制。
- replay buffer 保存 terminated 标记；time-limit truncated transition 应保留 bootstrap（terminated=False）。
- 训练循环、评估、checkpoint 与实验记录由调用方负责。
