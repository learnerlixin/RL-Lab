# 12. Soft Actor-Critic（SAC）

SAC 是 off-policy、最大熵 actor-critic。除了最大化回报，也鼓励策略保持熵：
\[
J(\pi)=\mathbb{E}\left[\sum_t r(s_t,a_t)+\alpha\mathcal H(\pi(\cdot|s_t))\right].
\]

## Soft Bellman target
\[
y=r+\gamma(1-d)\left[
\min_{j\in\{1,2\}}Q_{\bar\phi_j}(s',a')-
\alpha\log\pi_\theta(a'|s')
\right],\quad a'\sim\pi_\theta.
\]
两个 critic 取最小值以减轻 Q 过估计。Actor 最小化：
\[
L_\pi=\mathbb E[\alpha\log\pi(a|s)-\min_j Q_j(s,a)].
\]

## Squashed Gaussian
连续动作策略先采样高斯变量，再经 tanh 映射到动作范围。log probability 必须包含 tanh Jacobian 修正；代码还对动作 scale 做概率密度变换修正。

## 自动温度
常见做法学习 \(\alpha\)，使策略熵接近目标熵。当前实现使用双 Q、target Q、Polyak 更新与自动温度调节，面向连续向量观测/动作的教学基线。
