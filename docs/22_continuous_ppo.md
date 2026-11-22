# 22. 连续动作 PPO（Gaussian Actor-Critic）

离散 PPO 使用 categorical distribution；连续控制通常让 Actor 输出高斯分布参数：

\[
\mu_\theta(s),\quad \log\sigma_\theta(s)
\]

采样潜变量并压缩到有界动作区间：

\[
u\sim\mathcal N(\mu_\theta(s),\sigma_\theta(s)^2),\quad
a=\tanh(u)\odot c+b
\]

其中 \(c=(a_{max}-a_{min})/2\)，\(b=(a_{max}+a_{min})/2\)。

## Log probability 修正

因为动作经过 tanh 与线性缩放，不能直接使用 Normal 的 log probability。变量变换给出：

\[
\log\pi(a|s)=\sum_i\left[
\log\mathcal N(u_i;\mu_i,\sigma_i)
-\log(1-\tanh^2(u_i)+\epsilon)
-\log|c_i|
\right]
\]

实现中使用反双曲正切将 rollout 中保存的有界动作映回潜变量，以重新计算 PPO ratio。

## PPO clipped objective

\[
r_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t))
\]

\[
L_\pi=-\mathbb E\left[\min(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)\right]
\]

Critic 使用回报回归，整体优化策略损失、价值损失与熵正则。

## 代码边界

- ContinuousActorCritic：tanh-squashed Gaussian Actor 与状态价值 Critic。
- continuous_ppo_update：对已采集 rollout 做多 epoch minibatch 更新。
- GAE、环境交互、rollout 缓存和 time-limit bootstrap 仍由训练脚本负责。
- 对 tanh 变换后的策略，代码以单样本 -log_prob 作为熵估计；这不是解析熵。
