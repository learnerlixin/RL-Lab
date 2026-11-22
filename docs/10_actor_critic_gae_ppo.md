# 10. Actor-Critic、GAE 与 PPO 更新

## Actor-Critic
Actor 输出策略分布 \(\pi_\theta(a|s)\)，Critic 输出状态价值 \(V_\phi(s)\)。Critic 为 Actor 提供低方差的优势估计。

## GAE(lambda)
先计算 TD residual：
\[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t).
\]
再反向递推：
\[
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}.
\]
\(\lambda=0\) 接近一步 TD，\(\lambda\to1\) 更接近 Monte Carlo 回报。此处 `dones` 表示真正终止并需要屏蔽 bootstrap 的标记；时间截断若需要 bootstrap，应单独保留 final observation 的价值。

## PPO minibatch update
策略概率比：
\[
r_t=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)).
\]
总损失：
\[
L=-L^{CLIP}+c_v L_V-c_e H(\pi_\theta).
\]
实现提供离散动作、向量观测的 rollout update 组件；采样器和环境交互循环仍由实验脚本负责。对图像、连续动作和并行环境需扩展网络与 rollout storage。
