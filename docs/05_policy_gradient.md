# 05. REINFORCE：直接优化策略

策略梯度直接参数化 \(\pi_\theta(a|s)\)，目标为：
\[
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0].
\]
Likelihood-ratio trick 得到：
\[
\nabla_\theta J(\theta)=
\mathbb{E}[\sum_t\nabla_\theta\log\pi_\theta(A_t|S_t)G_t].
\]
因此最小化：
\[
L(\theta)=-\sum_t\log\pi_\theta(A_t|S_t)G_t.
\]

代码中的 `Categorical(logits=...)` 定义离散动作分布，`log_prob(action)` 对应公式中的 log policy。

## Baseline
减去只依赖状态的 baseline \(b(S_t)\) 不改变期望梯度：
\[
\mathbb{E}[\nabla\log\pi(A|S)b(S)]=0.
\]
合适 baseline 可降低方差。Actor-Critic 用 critic 估计价值作为 baseline；当前 REINFORCE 是无 critic 的教学版本。

## 局限
REINFORCE 是 on-policy、整条轨迹更新，样本效率通常不高；当前实现没有 entropy bonus、value baseline、GAE 或并行采样。
