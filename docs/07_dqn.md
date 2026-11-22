# 07. DQN：用神经网络近似动作价值

DQN 将表格 Q-learning 的 \(Q(s,a)\) 替换为神经网络 \(Q_\theta(s,a)\)。对经验回放中的转移 \((s,a,r,s',d)\)，目标为：

\[
y=r+\gamma(1-d)\max_{a'}Q_{\bar\theta}(s',a').
\]

损失常用 Huber loss：
\[
L(\theta)=\mathbb{E}[\operatorname{Huber}(Q_\theta(s,a)-y)].
\]

## 两个稳定化组件
- **Replay Buffer**：随机采样历史转移，降低相邻样本相关性并重复利用经验。
- **Target Network**：用较慢更新的 \(Q_{\bar\theta}\) 计算 target，减少目标随在线网络快速漂移。

## 代码阅读
`DQNAgent.update` 中 `gather` 取出执行动作对应的 Q 值；target 网络在 `no_grad` 下计算 bootstrap target；达到同步步数后复制 online 参数到 target。

## 局限
当前实现假设离散动作、向量观测，使用普通 DQN 的 max target；没有 Double DQN、Prioritized Replay、n-step return、图像 CNN 或并行环境。
