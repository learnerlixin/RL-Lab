# 19. Rainbow DQN：从组件到可训练 Agent

Rainbow 将多个互补改进组合到 DQN：Double DQN、Prioritized Experience Replay（PER）、Dueling Network、Multi-step Return、Distributional RL（C51）和 Noisy Network。这里的 `RainbowAgent` 集成 C51 分布头、Double-DQN 动作选择、Dueling 分支、NoisyLinear 探索，以及带重要性采样权重的 PER 更新。

## 1. C51 的学习目标

不再只估计标量 (Q(s,a))，而是估计回报随机变量的离散分布：

[
Z(s,a)=sum_{i=1}^{N}p_i(s,a)delta_{z_i},quad
z_i=V_{min}+iDelta z.
]

网络输出 logits，经 softmax 得到 (p_i)。动作价值是分布期望：

[
Q(s,a)=mathbb E[Z(s,a)]=sum_i p_i z_i.
]

Double-DQN 选择下一动作时使用 online 网络，评估该动作的分布时使用 target 网络，降低最大化偏差。随后将 Bellman 变换后的分布投影回固定 support，得到目标分布 (m)。

## 2. Dueling 与 NoisyNet

Dueling 将 logits 拆成状态价值和动作优势：

[
l(s,a,z)=v(s,z)+a(s,a,z)-rac1{|mathcal A|}sum_{a'}a(s,a',z).
]

NoisyLinear 使用可学习的噪声尺度参数，在参数空间注入因子化高斯噪声，让策略探索不必完全依赖固定 epsilon。

## 3. PER 的采样和加权损失

优先级通常取 TD / 分布误差的绝对值加小常数：

[
P(i)=rac{p_i^alpha}{sum_jp_j^alpha},qquad
w_i=rac{(N P(i))^{-eta}}{max_j (N P(j))^{-eta}}.
]

C51 的逐样本交叉熵为：

[
ell_i=-sum_z m_i(z)log p_	heta(z|s_i,a_i).
]

训练损失为 (mathcal L=rac1Bsum_iw_iell_i)，并用未加权的逐样本损失更新 replay priority。随着训练推进，通常将 (eta) 从较小值逐渐 anneal 到 1，以减小重要性采样偏差。

## 4. n-step 的边界

n-step 回报为：

[
R_t^{(n)}=sum_{k=0}^{n-1}gamma^k r_{t+k}
+gamma^n(1-d_{t+n})V(s_{t+n}).
]

本仓库的 `NStepAccumulator` 已将连续 transition 合并后写入 PER，并为每条 replay entry 保存 `gamma^k) bootstrap discount。episode 终止时会 flush 队列且将 terminal 标志写入每个跨越终点的前缀；time-limit truncation 也会 flush，但保留非 terminal 标志以便 bootstrap。Rainbow Agent 读取逐样本 discount 做 C51 投影，因此 n-step return 已进入真实训练目标。

## 5. 训练调用

```python
from algorithms.value_based.advanced_dqn import PrioritizedReplayBuffer
from algorithms.value_based.rainbow_agent import RainbowAgent

replay = PrioritizedReplayBuffer(capacity=100_000)
agent = RainbowAgent(observation_dim=4, action_dim=2)
# 每次环境交互后 replay.add(state, action, reward, next_state, terminated)
metrics = agent.update(replay, batch_size=64, beta=0.4)
```

当前 PER 采用 O(N) 概率计算，便于阅读但不适合超大 replay；生产实现可换成 sum tree。这个实现是教学级 Rainbow 组合基线，n-step accumulator、完整 Atari 预处理、分布式 actor/learner 与大规模 benchmark 仍需单独构建。
